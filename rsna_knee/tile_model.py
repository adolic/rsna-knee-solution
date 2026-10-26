"""Tile model: a bag of 224 px tiles (3 adjacent slices each) -> 12 logits.

tile -> backbone (ConvNeXt-T / CAFormer-S18 / ReXNet-R300, timm; ImageNet weights from DATA/pretrained) -> global average pool
-> dropout -> linear 256 + plane embedding + fluid-sensitive embedding -> LayerNorm = token. Per knee: one transformer encoder
layer over its tokens (planes and series interact), then for each of the 12 targets a gated attention pool over the tokens
(attention logits x tau per target at inference: att_tau) and a linear readout.
Augmentation (training, on the GPU): one affine (rotation +-3 deg, translate 2 %, scale 0.95-1, p 0.75) and one gain / shift
(0.9-1.1 / +-0.04, p 0.8) per knee, applied to all its tiles; optional study-level left-right mirror (cfg augmentation.mirror).
The final models use position "none", pool "attn"; the other cfg switches are kept so the released checkpoints load unchanged.
"""
import functools
import hashlib
import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .paths import PRETRAINED

BACKBONES = {   # name -> (timm name, pinned weights, sha256)
    "radimagenet-resnet10t": ("resnet10t", "radimagenet-resnet10t/backbone.safetensors",
                              "38797da376e1d18144ab984b47e69da746b526c89902e71ff7777dde3bde5510"),
    "radimagenet-resnet18": ("resnet18", "radimagenet-resnet18/backbone.safetensors",
                             "909fab8adbf6db8d50ab6fbd29ee81f840b1eaf21af79f222cbe46aea17caa90"),
    "radimagenet-resnet50": ("resnet50", "radimagenet-resnet50/backbone.safetensors",
                             "ebfced9cede466add6589c3c29f2069f72ac919f700242312dee13406c45ff8b"),
    "convnext-tiny-in12k": ("convnext_tiny", "convnext-tiny-in12k/backbone.safetensors",
                            "538bb4b9f6d551cac3fc4b5e567d24ac49690f32f5f8573a40a556e7df0b3f8d"),
    "caformer-s18-in22k": ("caformer_s18.sail_in22k_ft_in1k", "caformer-s18-in22k/model.safetensors",
                           "4466e9439fda23627c7cf449025ab8881f88b1e7d049975d90cebcda260b8e3e"),
    "rdnet-tiny-in1k": ("rdnet_tiny", "rdnet-tiny-in1k/backbone.safetensors",   # naver-ai rev c720d1a, re-saved (manifest.json)
                        "4b6048fade3e035caf76a7f1c162260e6d99eeeec0d038d7b12fd3bde0778d9f"),
    "convnext-small-in12k": ("convnext_small.in12k_ft_in1k", "convnext-small-in12k/model.safetensors",
                             "eb6f6171daa484fac5e74c28cba90de9328939c03ee96d7e163a0c421898280e"),
    "coatnet-rmlp2-224-in12k": ("coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k", "coatnet-rmlp2-224-in12k/model.safetensors",
                                "8558fd38c0c67be0425f3ca19752a7baa4356314940fa4954633a6395a97d973"),
    "regnety120-in12k": ("regnety_120.sw_in12k_ft_in1k", "regnety120-in12k/model.safetensors",
                         "5f48e987f6bfbf8fabcd84c1700ff47b38504429fe40c0b4b1afb3718a5bf0c0"),
    "rexnetr300": ("rexnetr_300.sw_in12k_ft_in1k", "rexnetr300-224/model.safetensors",
                   "1be2de4c312f1c0ea0cb58e18413d490eeea3dfddd8abc988dc466bc33966821"),
    "rexnetr200": ("rexnetr_200.sw_in12k_ft_in1k", "rexnetr200-224/model.safetensors",
                   "d92c098c499b8482bc824bec5c4c79acb47d82c54aa48285e5b3c39118f3e8fc"),
    "regnety160-in12k": ("regnety_160.sw_in12k_ft_in1k", "regnety160-in12k/model.safetensors",
                         "80da06ab029403fee128c573820b1f67477250709d302a3001a571eccdf613b9"),
    "convnext-tiny-dinov3": ("convnext_tiny.dinov3_lvd1689m", "convnext-tiny-dinov3/model.safetensors",
                             "a7637753a2e85d709d1a5474e13382d39e7bc581a35868b477585c970f36847f"),
    "mnv4-hybrid-medium-in12k": ("mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k", "mnv4-hybrid-medium-in12k/model.safetensors",
                                 "a9d12576199cb225134860a4dad88d1205889a072ecd2fb357881d62ee0369c2"),
}
N_VIEWS, DIM = 12, 256


def adapt_stem(net, c):
    """First conv to c input channels: pretrained RGB kernel mean repeated, scaled by 3/c (same activation scale)."""
    conv = next(m for m in net.modules() if isinstance(m, nn.Conv2d))
    w = conv.weight.data
    new = nn.Conv2d(c, w.shape[0], conv.kernel_size, conv.stride, conv.padding, bias=conv.bias is not None)
    new.weight.data = w.mean(1, keepdim=True).repeat(1, c, 1, 1) * 3 / c
    if conv.bias is not None:
        new.bias.data = conv.bias.data
    for name, m in net.named_modules():
        for cn, ch in m.named_children():
            if ch is conv:
                setattr(m, cn, new); return


def make_backbone(key, pretrained):
    name, rel, digest = BACKBONES[key]
    path = PRETRAINED / rel
    if pretrained and hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"{rel} changed")
    if rel.endswith("backbone.safetensors"):   # re-saved timm backbones, strict load
        net = timm.create_model(name, pretrained=False, num_classes=0, global_pool="")
        if pretrained:
            from safetensors.torch import load_file
            net.load_state_dict(load_file(path), strict=True)
        return net
    # timm pretrained cfg pointed at the pinned local file
    return timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="", pretrained_cfg_overlay={"file": str(path)})


def _width(net):
    """Channels of the backbone's output map, measured (timm num_features is the pre-head width for e.g. MobileNetV4,
    whose conv head still runs with num_classes=0); equals num_features for every earlier backbone."""
    with torch.no_grad():
        was = net.training; net.eval(); c = net(torch.zeros(1, 3, 64, 64)).shape[1]; net.train(was)
    return c


def _ckpt_forward(forward, x):
    return checkpoint(forward, x, use_reentrant=False) if torch.is_grad_enabled() else forward(x)


class BiasedLayer(nn.Module):
    """Pre-norm transformer layer whose attention logits get a learned per-head bias from the 3D distance between two
    tiles' centres (knee frame, mm / k): bias_h(d) = sum_r w_hr exp(-((d - c_r) / width)^2), 8 radial bases over
    0-120, w zero-initialised (starts as plain attention)."""
    def __init__(self, d, heads=4, ff=2, n_rbf=8, dropout=.1):
        super().__init__()
        self.h = heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, ff * d), nn.GELU(), nn.Linear(ff * d, d))
        self.drop, self.p = nn.Dropout(dropout), dropout
        self.register_buffer("centres", torch.linspace(0, 120, n_rbf)); self.width = 120 / n_rbf
        self.rbf_w = nn.Parameter(torch.zeros(heads, n_rbf))

    def forward(self, x, pad, xyz):
        B, L, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, L, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        d = torch.cdist(xyz.float(), xyz.float())                                   # [B, L, L]
        rbf = torch.exp(-((d[..., None] - self.centres) / self.width) ** 2)          # [B, L, L, R]
        bias = torch.einsum("blmr,hr->bhlm", rbf, self.rbf_w).masked_fill(pad[:, None, None, :], float("-inf"))
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype), dropout_p=self.p if self.training else 0.)
        x = x + self.drop(self.o(a.transpose(1, 2).reshape(B, L, D)))
        return x + self.drop(self.ff(self.n2(x)))


class TileModel(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        self.cfg = cfg
        N_TARGETS = cfg.get("n_targets", 12)
        self.backbone = make_backbone(cfg.get("backbone", "radimagenet-resnet10t"), pretrained)
        if cfg.get("in_chans", 3) != 3:
            adapt_stem(self.backbone, cfg["in_chans"])
        if cfg.get("train_pad_bucket") and any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in self.backbone.modules()):
            raise ValueError("train_pad_bucket would let zero padding into BatchNorm statistics")
        if cfg.get("box_aux"):   # training-only per-tile meniscus tear head on the pre-mixer
            # tokens (unused by the final models)
            self.box_head = nn.Linear(DIM, 2)
        if cfg.get("early_checkpointing"):   # checkpoint only the first two (high-resolution) stages: ~60 % of the
            for st in self.backbone.stages[:2]:   # activation memory for ~15 % extra compute (full: ~60 %); keys unchanged
                st.forward = functools.partial(_ckpt_forward, st.forward)
        # dual_backbone: a second, separately trained backbone (same architecture and weights at start) for the
        # base views; self.backbone then serves the patches only
        self.base_views = cfg.get("base_views", [0, 1, 2])
        self.backbone_base = make_backbone(cfg.get("backbone", "radimagenet-resnet10t"), pretrained) if cfg.get("dual_backbone") else None
        # plane_backbones: one backbone per imaging plane (view id 0 sagittal = self.backbone, 1 coronal, 2 axial), same
        # architecture and weights at start; parameter names start with "backbone" (backbone LR group)
        self.backbone_planes = (nn.ModuleList([make_backbone(cfg.get("backbone", "radimagenet-resnet10t"), pretrained) for _ in range(2)])
                                if cfg.get("plane_backbones") else None)
        # backbone2: a second backbone of another architecture over the same tiles; its pooled features are concatenated
        # to the first one's before the projection (parameter names start with "backbone": backbone LR group)
        self.backbone2 = make_backbone(cfg["backbone2"], pretrained) if cfg.get("backbone2") else None
        feat = _width(self.backbone) + (_width(self.backbone2) if self.backbone2 is not None else 0)
        self.register_buffer("mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1))
        self.drop = nn.Dropout(cfg["dropout"])
        self.proj = nn.Linear(feat, DIM)
        # od_dim: a frozen external feature vector per tile, LayerNorm'd and
        # projected into the same token (= concatenation before the projection, like backbone2)
        self.od_dim = int(cfg.get("od_dim", 0))
        if self.od_dim:
            self.od_norm = nn.LayerNorm(self.od_dim); self.od_proj = nn.Linear(self.od_dim, DIM)
        self.view_emb = nn.Embedding(N_VIEWS, DIM); self.fluid_emb = nn.Embedding(2, DIM)
        self.norm = nn.LayerNorm(DIM)
        if cfg.get("depth_embedding"):   # Fourier features of the anatomical depth (mm / knee scale), zero-init
            self.register_buffer("depth_freq", torch.tensor([1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4]) * 2 * math.pi)
            self.depth_proj = nn.Linear(10, DIM); nn.init.zeros_(self.depth_proj.weight); nn.init.zeros_(self.depth_proj.bias)
        self.position = cfg.get("position", "none")
        if self.position in ("corners", "corners_bias"):   # Fourier features of the 15 tile coordinates -> MLP, last layer zero
            self.register_buffer("pos_freq", 2 * math.pi / torch.tensor([8., 16., 32., 64., 128.]))
            self.pos_mlp = nn.Sequential(nn.Linear(150, DIM), nn.GELU(), nn.Linear(DIM, DIM))
            nn.init.zeros_(self.pos_mlp[2].weight); nn.init.zeros_(self.pos_mlp[2].bias)
        self.mixer = (BiasedLayer(DIM) if self.position == "corners_bias" else
                      nn.TransformerEncoderLayer(DIM, 4, 2 * DIM, dropout=.1, batch_first=True, norm_first=True))
        self.att_v = nn.Linear(DIM, DIM); self.att_u = nn.Linear(DIM, DIM)
        self.att_w = nn.Parameter(torch.randn(N_TARGETS, DIM) * .02)
        self.head_w = nn.Parameter(torch.randn(N_TARGETS, DIM) * .02); self.head_b = nn.Parameter(torch.zeros(N_TARGETS))
        # label-confidence token: per (study, target) level -> zero-init scale / shift of
        # that target's pooled vector; training uses each cell's level, inference a fixed level (conf_infer_level, default 0 =
        # confident report agreed by the teacher)
        self.conf_levels = int(cfg.get("conf_levels", 0)); self.conf_infer_level = int(cfg.get("conf_infer_level", 0))
        if self.conf_levels:
            self.conf_g = nn.Parameter(torch.zeros(self.conf_levels, N_TARGETS, DIM)); self.conf_b = nn.Parameter(torch.zeros(self.conf_levels, N_TARGETS, DIM))
        # topk: per target, mix the attention-pooled logit with the mean of the k highest per-token logits (same readout);
        # the mixing weight sigmoid(topk_gate) is learned per target, starting at 0.12
        self.topk = cfg.get("topk", 0)
        # pool "group": attention within each (view, fluid) group of tokens (plane x fluid-sensitive; overview views are
        # their own groups), then a second per-target gated attention over the group summaries plus a learned per-target
        # group bias (zero-init): how many tiles a group holds no longer sets its share of the study-level pool
        self.pool = cfg.get("pool", "attn")
        if self.pool == "group":
            self.n_groups = 2 * N_VIEWS
            self.att2_v = nn.Linear(DIM, DIM); self.att2_u = nn.Linear(DIM, DIM)
            self.att2_w = nn.Parameter(torch.randn(N_TARGETS, DIM) * .02)
            self.att2_b = nn.Parameter(torch.zeros(N_TARGETS, self.n_groups))
        if self.topk:
            self.topk_gate = nn.Parameter(torch.full((N_TARGETS,), -2.))
        for m in (self.view_emb, self.fluid_emb):
            nn.init.normal_(m.weight, std=.02)

    def augment(self, x, lengths, view=None):
        """Affine + intensity. Default: one draw per study (bag), applied to all its windows. cfg "per_window": an
        independent draw per window. Optional: "gamma" [lo, hi], "noise" (max Gaussian sigma, p 0.3), "flips"
        (independent per-window horizontal and vertical flips, p 0.5 each)."""
        a = self.cfg["augmentation"]; s, it = a["spatial"], a["intensity"]
        per_window = a.get("per_window", False)
        n, dev = (len(x) if per_window else len(lengths)), x.device
        u = lambda lo, hi: torch.empty(n, device=dev).uniform_(lo, hi)
        ang = u(-s["rotation_degrees"], s["rotation_degrees"]) * math.pi / 180; sc = u(*s["scale"])
        t = 2 * s["translate_fraction"]
        theta = torch.stack([torch.stack([ang.cos() * sc, -ang.sin() * sc, u(-t, t)], -1),
                             torch.stack([ang.sin() * sc, ang.cos() * sc, u(-t, t)], -1)], 1)
        eye = torch.eye(2, 3, device=dev).expand(n, 2, 3)
        theta = torch.where((torch.rand(n, device=dev) < s["probability"])[:, None, None], theta, eye)
        if a.get("flips"):
            fx = torch.where(torch.rand(n, device=dev) < .5, -1., 1.); fy = torch.where(torch.rand(n, device=dev) < .5, -1., 1.)
            theta = torch.cat([theta[:, :, :2] * torch.stack([fx, fy], 1)[:, None, :], theta[:, :, 2:]], 2)   # mirror output axes
        apply = torch.rand(n, device=dev) < it["probability"]
        gain = torch.where(apply, u(*it["gain"]), torch.ones(n, device=dev))
        shift = torch.where(apply, u(*it["shift"]), torch.zeros(n, device=dev))
        gamma = torch.where(apply, u(*it["gamma"]), torch.ones(n, device=dev)) if "gamma" in it else None
        rep = (lambda v: v) if per_window else (lambda v: v.repeat_interleave(lengths.to(dev), 0))
        grid = F.affine_grid(rep(theta), x.shape, align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        if a.get("mirror"):   # study-level left-right mirror (the cache shows every knee as a right knee, so this shows it as
            # a left knee; labels unchanged): coronal / axial tokens flip image x; sagittal tokens (view % 3 == 0) have
            # left-right along the slice axis, so their 3-slice window reverses its channel order (k+1, k, k-1)
            m = rep(torch.rand(n, device=dev) < a["mirror"])
            sag = (view.to(dev) % 3 == 0) if view is not None else torch.zeros_like(m)
            x = torch.where((m & ~sag)[:, None, None, None], x.flip(-1), x)
            x = torch.where((m & sag)[:, None, None, None], x.flip(1), x)
        if gamma is not None:
            x = x.clamp(0, 1) ** rep(gamma)[:, None, None, None]
        x = x * rep(gain)[:, None, None, None] + rep(shift)[:, None, None, None]
        if it.get("noise"):
            sig = torch.where(torch.rand(n, device=dev) < .3, u(0, it["noise"]), torch.zeros(n, device=dev))
            x = x + torch.randn_like(x) * rep(sig)[:, None, None, None]
        return x

    def tokens(self, images, view, fluid, lengths, chunk=None, depth=None, pos=None):
        x = images.float() / 255.
        if x.shape[1] == 1 and self.cfg.get("in_chans", 3) == 3:   # single-slice tiles: repeat on the GPU
            x = x.expand(-1, 3, -1, -1)
        zoom, size = self.cfg.get("input_zoom", 1.0), self.cfg.get("input_size", x.shape[-1])
        if zoom != 1.0:   # central crop: fraction `zoom` of the patch width (tighter field, same patch centre)
            n = x.shape[-1]; c = int(round(n * zoom)); o = (n - c) // 2
            x = x[..., o:o + c, o:o + c]
        if size != x.shape[-1]:   # resample to the model input size (bicubic, as Lanczos from native)
            x = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False).clamp(0, 1)
        if self.training and self.cfg["augmentation"]["enabled"]:
            x = self.augment(x, lengths, view)
        if x.shape[1] == 3:
            x = (x - self.mean) / self.std
        else:
            x = (x - .449) / .226
        if self.cfg.get("channels_last"):
            x = x.contiguous(memory_format=torch.channels_last)
        if self.backbone_planes is not None:
            f = torch.empty(len(x), self.backbone.num_features, device=x.device, dtype=x.dtype)
            for p, net in enumerate([self.backbone, *self.backbone_planes]):
                sel = view == p
                if sel.any():
                    f = f.index_copy(0, sel.nonzero().squeeze(1), self.features(net, x[sel], chunk).to(f.dtype))
            return self.finish(f, view, fluid, depth, pos)
        if self.backbone_base is not None:
            is_base = torch.isin(view, torch.as_tensor(self.base_views, device=view.device))
            f = torch.empty(len(x), self.backbone.num_features, device=x.device, dtype=x.dtype)
            for net, sel in ((self.backbone_base, is_base), (self.backbone, ~is_base)):
                if sel.any():
                    f = f.index_copy(0, sel.nonzero().squeeze(1), self.features(net, x[sel], chunk).to(f.dtype))
            return self.finish(f, view, fluid, depth, pos)
        f = self.features(self.backbone, x, chunk)
        if self.backbone2 is not None:
            f = torch.cat([f, self.features(self.backbone2, x, chunk).to(f.dtype)], -1)
        g2 = self.cfg.get("cell_grid", 0) ** 2
        if g2:   # one token per cell: repeat the slice's plane / fluid ids (and position) for its cells
            view, fluid = view.repeat_interleave(g2), fluid.repeat_interleave(g2)
            pos = pos.repeat_interleave(g2, 0) if pos is not None else None
        return self.finish(f, view, fluid, depth, pos)

    def features(self, net, x, chunk=None):
        if chunk:   # fixed-size chunks (zero-padded tail) keep one kernel plan in evaluation
            parts = []
            for i in range(0, len(x), chunk):
                c = x[i:i + chunk]; n = len(c)
                if n < chunk:
                    c = torch.cat([c, c.new_zeros((chunk - n, *c.shape[1:]))])
                parts.append(self.pool_cells(net(c))[:n].flatten(0, 1))
            return torch.cat(parts)
        bucket = self.cfg.get("train_pad_bucket", 0)
        if self.training and bucket and len(x) % bucket:   # fixed shapes (see train.py --train-pad-bucket); per-sample norms only
            n = len(x); x = torch.cat([x, x.new_zeros((bucket - n % bucket, *x.shape[1:]))])
            return self.pool_cells(net(x))[:n].flatten(0, 1)
        if self.training and self.cfg.get("grad_checkpointing"):   # chunked activation checkpointing (memory)
            from torch.utils.checkpoint import checkpoint
            step = self.cfg.get("checkpoint_chunk", 128)
            return torch.cat([checkpoint(lambda c: self.pool_cells(net(c)).flatten(0, 1), x[i:i + step], use_reentrant=False)
                              for i in range(0, len(x), step)])
        return self.pool_cells(net(x)).flatten(0, 1)

    def pool_cells(self, fmap):
        """[n, C, H, W] -> [n, cells, C]: global mean (1 cell) or, with cfg "cell_grid" g, a g x g average-pooled grid
        (full-slice inputs: each cell of the slice becomes its own token)."""
        g = self.cfg.get("cell_grid", 0)
        if not g:
            return fmap.mean((2, 3))[:, None]
        return F.adaptive_avg_pool2d(fmap, g).flatten(2).transpose(1, 2)

    def finish(self, f, view, fluid, depth, pos=None):
        z = self.proj(self.drop(f)) + self.view_emb(view) + self.fluid_emb(fluid)
        if self.od_dim:
            z = z + self.od_proj(self.drop(self.od_norm(self._od.float()))).to(z.dtype)
        if self.cfg.get("depth_embedding"):
            a = depth.float()[:, None] * self.depth_freq
            z = z + self.depth_proj(torch.cat([a.sin(), a.cos()], -1))
        if self.position in ("corners", "corners_bias"):
            a = (pos.float()[..., None] * self.pos_freq).flatten(1)                  # [N, 15 x 5]
            e = self.pos_mlp(torch.cat([a.sin(), a.cos()], -1)).to(z.dtype)
            pd = self.cfg.get("pos_dropout", 0.)
            if self.training and pd:   # position dropout: whole embedding off for a random share of tokens
                e = e * (torch.rand(len(e), 1, device=e.device) >= pd).to(e.dtype)
            z = z + e
        return self.norm(z)

    def forward(self, batch, chunk=None):
        lengths = batch["lengths"]
        self._od = batch.get("od")   # read by finish() (per-tile OrthoDiffusion vectors, od_dim models only)
        tok = self.tokens(batch["images"], batch["view"], batch["fluid"], lengths, chunk, batch.get("depth"), batch.get("pos"))
        self._tok = tok if self.cfg.get("box_aux") else None   # tile auxiliary loss (unused by the final models)
        if self.cfg.get("cell_grid"):
            lengths = lengths * self.cfg["cell_grid"] ** 2
        B, L = len(lengths), int(lengths.max())
        pad = torch.zeros(B, L, DIM, device=tok.device, dtype=tok.dtype)
        mask = torch.arange(L, device=tok.device)[None] >= lengths.to(tok.device)[:, None]   # True = padding
        pad[~mask] = tok
        if self.position == "corners_bias":
            xyz = torch.zeros(B, L, 3, device=tok.device)
            xyz[~mask] = batch["pos"][:, :3].float()                                # tile centres
            h = self.mixer(pad, mask, xyz)
        else:
            h = self.mixer(pad, src_key_padding_mask=mask)
        g = torch.tanh(self.att_v(h)) * torch.sigmoid(self.att_u(h))           # [B, L, D]
        e = torch.einsum("bld,td->btl", g, self.att_w) * getattr(self, "att_tau", 1.0)
        if self.pool == "group":
            gid = torch.full((B, L), -1, device=tok.device, dtype=torch.long)
            gid[~mask] = (2 * batch["view"] + batch["fluid"]).to(tok.device)
            member = gid[:, None, :] == torch.arange(self.n_groups, device=tok.device)[None, :, None]       # [B, G, L]
            a = e[:, :, None, :].float().masked_fill(~member[:, None], -1e4).softmax(-1) * member[:, None]  # [B, T, G, L]
            a = a.to(h.dtype)                                    # finite masking: empty groups get all-zero weights
            pg = torch.einsum("btgl,bld->btgd", a, h)                                                       # [B, T, G, D]
            g2 = torch.tanh(self.att2_v(pg)) * torch.sigmoid(self.att2_u(pg))
            e2 = (g2 * self.att2_w[None, :, None]).sum(-1) + self.att2_b                                    # [B, T, G]
            a2 = e2.masked_fill(~member.any(-1)[:, None], float("-inf")).softmax(-1)
            pooled = torch.einsum("btg,btgd->btd", a2, pg)
            self.last_group_attention = a2.detach()                                                         # for analysis
        else:
            a = e.masked_fill(mask[:, None], float("-inf")).softmax(-1)
            pooled = torch.einsum("btl,bld->btd", a, h)
        if self.conf_levels:
            lvl = batch.get("conf_level") if self.training else None
            if lvl is None:
                lvl = torch.full(pooled.shape[:2], self.conf_infer_level, dtype=torch.long, device=pooled.device)
            lvl = lvl.to(pooled.device).long(); tix = torch.arange(pooled.shape[1], device=pooled.device)[None].expand_as(lvl)
            pooled = pooled * (1 + self.conf_g[lvl, tix].to(pooled.dtype)) + self.conf_b[lvl, tix].to(pooled.dtype)
        out = (pooled * self.head_w).sum(-1) + self.head_b
        if self.topk:
            inst = torch.einsum("bld,td->btl", h, self.head_w).masked_fill(mask[:, None], float("-inf"))
            top = inst.topk(min(self.topk, L), -1).values.mean(-1) + self.head_b
            w = torch.sigmoid(self.topk_gate)
            out = (1 - w) * out + w * top
        return out
