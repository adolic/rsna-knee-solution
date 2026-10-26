"""Crop model: a bag of 3-slice windows of keypoint-centred crops (several views) -> one logit per output.

window -> backbone (ConvNeXt-T or RadImageNet ResNet50) -> global average pool -> dropout -> linear 256 + view embedding +
fluid-sensitive embedding -> LayerNorm = token; one transformer encoder layer over the knee's tokens, then per output a gated
attention pool and a linear readout. Augmentation as tile_model.py (one affine + gain / shift draw per knee).
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
    "convnext-small-in12k": ("convnext_small.in12k_ft_in1k", "convnext-small-in12k/model.safetensors",
                             "eb6f6171daa484fac5e74c28cba90de9328939c03ee96d7e163a0c421898280e"),
    "coatnet-rmlp2-224-in12k": ("coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k", "coatnet-rmlp2-224-in12k/model.safetensors",
                                "8558fd38c0c67be0425f3ca19752a7baa4356314940fa4954633a6395a97d973"),
    "regnety120-in12k": ("regnety_120.sw_in12k_ft_in1k", "regnety120-in12k/model.safetensors",
                         "5f48e987f6bfbf8fabcd84c1700ff47b38504429fe40c0b4b1afb3718a5bf0c0"),
    "rexnetr300": ("rexnetr_300.sw_in12k_ft_in1k", "rexnetr300-224/model.safetensors",
                   "1be2de4c312f1c0ea0cb58e18413d490eeea3dfddd8abc988dc466bc33966821"),
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


def _ckpt_forward(forward, x):
    return checkpoint(forward, x, use_reentrant=False) if torch.is_grad_enabled() else forward(x)


class CropModel(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        self.cfg = cfg
        N_TARGETS = cfg.get("n_targets", 12)
        self.backbone = make_backbone(cfg.get("backbone", "radimagenet-resnet10t"), pretrained)
        if cfg.get("in_chans", 3) != 3:
            adapt_stem(self.backbone, cfg["in_chans"])
        if cfg.get("early_checkpointing"):   # checkpoint only the first two (high-resolution) stages: ~60 % of the
            for st in self.backbone.stages[:2]:   # activation memory for ~15 % extra compute (full: ~60 %); keys unchanged
                st.forward = functools.partial(_ckpt_forward, st.forward)
        # dual_backbone: a second, separately trained backbone (same architecture and weights at start) for the
        # base views; self.backbone then serves the patches only
        self.base_views = cfg.get("base_views", [0, 1, 2])
        self.backbone_base = make_backbone(cfg.get("backbone", "radimagenet-resnet10t"), pretrained) if cfg.get("dual_backbone") else None
        feat = self.backbone.num_features
        self.register_buffer("mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1))
        self.drop = nn.Dropout(cfg["dropout"])
        self.proj = nn.Linear(feat, DIM)
        self.view_emb = nn.Embedding(cfg.get("n_views", N_VIEWS), DIM); self.fluid_emb = nn.Embedding(2, DIM)
        self.norm = nn.LayerNorm(DIM)
        if cfg.get("depth_embedding"):   # Fourier features of the anatomical depth (mm / knee scale), zero-init
            self.register_buffer("depth_freq", torch.tensor([1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4]) * 2 * math.pi)
            self.depth_proj = nn.Linear(10, DIM); nn.init.zeros_(self.depth_proj.weight); nn.init.zeros_(self.depth_proj.bias)
        self.mixer = nn.TransformerEncoderLayer(DIM, 4, 2 * DIM, dropout=.1, batch_first=True, norm_first=True)
        self.att_v = nn.Linear(DIM, DIM); self.att_u = nn.Linear(DIM, DIM)
        self.att_w = nn.Parameter(torch.randn(N_TARGETS, DIM) * .02)
        self.head_w = nn.Parameter(torch.randn(N_TARGETS, DIM) * .02); self.head_b = nn.Parameter(torch.zeros(N_TARGETS))
        for m in (self.view_emb, self.fluid_emb):
            nn.init.normal_(m.weight, std=.02)

    def augment(self, x, lengths):
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
        if gamma is not None:
            x = x.clamp(0, 1) ** rep(gamma)[:, None, None, None]
        x = x * rep(gain)[:, None, None, None] + rep(shift)[:, None, None, None]
        if it.get("noise"):
            sig = torch.where(torch.rand(n, device=dev) < .3, u(0, it["noise"]), torch.zeros(n, device=dev))
            x = x + torch.randn_like(x) * rep(sig)[:, None, None, None]
        return x

    def tokens(self, images, view, fluid, lengths, chunk=None, depth=None):
        x = images.float() / 255.
        zoom, size = self.cfg.get("input_zoom", 1.0), self.cfg.get("input_size", x.shape[-1])
        if zoom != 1.0:   # central crop: fraction `zoom` of the patch width (tighter field, same patch centre)
            n = x.shape[-1]; c = int(round(n * zoom)); o = (n - c) // 2
            x = x[..., o:o + c, o:o + c]
        if size != x.shape[-1]:   # resample to the model input size (bicubic, as Lanczos from native)
            x = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False).clamp(0, 1)
        if self.training and self.cfg["augmentation"]["enabled"]:
            x = self.augment(x, lengths)
        if x.shape[1] == 3:
            x = (x - self.mean) / self.std
        else:
            x = (x - .449) / .226
        if self.cfg.get("channels_last"):
            x = x.contiguous(memory_format=torch.channels_last)
        if self.backbone_base is not None:
            is_base = torch.isin(view, torch.as_tensor(self.base_views, device=view.device))
            f = torch.empty(len(x), self.backbone.num_features, device=x.device, dtype=x.dtype)
            for net, sel in ((self.backbone_base, is_base), (self.backbone, ~is_base)):
                if sel.any():
                    f = f.index_copy(0, sel.nonzero().squeeze(1), self.features(net, x[sel], chunk).to(f.dtype))
            return self.finish(f, view, fluid, depth)
        return self.finish(self.features(self.backbone, x, chunk), view, fluid, depth)

    def features(self, net, x, chunk=None):
        if chunk:   # fixed-size chunks (zero-padded tail) keep one kernel plan in evaluation
            parts = []
            for i in range(0, len(x), chunk):
                c = x[i:i + chunk]; n = len(c)
                if n < chunk:
                    c = torch.cat([c, c.new_zeros((chunk - n, *c.shape[1:]))])
                parts.append(net(c).mean((2, 3))[:n])
            return torch.cat(parts)
        if self.training and self.cfg.get("grad_checkpointing"):   # chunked activation checkpointing (memory)
            from torch.utils.checkpoint import checkpoint
            step = self.cfg.get("checkpoint_chunk", 128)
            return torch.cat([checkpoint(lambda c: net(c).mean((2, 3)), x[i:i + step], use_reentrant=False)
                              for i in range(0, len(x), step)])
        return net(x).mean((2, 3))

    def finish(self, f, view, fluid, depth):
        z = self.proj(self.drop(f)) + self.view_emb(view) + self.fluid_emb(fluid)
        if self.cfg.get("depth_embedding"):
            a = depth.float()[:, None] * self.depth_freq
            z = z + self.depth_proj(torch.cat([a.sin(), a.cos()], -1))
        return self.norm(z)

    def forward(self, batch, chunk=None):
        lengths = batch["lengths"]
        tok = self.tokens(batch["images"], batch["view"], batch["fluid"], lengths, chunk, batch.get("depth"))
        B, L = len(lengths), int(lengths.max())
        pad = torch.zeros(B, L, DIM, device=tok.device, dtype=tok.dtype)
        mask = torch.arange(L, device=tok.device)[None] >= lengths.to(tok.device)[:, None]   # True = padding
        pad[~mask] = tok
        h = self.mixer(pad, src_key_padding_mask=mask)
        g = torch.tanh(self.att_v(h)) * torch.sigmoid(self.att_u(h))           # [B, L, D]
        a = torch.einsum("bld,td->btl", g, self.att_w).masked_fill(mask[:, None], float("-inf")).softmax(-1)
        pooled = torch.einsum("btl,bld->btd", a, h)
        return (pooled * self.head_w).sum(-1) + self.head_b
