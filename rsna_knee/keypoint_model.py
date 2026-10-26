"""EfficientNet-B0 heatmap localizer: FPN decoder to stride 4 (56 x 56), 5 landmark heatmaps per plane (plane-specific
heads on a shared backbone), soft-argmax coordinates, and a per-plane knee-side logit (P(left knee))."""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

N_LM, N_PLANES, GRID, STRIDE = 5, 3, 56, 4
MEAN, STD = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])


class Localizer(nn.Module):
    def __init__(self, backbone="efficientnet_b0.ra_in1k", pretrained=True, width=96):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, features_only=True, out_indices=(1, 2, 3, 4))
        ch = self.backbone.feature_info.channels()
        self.lateral = nn.ModuleList(nn.Conv2d(c, width, 1) for c in ch)
        self.refine = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width), nn.SiLU(),
                                    nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width), nn.SiLU())
        self.heat = nn.Conv2d(width, N_PLANES * N_LM, 1)
        self.side = nn.Linear(ch[-1], N_PLANES)
        self.register_buffer("mean", MEAN.view(1, 3, 1, 1)); self.register_buffer("std", STD.view(1, 3, 1, 1))
        centres = (torch.arange(GRID).float() + .5) * STRIDE - .5            # cell centre in 224 px coordinates
        self.register_buffer("cx", centres)

    def forward(self, image, plane):
        x = (image - self.mean) / self.std
        feats = self.backbone(x)
        p = self.lateral[-1](feats[-1])
        for f, lat in zip(feats[-2::-1], list(self.lateral)[-2::-1]):
            p = F.interpolate(p, size=f.shape[-2:], mode="bilinear", align_corners=False) + lat(f)
        b = image.shape[0]
        logits = self.heat(self.refine(p)).view(b, N_PLANES, N_LM, GRID, GRID)[torch.arange(b), plane]
        side = self.side(feats[-1].mean((2, 3)))[torch.arange(b), plane]
        return logits.float(), side.float()

    def coords(self, logits):
        """Soft-argmax: (B, L, 56, 56) logits -> (B, L, 2) expected (i, j) in 224 px."""
        prob = logits.flatten(2).softmax(-1).view_as(logits)
        i = (prob.sum(2) * self.cx).sum(-1)
        j = (prob.sum(3) * self.cx).sum(-1)
        return torch.stack([i, j], -1), prob
