"""Download the ImageNet / RadImageNet backbone weights the models start from, pinned and hash-checked, into DATA/pretrained.

  convnext-tiny-in12k   timm convnext_tiny.in12k_ft_in1k, re-saved as a headless backbone (backbone.safetensors)
  caformer-s18-in22k    timm caformer_s18.sail_in22k_ft_in1k (model.safetensors)
  rexnetr300-224        timm rexnetr_300.sw_in12k_ft_in1k (model.safetensors)
  radimagenet-resnet50  RadImageNet ResNet50 (convergedmachine/RadImagenet resnet50.pth) -> headless timm resnet50 backbone
The keypoint localizer's EfficientNet-B0 (timm efficientnet_b0.ra_in1k) is fetched by timm itself when training starts.
Every file must match the sha256 the solution was trained with (tile_model.BACKBONES / crop_model.BACKBONES).
usage: python scripts/00_download_pretrained.py"""
import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from rsna_knee.paths import PRETRAINED  # noqa: E402
from rsna_knee.tile_model import BACKBONES  # noqa: E402

PINNED = {   # key: (hub repo, file, revision)
    "convnext-tiny-in12k": ("timm/convnext_tiny.in12k_ft_in1k", "model.safetensors", "main"),
    "caformer-s18-in22k": ("timm/caformer_s18.sail_in22k_ft_in1k", "model.safetensors", "cc6467f00ce5f7c89ad66e02900bc17e6df2c135"),
    "rexnetr300": ("timm/rexnetr_300.sw_in12k_ft_in1k", "model.safetensors", "524b9020be9649c58f7b4c48e52def228b97ea08"),
    "radimagenet-resnet50": ("convergedmachine/RadImagenet", "resnet50.pth", "7fa62dccee2b14e49188d11c4b0e66d88c29b41e"),
}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def headless(timm_name, state):
    """A timm backbone (num_classes=0, global_pool='') loaded strictly from `state` (classifier keys dropped)."""
    import timm
    net = timm.create_model(timm_name, pretrained=False, num_classes=0, global_pool="")
    own = net.state_dict()
    net.load_state_dict({k: v for k, v in state.items() if k in own}, strict=True)
    return {k: v.contiguous() for k, v in net.state_dict().items()}


def main():
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file, save_file
    import torch
    for key, (repo, fname, rev) in PINNED.items():
        timm_name, rel, digest = BACKBONES[key]
        dst = PRETRAINED / rel; dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and sha(dst) == digest:
            print(f"ok      {key}"); continue
        src = hf_hub_download(repo, fname, revision=rev)
        if rel.endswith("model.safetensors"):
            dst.write_bytes(Path(src).read_bytes())
        elif fname.endswith(".safetensors"):
            save_file(headless(timm_name, load_file(src)), str(dst))
        else:   # torchvision-reference checkpoint: {"model": state_dict, "args": Namespace, ...}, possibly torch.compile prefixed
            import argparse
            torch.serialization.add_safe_globals([argparse.Namespace])
            ck = torch.load(src, map_location="cpu", weights_only=True)
            state = {k.replace("_orig_mod.", ""): v for k, v in ck.get("model", ck).items()}
            save_file(headless(timm_name, state), str(dst))
        got = sha(dst)
        print(f"{'ok' if got == digest else 'MISMATCH':7s} {key} {dst} {got[:16]}")
        if got != digest:
            raise SystemExit(f"{key}: sha256 {got} != expected {digest} (different upstream revision?)")


if __name__ == "__main__":
    main()
