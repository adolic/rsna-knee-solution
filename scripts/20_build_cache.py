"""Build a preprocessed cache (native224 + keypoint knee frames + knee512 canvases) from a Kaggle-style DICOM folder.

  RSNA:     python scripts/20_build_cache.py --input DATA/raw/rsna --split train --out DATA/cache/rsna --keypoints RUNS/kp-f{0..4}/last.pt
  OAI:      python scripts/20_build_cache.py --input DATA/raw/oai-dicom --split oai --out DATA/cache/oai --axial-every 2 ...
  fastMRI:  python scripts/20_build_cache.py --input DATA/raw/fastmri-dicom --split fastmri --out DATA/cache/fastmri ...
  (OAI / fastMRI folders are made by scripts/11_prepare_oai.py / 12_prepare_fastmri.py.)
--native-only writes native224 only (the input of keypoint training, before keypoint models exist; then use
--keypoints with any existing models or none: frames need models). The keypoint models default to the model pack's.
Resumes: studies already in <out>/rows/ are skipped; --merge-only rebuilds the CSVs."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsna_knee import cache  # noqa: E402
from rsna_knee.paths import MODELS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True); ap.add_argument("--split", default="train"); ap.add_argument("--out", required=True)
    ap.add_argument("--studies", nargs="*"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--keypoints", nargs="*", help="keypoint localizer checkpoints (default: the model pack's five)")
    ap.add_argument("--devices", default="cuda:0"); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--axial-every", type=int, default=1, help="keep every k-th axial slice of the canvases (OAI: 2)")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--native-only", action="store_true", help="bootstrap: native224 volumes only (keypoint training input)")
    a = ap.parse_args()
    if a.merge_only:
        cache.merge(a.out); return
    kps = a.keypoints or sorted(str(p) for p in (MODELS / "keypoints").glob("kp-f*.pt"))
    if not kps and not a.native_only:
        raise SystemExit("no keypoint models: pass --keypoints or put a model pack in DATA/models")
    jobs = cache.jobs_from_folder(a.input, a.split, a.studies)
    if a.limit:
        jobs = jobs[:a.limit]
    cache.build(a.out, jobs, kps, a.devices.split(","), a.workers, axial_every=a.axial_every, native_only=a.native_only)


if __name__ == "__main__":
    main()
