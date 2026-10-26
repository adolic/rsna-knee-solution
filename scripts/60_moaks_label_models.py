"""Label refinement with expert OAI reads (docs/LABELS.md): image models trained ONLY on the MOAKS reads of the OAI baseline
knees, applied zero-shot to every RSNA training study, and blended into the report-derived soft labels.

  --stage targets   MOAKS targets from the NDA file oai_kmrisemiquant01.txt (rsna_knee.moaks): DATA/moaks/moaks-<kind>-{labels,weights}.csv
                    for kind = syneff (effusion / Hoffa synovitis), oa (cartilage loss per compartment), menbak (meniscal tears,
                    popliteal cyst, ACL); knees restricted to the OAI cache
  --stage train     one tile model per kind (all planes, ConvNeXt-T, 20 epochs, final EMA weights last.pt; validation on RSNA fold 1
                    is reported but never selects the checkpoint) -> DATA/runs/moaks-<kind>
  --stage predict   zero-shot probabilities of every head for every RSNA study (the validation bags: 96 tiles per series)
                    -> DATA/moaks/<kind>-zeroshot.csv
  --stage blend     --base SOFT.csv -> --out SOFT.csv: (1 - w) x base + w x MOAKS head, per BLEND below; the 58 expert-labelled
                    Gold studies keep their labels. The released labels/soft.csv already contains these blends (applied in three
                    steps on the report-derived labels); this stage documents and reproduces the operation on any base.
usage: python scripts/60_moaks_label_models.py --stage targets --nda PATH/oai_kmrisemiquant01.txt
       python scripts/60_moaks_label_models.py --stage train|predict [--kinds syneff oa menbak]
       python scripts/60_moaks_label_models.py --stage blend --base base-soft.csv --out new-soft.csv"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from rsna_knee.paths import CACHE, DATA, LABELS, RUNS, TARGETS  # noqa: E402

KINDS = ("syneff", "oa", "menbak")
BLEND = {   # target: (kind, head, weight); evidence in docs/LABELS.md
    "Effusion": ("syneff", "eff_ge2", 0.50),          # P(MOAKS whole-knee effusion grade >= 2)
    "Synovitis": ("syneff", "syic_ge2", 0.25),        # P(MOAKS Hoffa synovitis grade >= 2)
    "Medial OA": ("oa", "m_a2", 0.25),                # P(any medial subregion with cartilage loss over >= 10 % of it)
    "Medial Meniscus": ("menbak", "Medial Meniscus", 0.25),   # P(any medial meniscal tear)
    "Baker's": ("menbak", "Baker's", 0.25),           # P(popliteal cyst)
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("targets", "train", "predict", "blend"))
    ap.add_argument("--kinds", nargs="*", default=list(KINDS)); ap.add_argument("--nda")
    ap.add_argument("--base"); ap.add_argument("--out"); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out_dir = DATA / "moaks"; out_dir.mkdir(parents=True, exist_ok=True)
    if a.stage == "targets":
        from rsna_knee import moaks
        knees = set(pd.read_csv(CACHE / "oai" / "knee_frame.csv").study_uid) if (CACHE / "oai" / "knee_frame.csv").exists() else None
        for kind in a.kinds:
            lab, _ = moaks.write(a.nda, kind, out_dir, knees)
            print(f"{kind}: {len(lab)} knees, reads per column {lab.notna().sum()[lab.notna().sum() > 0].to_dict()}")
    elif a.stage == "train":
        for kind in a.kinds:
            run = RUNS / ("smoke" if a.smoke else "") / f"moaks-{kind}"
            cmd = [sys.executable, "-m", "rsna_knee.train_tiles", "--out", str(run), "--fold", "1", "--cache", str(CACHE / "rsna"),
                   "--view", "all", "--train-cache", str(CACHE / "oai"), "--train-labels-file", str(out_dir / f"moaks-{kind}-labels.csv"),
                   "--train-weights-file", str(out_dir / f"moaks-{kind}-weights.csv"), "--backbone", "convnext-tiny-in12k",
                   "--tiles-per-series", "12", "--other-mult", "0.5", "--epochs", "20", "--freeze-backbone-epochs", "2",
                   "--seed", "20261020", "--save-last"] + (["--epochs", "2", "--limit-train", "24", "--limit-val", "40", "--workers", "2"] if a.smoke else [])
            print(" ".join(cmd), flush=True); subprocess.run(cmd, cwd=REPO, check=True)
    elif a.stage == "predict":
        import torch
        from rsna_knee.labels import labels
        from rsna_knee.tile_data import TileBags, knees as knee_map
        from rsna_knee.tile_model import TileModel
        from rsna_knee.train_tiles import predict
        km = knee_map(CACHE / "rsna", "all"); uids = sorted(km)[:40] if a.smoke else sorted(km)
        for kind in a.kinds:
            run = RUNS / ("smoke" if a.smoke else "") / f"moaks-{kind}"
            st = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
            model = TileModel(st["cfg"]).cuda().to(memory_format=torch.channels_last); model.load_state_dict(st["model"]); model.eval()
            ds = TileBags(uids, km, labels(uids, "hard"), labels(uids, "weights"), False, val_tiles_per_series=96, slices_per_tile=3,
                          fluid_mult=1.0, other_mult=0.5)
            with torch.inference_mode():
                z = predict(model, ds, 6)
            cols = list(TARGETS) + list(st["cfg"].get("extra_targets", []))
            pd.DataFrame(1 / (1 + np.exp(-z)), index=pd.Index(uids, name="StudyInstanceUID"), columns=cols).to_csv(out_dir / f"{kind}-zeroshot.csv", float_format="%.6f")
            print(f"{kind}: {len(uids)} studies -> {out_dir / f'{kind}-zeroshot.csv'}")
    else:
        base = pd.read_csv(a.base, index_col=0); new = base.copy()
        gold = pd.read_csv(LABELS / "folds.csv", index_col=0).is_gold.reindex(base.index).fillna(0).astype(bool)
        for t, (kind, head, w) in BLEND.items():
            p = pd.read_csv(out_dir / f"{kind}-zeroshot.csv", index_col=0)[head].reindex(base.index)
            ok = p.notna() & ~gold
            new.loc[ok, t] = (1 - w) * base.loc[ok, t] + w * p[ok]
            print(f"{t:16s} <- {kind}:{head} x {w}: {int(ok.sum())} studies, mean {base[t].mean():.3f} -> {new[t].mean():.3f}")
        new.to_csv(a.out, float_format="%.6f"); print(f"-> {a.out}")


if __name__ == "__main__":
    main()
