"""Assemble the released model pack from the Kaggle model bundles of the final submission (one-off provenance tool; a
retrained solution is packed from its training runs with scripts/90_pack_models.py instead).

The submitted kernel read five bundles: a main tile bundle (10 cells), an override bundle (allfm retrained, allcfm), a
third bundle whose allfm fold set was added as the eleventh cell "allfmold", the ResNet50 OA bundle and the specialist
bundle. Member order is kept exactly (draw key = member position): main bundle order with overrides in place, then the
added cell. Checkpoint files are copied unchanged (sha256 kept).
usage: python tools/assemble_pack_from_bundles.py --main DIR --override DIR --add DIR --oa DIR --spec DIR --recipe JSON --out DIR"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scrub_crop_config import scrub  # noqa: E402

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    for k in ("main", "override", "add", "oa", "spec", "recipe", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--add-cells", default='{"allfmold": "allfm"}')
    a = ap.parse_args()
    out = Path(a.out); (out / "tiles").mkdir(parents=True, exist_ok=True); (out / "keypoints").mkdir(exist_ok=True)
    main_m = json.loads((Path(a.main) / "model-manifest.json").read_text())
    over_m = {m["name"]: m for m in json.loads((Path(a.override) / "model-manifest.json").read_text())["members"]}
    add_m = json.loads((Path(a.add) / "model-manifest.json").read_text())["members"]
    files, members = {}, []

    def take(src_dir, ck, cell):
        dst = f"tiles/pack_{cell}.pt"
        if dst not in files:
            shutil.copy2(Path(src_dir) / ck, out / dst); files[dst] = sha(out / dst)
        return dst

    def member(m, src_dir, cell, name):
        keep = {k: m[k] for k in ("acquisition", "tiles_per_series", "fluid_mult", "other_mult", "slices_per_tile", "attention_tau", "draw_keys") if k in m}
        if "plane" in m:
            keep["plane"] = m["plane"]
        return dict(name=name, cell=cell, fold=int(name.rsplit("_f", 1)[1]), group=m.get("group", "planes"),
                    checkpoint=take(src_dir, m["checkpoint"], cell), checkpoint_key=m["checkpoint_key"], **keep)
    for m in main_m["members"]:
        src, mm = (a.override, over_m[m["name"]]) if m["name"] in over_m else (a.main, m)
        members.append(member(mm, src, m["name"].rsplit("_f", 1)[0], m["name"]))
    for new, old in json.loads(a.add_cells).items():
        for m in add_m:
            if m["name"].rsplit("_f", 1)[0] == old:
                members.append(member(m, a.add, new, new + m["name"][len(old):]))
    kps = []
    for i, f in enumerate(main_m["keypoint_models"]):
        dst = f"keypoints/kp-f{i}.pt"; shutil.copy2(Path(a.main) / f, out / dst); files[dst] = sha(out / dst); kps.append(dst)
    for sub, src, man_name in (("oa", a.oa, "oa-manifest.json"), ("specialists", a.spec, "spec-manifest.json")):
        (out / sub).mkdir(exist_ok=True)
        man = json.loads((Path(src) / man_name).read_text())
        cks = man["checkpoints"] if sub == "oa" else [c for sp in man["specialists"] for c in sp["checkpoints"]]
        for c in cks:
            shutil.copy2(Path(src) / c, out / sub / c); files[f"{sub}/{c}"] = sha(out / sub / c)
        for k in ("sources", "evidence", "parity"):   # internal run paths / notes
            man.pop(k, None)
        if "crop_config" in man:
            man["crop_config"] = scrub(man["crop_config"])
        for sp in man.get("specialists", []):
            sp.pop("sources", None)
            if "crop_config" in sp:
                sp["crop_config"] = scrub(sp["crop_config"])
            if "crop_configs" in sp:
                sp["crop_configs"] = {k: scrub(v) for k, v in sp["crop_configs"].items()}
            for k in ("weights", "offline_stats"):   # blend settings live in the recipe
                sp.pop(k, None)
        man.pop("weights", None); man.pop("offline_stats", None)
        (out / sub / man_name).write_text(json.dumps(man, indent=1) + "\n")
    recipe = json.loads(Path(a.recipe).read_text())
    manifest = dict(schema="rsna-knee-models-v1", target_columns=main_m["target_columns"], prior=main_m["prior"],
                    group_weights=main_m["group_weights"], keypoint_models=kps, members=members, oa="oa", specialists="specialists",
                    recipe=recipe, files=files)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"{len(members)} tile members in {len({m['cell'] for m in members})} cells, {len(files)} files -> {out}")


if __name__ == "__main__":
    main()
