"""Keep only the functional fields of a crop config (view geometry / sampling and the numeric planning rules); descriptive
notes are dropped. View order is preserved (models index views by position).
usage: python tools/scrub_crop_config.py FILE.json [...]   (in place)"""
import json
import sys
from pathlib import Path

VIEW_KEYS = ("plane", "anchor", "offset", "width_mm", "scale_width", "slab_mm", "slab_axis", "slab_range", "slab_range_m",
             "sigma_mm", "train_budget", "infer_cap")
RULE_KEYS = ("output_px", "triplet_step_mm", "min_triplets")


def scrub(cfg):
    return dict(name=cfg["name"], output_px=cfg.get("output_px"),
                rules={k: cfg["rules"][k] for k in RULE_KEYS if k in cfg["rules"]},
                views={n: {k: v[k] for k in VIEW_KEYS if k in v} for n, v in cfg["views"].items()})


if __name__ == "__main__":
    for f in sys.argv[1:]:
        p = Path(f); p.write_text(json.dumps(scrub(json.loads(p.read_text())), indent=1) + "\n"); print("scrubbed", p)
