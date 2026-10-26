"""OAI baseline knee MRI (Osteoarthritis Initiative via the NIMH Data Archive; requires an NDA data use certification) -> a
Kaggle-style folder, so every later step treats OAI knees exactly like RSNA studies.

Input: the image03 metadata of the OAI collection (image03.txt from the NDA package) and the downloaded image archives (one
.tar.gz per series; NDA "downloadcmd" keeps the s3 key layout, e.g. <root>/.../00m/0.E.1/<subject>/<date>/<id>.tar.gz).
Selection: visit V00, the three series families closest to the RSNA protocol, per knee:
  SAG_IW_TSE (sagittal intermediate-weighted fat-suppressed TSE, fluid sensitive), COR_IW_TSE (coronal IW TSE, not
  fat-suppressed: non-fluid), AX_MPR (axial reformat of the 3D DESS, fluid sensitive).
Study id = oai-<src_subject_id>-V00-<R|L> (one study per knee; the side is used for QC only, never as model input).
Members whose slice plane differs from the family's plane and "PROJECTION IMAGE" members (the MPR archives hold one) are dropped.
The axial DESS reformat is 1.5 mm thick: build its cache with --axial-every 2 (scripts/20_build_cache.py) to match the ~3 mm
RSNA axial series.
Output (--out): oai.csv, oai_series.csv (StudyInstanceUID, SeriesInstanceUID, Fluid_Sensitive, Anatomical_Plane),
oai_series/<knee>/<series>/<member>.dcm. Resumable.
usage: python scripts/11_prepare_oai.py --image03 image03.txt --archives DATA/raw/oai --out DATA/raw/oai-dicom [--limit-knees N]"""
import argparse
import csv
import io
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

FAMILIES = {"SAG_IW_TSE": ("Sagittal", 1), "COR_IW_TSE": ("Coronal", 0), "AX_MPR": ("Axial", 1)}
PLANES = ("Sagittal", "Coronal", "Axial")


def selection(image03):
    d = pd.read_csv(image03, sep="\t", skiprows=[1], low_memory=False, dtype=str)
    d = d[d.visit == "V00"].copy()
    d["family"] = d.image_description.str.rsplit("_", n=1).str[0]
    d["side"] = d.image_description.str.rsplit("_", n=1).str[1].str[0]
    d = d[d.family.isin(FAMILIES) & d.side.isin(["R", "L"])]
    d["knee"] = "oai-" + d.src_subject_id + "-V00-" + d.side
    d["series"] = "oai" + d.image03_id
    d["key"] = d.image_file.str.split("/", n=4).str[4]   # s3://<bucket>/<submission>/<key> -> key
    return d.drop_duplicates("series").sort_values(["knee", "family"])


def find_archive(root, key, index={}):
    if not index:
        for p in Path(root).rglob("*.tar.gz"):
            index[p.as_posix()] = p
    hits = [p for s, p in index.items() if s.endswith(key)]
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image03", required=True); ap.add_argument("--archives", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--limit-knees", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); (out / "oai_series").mkdir(parents=True, exist_ok=True)
    sel = selection(a.image03)
    done = set(pd.read_csv(out / "oai.csv").StudyInstanceUID) if (out / "oai.csv").exists() else set()
    if not done:
        (out / "oai.csv").write_text("StudyInstanceUID\n")
        (out / "oai_series.csv").write_text("StudyInstanceUID,SeriesInstanceUID,Fluid_Sensitive,Anatomical_Plane\n")
    n = 0
    for knee, g in sel.groupby("knee", sort=True):
        if knee in done:
            continue
        rows = []
        for r in g.itertuples():
            plane, fluid = FAMILIES[r.family]
            arc = find_archive(a.archives, r.key)
            if arc is None:
                continue
            d = out / "oai_series" / knee / r.series; d.mkdir(parents=True, exist_ok=True); kept = 0
            with tarfile.open(arc, "r:gz") as tar:
                for m in tar:
                    if not m.isfile():
                        continue
                    data = tar.extractfile(m).read()
                    try:
                        ds = pydicom.dcmread(io.BytesIO(data), stop_before_pixels=True)
                        o = np.asarray(ds.ImageOrientationPatient, float)
                        if PLANES[int(np.argmax(np.abs(np.cross(o[:3], o[3:]))))] != plane or "PROJECTION IMAGE" in map(str, ds.get("ImageType", [])):
                            continue
                    except Exception:
                        pass   # unreadable header: kept; the reader records it as a bad slice
                    (d / (Path(m.name).name or f"{kept:05d}")).with_suffix(".dcm").write_bytes(data); kept += 1
            if kept:
                rows.append([knee, r.series, fluid, plane])
        if rows:
            with (out / "oai_series.csv").open("a", newline="") as fh:
                csv.writer(fh).writerows(rows)
            with (out / "oai.csv").open("a") as fh:
                fh.write(knee + "\n")
        n += 1
        if a.limit_knees and n >= a.limit_knees:
            break
    print(f"{n} knees written to {out}")


if __name__ == "__main__":
    main()
