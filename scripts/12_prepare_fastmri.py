"""fastMRI knee DICOMs (NYU clinical exams; requires the fastMRI data sharing agreement) -> a Kaggle-style folder, so every
later step (cache building, pseudo labelling) treats them exactly like RSNA studies.

Input: the knee DICOM archives as downloaded from the fastMRI site (knee_DICOMs_batch1.tar.xz, knee_DICOMs_batch2.tar.xz),
layout <batch dir>/<patient dir>/study_<id>/<series dir>/<file>.dcm. They are read as one sequential stream (xz decodes in
parallel), nothing else is unpacked.
Selection per series: slices of the series' majority plane only (others dropped), at least 10 slices. Fluid-sensitive flag
from the SeriesDescription: fat-saturated, T2 or STIR ("fs" as its own token, "fat sat", "ftst", "t2", "stir"; "fse" alone is
a fast spin echo, not fat-sat).
Ids carry no PHI: study = fastmri-b<batch>-<sha256(patient dir)[:12]>-<study id>, series = <study>-<series dir>.
Output (--out): fastmri.csv (StudyInstanceUID), fastmri_series.csv (StudyInstanceUID, SeriesInstanceUID, Fluid_Sensitive,
Anatomical_Plane, SeriesDescription), fastmri_series/<study>/<series>/<nnnn>.dcm. Resumable (finished studies are skipped).
usage: python scripts/12_prepare_fastmri.py --archives DATA/raw/fastmri/knee_DICOMs_batch1.tar.xz DATA/raw/fastmri/knee_DICOMs_batch2.tar.xz
                                            --out DATA/raw/fastmri-dicom [--limit-studies N]"""
import argparse
import csv
import hashlib
import io
import re
import subprocess
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np
import pydicom

PLANES = ("Sagittal", "Coronal", "Axial")
FLUID = re.compile(r"(?<![a-z])fs(?![a-z])|fat[ -]?sat|ftst|t2|stir")


def members(archive):
    proc = subprocess.Popen(["xz", "-T0", "-dc", str(archive)], stdout=subprocess.PIPE, bufsize=1 << 22)
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
            for m in tar:
                if m.isfile() and m.name.endswith(".dcm"):
                    parts = m.name.split("/")
                    if len(parts) == 5:
                        yield parts, tar.extractfile(m).read()
    finally:
        proc.stdout.close(); proc.wait()


def studies(archive, batch):
    """(study id, {series id: [(name, bytes)]}) per study, in archive order (a study's members are contiguous)."""
    cur, series = None, {}
    for parts, data in members(archive):
        s = f"fastmri-b{batch}-{hashlib.sha256(parts[1].encode()).hexdigest()[:12]}-{parts[2].replace('study_', '')}"
        if s != cur:
            if cur is not None and series:
                yield cur, series
            cur, series = s, {}
        series.setdefault(f"{s}-{parts[3]}", []).append((parts[4], data))
    if cur is not None and series:
        yield cur, series


def plane_of(ds):
    o = np.asarray(ds.ImageOrientationPatient, float)
    return PLANES[int(np.argmax(np.abs(np.cross(o[:3], o[3:]))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archives", nargs="+", required=True, help="knee_DICOMs_batch<N>.tar.xz (batch number from the file name)")
    ap.add_argument("--out", required=True); ap.add_argument("--limit-studies", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); (out / "fastmri_series").mkdir(parents=True, exist_ok=True)
    series_csv, studies_csv = out / "fastmri_series.csv", out / "fastmri.csv"
    done = set()
    if studies_csv.exists():
        done = {r["StudyInstanceUID"] for r in csv.DictReader(studies_csv.open())}
    else:
        studies_csv.write_text("StudyInstanceUID\n")
        series_csv.write_text("StudyInstanceUID,SeriesInstanceUID,Fluid_Sensitive,Anatomical_Plane,SeriesDescription\n")
    n = 0
    for archive in a.archives:
        batch = int(re.search(r"batch(\d+)", Path(archive).name).group(1))
        for study, series in studies(archive, batch):
            if study in done:
                continue
            rows = []
            for uid, files in series.items():
                heads = []
                for name, data in files:
                    try:
                        heads.append(pydicom.dcmread(io.BytesIO(data), stop_before_pixels=True))
                    except Exception:
                        heads.append(None)
                planes = [plane_of(h) if h is not None and "ImageOrientationPatient" in h else None for h in heads]
                major = Counter(p for p in planes if p).most_common(1)
                if not major:
                    continue
                major = major[0][0]
                keep = [(nm, d) for (nm, d), p in zip(files, planes) if p == major]
                if len(keep) < 10:
                    continue
                desc = next((str(getattr(h, "SeriesDescription", "")) for h in heads if h is not None), "")
                d = out / "fastmri_series" / study / uid; d.mkdir(parents=True, exist_ok=True)
                for nm, data in keep:
                    (d / nm).write_bytes(data)
                rows.append([study, uid, int(bool(FLUID.search(desc.lower()))), major, desc.replace(",", " ")])
            if rows:   # the study line last: a study is complete once it is listed in fastmri.csv
                with series_csv.open("a", newline="") as fh:
                    csv.writer(fh).writerows(rows)
                with studies_csv.open("a") as fh:
                    fh.write(study + "\n")
            n += 1
            if n % 100 == 0:
                print(f"{n} studies", flush=True)
            if a.limit_studies and n >= a.limit_studies:
                return


if __name__ == "__main__":
    main()
