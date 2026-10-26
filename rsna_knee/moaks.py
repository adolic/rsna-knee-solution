"""Expert MOAKS reads of the OAI baseline knees -> training targets for the label-refinement models (docs/LABELS.md).

Source: the NIMH Data Archive structure oai_kmrisemiquant01 (MRI semi-quantitative reads, one row per read: subject, visit,
side 1 = right / 2 = left, reading project, reader). Per knee (study id oai-<subject>-V00-<R|L>) and field, every baseline
read counts equally: P(code >= k) = share of the knee's reads with code >= k (fields without a read: no target, weight 0).
MOAKS cartilage morphology per subregion is coded area.full_thickness (e.g. 2.1 = area grade 2, full-thickness grade 1;
grades 0 / <10 % / 10-33 % / >33 % of the subregion). "any" over subregions = 1 - prod(1 - p) (unread subregions add nothing).
Target sets (12 RSNA columns first, NaN = no loss; extra columns are extra outputs used only during this training):
  syneff  Effusion = E[whole-knee effusion grade] / 3, Synovitis = E[Hoffa synovitis grade] / 3; extras P(grade >= 1), P(>= 2)
  oa      Medial / Lateral / PF OA = any subregion of the compartment with full-thickness loss (>= 1) and area >= 10 % (>= 2);
          extras per compartment: any full thickness >= 1 / >= 2, any area >= 2 / >= 3 (m_a2 = the head blended into Medial OA)
  menbak  Medial / Lateral Meniscus = any tear (morphology >= 2 in anterior horn / body / posterior horn, or a posterior root
          tear); Baker's = popliteal cyst (MOAKS, else WORMS >= 1); ACL = tear >= 1; extras: posterior-horn tear, maceration,
          signal (>= 1), complete ACL tear."""
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import TARGETS

TN = list(TARGETS)
CARTILAGE = {"Medial OA": ("m", ["m_cm_cmf", "m_cm_pmf", "m_cm_amt", "m_cm_cmt", "m_cm_pmt"]),
             "Lateral OA": ("l", ["m_cm_clf", "m_cm_plf", "m_cm_alt", "m_cm_clt", "m_cm_plt"]),
             "PF OA": ("pf", ["m_cm_alf", "m_cm_amf", "m_cm_lp", "m_cm_mp"])}


def read_nda(path):
    """Baseline (V00) reads with a knee id; the second header line of NDA text files is a description row."""
    d = pd.read_csv(path, sep="\t", skiprows=[1], low_memory=False)
    d = d[d.visit == "V00"].copy()
    d["knee"] = "oai-" + d.src_subject_id.astype(str) + "-V00-" + d.side.map({1: "R", 2: "L"})
    return d[d.knee.notna()]


def pge(d, field, k, part=None):
    """Per knee: (P(code >= k) over the knee's reads of `field`, read?) ; part = 'area' / 'ft' for cartilage codes."""
    v = d[["knee", field]].dropna()
    x = v[field].astype(float)
    if part == "area":
        x = np.floor(x + 1e-9)
    elif part == "ft":
        x = np.round((x - np.floor(x + 1e-9)) * 10)
    return (x >= k).groupby(v.knee).mean()


def anyof(series):
    """1 - prod(1 - p) over the parts, unread parts contributing nothing; NaN where no part was read."""
    df = pd.concat(series, axis=1)
    return (1 - (1 - df.fillna(0)).prod(axis=1)).where(df.notna().any(axis=1))


def targets(path, kind):
    """-> (labels, weights) DataFrames indexed by knee id, columns = 12 RSNA targets + the set's extra columns."""
    d = read_nda(path); cols = {}
    if kind == "syneff":
        for field, col, pre in (("m_syic", "Synovitis", "syic"), ("m_eff_wk", "Effusion", "eff")):
            v = d[["knee", field]].dropna(); g = v[field].astype(float).groupby(v.knee)
            cols[col] = g.mean() / 3; cols[f"{pre}_ge1"] = pge(d, field, 1); cols[f"{pre}_ge2"] = pge(d, field, 2)
        extras = ["syic_ge1", "syic_ge2", "eff_ge1", "eff_ge2"]
    elif kind == "oa":
        extras = []
        for col, (c, subs) in CARTILAGE.items():
            f1 = [pge(d, s, 1, "ft") for s in subs]; f2 = [pge(d, s, 2, "ft") for s in subs]
            a2 = [pge(d, s, 2, "area") for s in subs]; a3 = [pge(d, s, 3, "area") for s in subs]
            main = [pd.concat([f, a], axis=1).min(axis=1) for f, a in zip(f1, a2)]
            cols[col] = anyof(main)
            for k, parts in (("ft1", f1), ("ft2", f2), ("a2", a2), ("a3", a3)):
                cols[f"{c}_{k}"] = anyof(parts); extras.append(f"{c}_{k}")
    elif kind == "menbak":
        for col, m in (("Medial Meniscus", "mm"), ("Lateral Meniscus", "lm")):
            regs = [f"m_{m}ah_morph", f"m_{m}body_morph", f"m_{m}ph_morph"]
            cols[col] = anyof([pge(d, r, 2) for r in regs] + [pge(d, f"m_{m}pr_tear", 1)])
            cols[f"{m}_ph"] = pge(d, f"m_{m}ph_morph", 2)
            cols[f"{m}_mac"] = anyof([pge(d, r, 6) for r in regs]); cols[f"{m}_sig"] = anyof([pge(d, r, 1) for r in regs])
        m, w = pge(d, "m_popcys", 1), pge(d, "w_popcys", 1)
        cols["Baker's"] = pd.concat([m, w[~w.index.isin(m.index)]])
        cols["ACL"] = pge(d, "m_acl_tear", 1); cols["acl_c"] = pge(d, "m_acl_tear", 2)
        extras = [f"{m}_{k}" for m in ("mm", "lm") for k in ("ph", "mac", "sig")] + ["acl_c"]
    else:
        raise ValueError(kind)
    lab = pd.DataFrame(cols).reindex(columns=TN + extras)
    lab = lab[lab[[c for c in TN if c in cols]].notna().any(axis=1)].sort_index()
    lab.index.name = "StudyInstanceUID"
    return lab.astype(float), lab.notna().astype(float).where(lab.notna())


def write(path, kind, out_dir, knees=None):
    """labels.csv / weights.csv of one target set into out_dir (knees: optional iterable of knee ids to keep, e.g. the OAI cache's)."""
    lab, w = targets(path, kind)
    if knees is not None:
        keep = lab.index.isin(set(knees)); lab, w = lab[keep], w[keep]
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    lab.to_csv(out / f"moaks-{kind}-labels.csv", float_format="%.6f"); w.to_csv(out / f"moaks-{kind}-weights.csv", float_format="%.1f")
    return lab, w
