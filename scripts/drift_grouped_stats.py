"""Grouped + bootstrap uncertainty for the drift-collapse association.

Hyperparameter sweeps within one correction family (e.g. the B1 lambda curve) are correlated and
must not be counted as independent replications. We therefore:
  (a) collapse each class-specific correction FAMILY to one point (the family mean of drift and
      collapse index) and report the family-level Pearson r with a bootstrap CI over families;
  (b) report a cluster bootstrap over families on the full point set for comparison.
Writes results/extended/drift_grouped.json and prints the numbers reported in the paper.
"""
import json

import numpy as np

FAM = {  # class-specific corrections only (the scoped claim)
    "decoder_seen_only": "seen_head", "head_ce": "seen_head",
    "B1_": "B1_distill", "B2_": "B2_embKD", "B3_": "B3_gate", "B4_": "B4_winterp",
    "B5_episodic": "B5_episodic", "C1_": "C1_balanced", "C2_": "C2_objective",
    "C3_": "C3_scale", "C4_": "C4_trust", "C5_": "C5_breadth", "C6_": "C6_purity",
    "base+head": "seen_head", "LPOSS+head": "lposs_head",
}
SYM = {"uncal", "globalT", "diffusion", "laplacian", "B6_learned_affinity", "LPOSS"}


def fam_of(k):
    for pre, f in FAM.items():
        if k.startswith(pre):
            return f
    return None


def main():
    pts = []
    for fn in ["results/extended/metrics_align.json", "results/extended/metrics_diag.json",
               "results/extended/metrics_lposs.json"]:
        d = json.load(open(fn))["summary"]
        for k, v in d.items():
            ci = v.get("collapse_index")
            dr = v.get("drift", {}).get("mean")
            if ci is None or dr is None or dr != dr or k in SYM:
                continue
            f = fam_of(k)
            if f:
                pts.append((f, dr, ci))
    fams = sorted({f for f, _, _ in pts})
    fam_pts = []
    for f in fams:
        ds = [d for ff, d, _ in pts if ff == f]
        cs = [c for ff, _, c in pts if ff == f]
        fam_pts.append((f, float(np.mean(ds)), float(np.mean(cs))))
    x = np.array([d for _, d, _ in fam_pts])
    y = np.array([c for _, _, c in fam_pts])
    r_fam = float(np.corrcoef(x, y)[0, 1])

    rng = np.random.RandomState(0)
    boots = []
    for _ in range(10000):
        idx = rng.randint(0, len(fam_pts), len(fam_pts))
        xb, yb = x[idx], y[idx]
        if xb.std() > 0 and yb.std() > 0:
            boots.append(np.corrcoef(xb, yb)[0, 1])
    lo_f, hi_f = np.percentile(boots, [2.5, 97.5])

    # cluster bootstrap on the full point set (resample families, keep their members)
    allx = np.array([d for _, d, _ in pts])
    boots2 = []
    for _ in range(10000):
        fs = rng.choice(fams, len(fams), replace=True)
        xb, yb = [], []
        for f in fs:
            xb += [d for ff, d, _ in pts if ff == f]
            yb += [c for ff, _, c in pts if ff == f]
        xb, yb = np.array(xb), np.array(yb)
        if xb.std() > 0 and yb.std() > 0:
            boots2.append(np.corrcoef(xb, yb)[0, 1])
    lo_c, hi_c = np.percentile(boots2, [2.5, 97.5])
    r_all = float(np.corrcoef(allx, np.array([c for _, _, c in pts]))[0, 1])

    # --- within-run association on the DINO-proxy base (reported separately in Sec. 6) ---
    # Same point definition (class-specific corrections only), but restricted to the single
    # metrics_align_proxy.json run, so it is one run's pointwise r -- not a second family-level
    # replication of the main analysis.
    ppts = []
    dpx = json.load(open("results/extended/metrics_align_proxy.json"))["summary"]
    for k, v in dpx.items():
        ci = v.get("collapse_index")
        dr = v.get("drift", {}).get("mean")
        if ci is None or dr is None or dr != dr or k in SYM:
            continue
        if fam_of(k):
            ppts.append((dr, ci))
    px = np.array([d for d, _ in ppts])
    py = np.array([c for _, c in ppts])
    r_proxy = float(np.corrcoef(px, py)[0, 1]) if len(ppts) > 2 else float("nan")

    out = dict(n_points=len(pts), n_families=len(fam_pts),
               family_points={f: dict(drift=round(d, 1), ci=round(c, 3)) for f, d, c in fam_pts},
               r_family_level=round(r_fam, 3),
               r_family_boot95=[round(float(lo_f), 3), round(float(hi_f), 3)],
               r_pointwise=round(r_all, 3),
               r_cluster_boot95=[round(float(lo_c), 3), round(float(hi_c), 3)],
               proxy_base_within_run=dict(
                   source="results/extended/metrics_align_proxy.json",
                   n_points=len(ppts),
                   r_pointwise=round(r_proxy, 3),
                   note="Single run on the DINO-proxy base; pointwise only, no family-level or "
                        "bootstrap statistic is claimed for it."))
    json.dump(out, open("results/extended/drift_grouped.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
