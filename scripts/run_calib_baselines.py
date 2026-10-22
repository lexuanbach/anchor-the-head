"""Calibration-baseline comparison (REAL, CPU/numpy on the cached frozen-CLIP logits).

Compares parameter-free graph diffusion against actual post-hoc calibrators, not only the
*uncalibrated* baseline. Here we fit the
standard post-hoc calibrators on SEEN-class masks (held-out train cache) and evaluate them on the
full ADE20K-150 val set, alongside the parameter-free graph diffusion, reporting:

  - overall ECE, NOVEL-class ECE (the decisive test: a global temperature cannot specialise to the
    held-out classes, but graph evidence can), seen-class ECE, mIoU, novel-IoU, hIoU.

Calibrators (all on the identical per-mask CLIP logits z = scale * (region_emb @ text_protos.T)):
  (a) temperature scaling   : single global T, fit by NLL on seen-class masks.
  (b) per-class temperature : one T_c per seen class (novel classes inherit the mean seen T), NLL fit.
  (c) vector/Platt scaling  : per-class affine on logits z'_c = a_c*z_c + b_c, fit by NLL on seen masks.
  (d) graph diffusion       : the paper's parameter-free method (anchored row-stochastic smoothing).

Statistical rigour: results are averaged over >=3 random seen/novel splits (seeded), and we report
bootstrap 95% CIs over val images for the headline metrics. Pure numpy/CPU, no GPU, no training.

Run:  PYTHONPATH=src python scripts/run_calib_baselines.py --cache cache --out-dir results
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from protomask import core  # noqa: E402

IGNORE = 255
SCALE = 100.0  # the fixed CLIP logit scale used everywhere in this project


# --------------------------------------------------------------------------- io
def load_split(cache, split):
    files = sorted(glob.glob(os.path.join(cache, split, "*.npz")))
    recs = []
    for f in files:
        d = np.load(f, allow_pickle=False)
        recs.append({k: d[k] for k in d.files})
    return recs


# ---------------------------------------------------------------- numpy helpers
def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(-1, keepdims=True) + 1e-9)


def base_logits(rec, protos):
    """Per-mask CLIP logits, identical to the baseline / diffusion start point."""
    z = rec["region_emb"].astype(np.float32)
    return SCALE * (z @ protos.T)


def diffuse(logit0, ei, ef, alpha, k, edge_mode="all"):
    """Anchored row-stochastic graph diffusion of per-mask logits (parameter-free)."""
    n = logit0.shape[0]
    if ei.shape[1] == 0 or alpha == 0:
        return logit0
    iou, bnd, cos = ef[:, 0], ef[:, 1], np.clip(ef[:, 2], 0, None)
    if edge_mode == "all":
        w = iou + bnd + cos
    elif edge_mode == "iou":
        w = iou
    elif edge_mode == "boundary":
        w = bnd
    elif edge_mode == "cos":
        w = cos
    else:
        raise ValueError(edge_mode)
    A = np.zeros((n, n), np.float32)
    np.add.at(A, (ei[0], ei[1]), w)
    A /= (A.sum(1, keepdims=True) + 1e-6)
    h = logit0.copy()
    for _ in range(k):
        h = (1 - alpha) * logit0 + alpha * (A @ h)
    return h


# --------------------------------------------------------- calibrator fitting
def _gather_seen(train_recs, protos, seen_mask):
    """Stack per-mask logits + GT labels for masks whose majority GT is a SEEN class."""
    Z, Y = [], []
    for rec in train_recs:
        z = base_logits(rec, protos)
        glab = rec["gt_label"].astype(np.int64)
        keep = (glab != IGNORE) & seen_mask[np.clip(glab, 0, core.NUM_CLASSES - 1)]
        if keep.any():
            Z.append(z[keep]); Y.append(glab[keep])
    if not Z:
        return np.zeros((0, core.NUM_CLASSES), np.float32), np.zeros((0,), np.int64)
    return np.concatenate(Z, 0), np.concatenate(Y, 0)


def _nll(z, y):
    p = softmax(z)
    return -np.log(p[np.arange(len(y)), y] + 1e-12).mean()


def fit_temperature(Z, Y, grid=None):
    """Single global temperature: minimise NLL of z/T over a fine grid (standard Guo et al. 2017)."""
    if grid is None:
        grid = np.concatenate([np.linspace(0.05, 2.0, 40), np.linspace(2.0, 12.0, 50)])
    best_T, best = 1.0, np.inf
    for T in grid:
        nll = _nll(Z / T, Y)
        if nll < best:
            best, best_T = nll, float(T)
    return best_T


def fit_per_class_temperature(Z, Y, seen, T_global=None, iters=400, lr=0.05):
    """Per-class temperature: calibrated logit column c is z_c / T_c (softmax over all c). The vector
    of per-class temperatures over SEEN classes is fit *jointly* by NLL gradient descent on the
    seen-class masks (initialised at the single global temperature). NOVEL classes inherit the global
    temperature, since no novel supervision exists -- this is the strongest a *temperature* family can
    do toward held-out classes, and is exactly why it cannot specialise to them. We parameterise
    t_c = log T_c for an unconstrained, positive-temperature fit."""
    C = core.NUM_CLASSES
    if T_global is None:
        T_global = fit_temperature(Z, Y)
    seen_idx = np.array(sorted(seen))
    upd = np.zeros(C, bool); upd[seen_idx] = True
    logT = np.full(C, np.log(T_global), np.float64)
    n = len(Y)
    if n == 0:
        return np.full(C, float(T_global), np.float32)
    Yoh = np.zeros((n, C)); Yoh[np.arange(n), Y] = 1.0
    Z = Z.astype(np.float64)
    for _ in range(iters):
        T = np.exp(logT)
        zp = Z / T
        p = softmax(zp)
        g = (p - Yoh)                                   # dL/d(zp)  (n, C)
        # d zp_c / d logT_c = -z_c / T_c ; sum over samples
        gT = (g * (-Z / T)).mean(0)
        logT[upd] -= lr * gT[upd]
        logT = np.clip(logT, np.log(0.2), np.log(20.0))
    T = np.exp(logT).astype(np.float32)
    T[~upd] = float(T_global)                            # novel classes: global temperature
    return T


def fit_vector_scaling(Z, Y, seen, T_global=None, iters=600, lr=0.02):
    """Per-class affine (Platt/vector) scaling z'_c = a_c * z_c + b_c, fit by NLL (full-softmax) with
    gradient descent on the SEEN-class masks only. Initialised from the global temperature
    (a = 1/T_global, b = 0) so it generalises temperature scaling. a,b for NOVEL classes stay at this
    global-temperature value (never updated), so novel logits get only the global softening and no
    class-specific correction -- the expected limitation of a per-class scaler
    that never sees novel data."""
    C = core.NUM_CLASSES
    if T_global is None:
        T_global = fit_temperature(Z, Y)
    a0 = 1.0 / float(T_global)
    a = np.full(C, a0, np.float64); b = np.zeros(C, np.float64)
    seen_idx = np.array(sorted(seen))
    n = len(Y)
    if n == 0:
        return a.astype(np.float32), b.astype(np.float32)
    Yoh = np.zeros((n, C)); Yoh[np.arange(n), Y] = 1.0
    Z = Z.astype(np.float64)
    upd = np.zeros(C, bool); upd[seen_idx] = True
    for _ in range(iters):
        zp = a * Z + b
        p = softmax(zp)
        g = (p - Yoh)                      # (n, C) softmax-CE gradient wrt logits
        ga = (g * Z).mean(0)               # dL/da_c
        gb = g.mean(0)                     # dL/db_c
        a[upd] -= lr * ga[upd]
        b[upd] -= lr * gb[upd]
        a = np.clip(a, 0.01, 20.0)
    return a.astype(np.float32), b.astype(np.float32)


# --------------------------------------------------------------- per-image eval
def eval_method(val, protos, seen, novel, transform):
    """transform(logit0, rec) -> calibrated per-mask logits. Returns per-image accumulators so we can
    bootstrap over images. Computes pixel mIoU/novel-IoU/hIoU and mask-level ECE (overall/novel/seen),
    matching the project's existing metric definitions exactly."""
    seen_set, novel_set = set(seen), set(novel)
    per_img = []
    for rec in val:
        logit0 = base_logits(rec, protos)
        h = transform(logit0, rec)
        prob = softmax(h); cls = h.argmax(1); conf = prob.max(1)
        masks = rec["masks"]; gt = rec["gt"].astype(np.int64)
        pred = core.paint(list(masks), cls, conf, gt.shape[0], gt.shape[1])
        inter, union = core.confusion(pred, gt)
        glab = rec["gt_label"].astype(np.int64); v = glab != IGNORE
        c = conf[v]; corr = (cls[v] == glab[v]).astype(np.float32); lab = glab[v]
        is_novel = np.array([l in novel_set for l in lab])
        is_seen = np.array([l in seen_set for l in lab])
        per_img.append(dict(inter=inter, union=union,
                            conf=c, corr=corr, novel=is_novel, seen=is_seen))
    return per_img


def _ece(conf, corr, n_bins=15):
    return core.ece_bins(conf, corr, n_bins=n_bins)


def reduce_metrics(per_img, seen, novel, idx=None):
    """Aggregate per-image accumulators (optionally over a bootstrap index subset) into scalar metrics."""
    if idx is None:
        idx = range(len(per_img))
    inter = np.zeros(core.NUM_CLASSES, np.int64); union = np.zeros(core.NUM_CLASSES, np.int64)
    conf = []; corr = []; nv = []; sn = []
    for i in idx:
        d = per_img[i]
        inter += d["inter"]; union += d["union"]
        conf.append(d["conf"]); corr.append(d["corr"]); nv.append(d["novel"]); sn.append(d["seen"])
    conf = np.concatenate(conf) if conf else np.zeros(0)
    corr = np.concatenate(corr) if corr else np.zeros(0)
    nv = np.concatenate(nv) if nv else np.zeros(0, bool)
    sn = np.concatenate(sn) if sn else np.zeros(0, bool)
    miou = core.miou_from(inter, union)
    h, s_m, n_m = core.hiou(inter, union, seen, novel)
    return dict(mIoU=miou, hIoU=h, seenIoU=s_m, novelIoU=n_m,
                ECE=_ece(conf, corr),
                ECE_novel=_ece(conf[nv], corr[nv]) if nv.any() else 0.0,
                ECE_seen=_ece(conf[sn], corr[sn]) if sn.any() else 0.0)


def bootstrap_ci(per_img, seen, novel, keys, n_boot=500, seed=0):
    """Bootstrap 95% CIs over images for the given metric keys."""
    rng = np.random.default_rng(seed)
    N = len(per_img)
    samples = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, N, N)
        m = reduce_metrics(per_img, seen, novel, idx)
        for k in keys:
            samples[k].append(m[k])
    return {k: (float(np.percentile(samples[k], 2.5)), float(np.percentile(samples[k], 97.5)))
            for k in keys}


# ----------------------------------------------------------------------- driver
def make_split(seed):
    """Seen/novel split for a seed. Prefers the released ``splits/ade20k_seed{seed}.json``
    (byte-identical to the RNG below; see ``scripts/dump_splits.py``) so the protocol is read
    from the shipped file, and falls back to regenerating it deterministically if absent."""
    sp = os.path.join(os.path.dirname(__file__), "..", "splits", f"ade20k_seed{seed}.json")
    if os.path.exists(sp):
        d = json.load(open(sp))
        return d["seen"], d["novel"]
    rng = np.random.default_rng(seed)
    novel = sorted(rng.choice(core.NUM_CLASSES, 30, replace=False).tolist())
    seen = [c for c in range(core.NUM_CLASSES) if c not in set(novel)]
    return seen, novel


def run_one_split(val, train, protos, seen, novel, alpha, k):
    seen_mask = np.zeros(core.NUM_CLASSES, bool); seen_mask[seen] = True
    Z, Y = _gather_seen(train, protos, seen_mask)

    T_glob = fit_temperature(Z, Y)
    T_pc = fit_per_class_temperature(Z, Y, seen, T_global=T_glob)
    a_vs, b_vs = fit_vector_scaling(Z, Y, seen, T_global=T_glob)

    methods = {
        "baseline":     lambda z, r: z,
        "temp_global":  lambda z, r: z / T_glob,
        "temp_perclass": lambda z, r: z / T_pc[None, :],
        "vector_scaling": lambda z, r: a_vs[None, :] * z + b_vs[None, :],
        "diffusion":    lambda z, r: diffuse(z, r["edge_index"].astype(np.int64),
                                             r["edge_feat"].astype(np.float32), alpha, k),
    }
    out = {}
    per_img_store = {}
    for name, fn in methods.items():
        per_img = eval_method(val, protos, seen, novel, fn)
        out[name] = reduce_metrics(per_img, seen, novel)
        per_img_store[name] = per_img
    out["_fit"] = dict(T_global=round(float(T_glob), 3),
                       T_perclass_mean=round(float(T_pc.mean()), 3),
                       vector_a_mean=round(float(a_vs.mean()), 3))
    return out, per_img_store


def run_edge_ablation(val, protos, seen, novel, alpha, k):
    res = {}
    for mode in ("all", "iou", "boundary", "cos"):
        fn = lambda z, r, m=mode: diffuse(z, r["edge_index"].astype(np.int64),
                                          r["edge_feat"].astype(np.float32), alpha, k, edge_mode=m)
        per_img = eval_method(val, protos, seen, novel, fn)
        res[mode] = reduce_metrics(per_img, seen, novel)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-boot", type=int, default=500)
    a = ap.parse_args()

    protos = np.load(os.path.join(a.cache, "text_protos.npy")).astype(np.float32)
    val = load_split(a.cache, "val")
    train = load_split(a.cache, "train")
    print(f"[calib] val={len(val)} train={len(train)} alpha={a.alpha} k={a.k} seeds={a.seeds}",
          flush=True)

    methods = ["baseline", "temp_global", "temp_perclass", "vector_scaling", "diffusion"]
    metrics = ["mIoU", "hIoU", "seenIoU", "novelIoU", "ECE", "ECE_novel", "ECE_seen"]
    acc = {m: {k: [] for k in metrics} for m in methods}
    per_seed = {}
    fits = []
    boot_seed0 = None

    for s in a.seeds:
        seen, novel = make_split(s)
        out, per_img_store = run_one_split(val, train, protos, seen, novel, a.alpha, a.k)
        per_seed[s] = {m: out[m] for m in methods}
        fits.append(out["_fit"])
        for m in methods:
            for k in metrics:
                acc[m][k].append(out[m][k])
        print(f"[calib] seed={s} fit={out['_fit']}", flush=True)
        for m in methods:
            o = out[m]
            print(f"          {m:16s} mIoU={o['mIoU']:.1f} novelIoU={o['novelIoU']:.1f} "
                  f"ECE={o['ECE']:.1f} ECE_novel={o['ECE_novel']:.1f} ECE_seen={o['ECE_seen']:.1f}",
                  flush=True)
        if s == a.seeds[0]:
            boot_seed0 = {m: bootstrap_ci(per_img_store[m], seen, novel,
                                          ["mIoU", "novelIoU", "ECE", "ECE_novel"], n_boot=a.n_boot)
                          for m in methods}

    summary = {m: {k: dict(mean=round(float(np.mean(acc[m][k])), 2),
                           std=round(float(np.std(acc[m][k])), 2))
                   for k in metrics} for m in methods}

    # edge-source ablation on the first seed only (cheap, illustrative)
    seen0, novel0 = make_split(a.seeds[0])
    edge = run_edge_ablation(val, protos, seen0, novel0, a.alpha, a.k)
    edge = {m: {k: round(float(v[k]), 2) for k in metrics} for m, v in edge.items()}

    result = dict(
        config=dict(alpha=a.alpha, k=a.k, seeds=a.seeds, n_val=len(val), n_train=len(train),
                    eval_res=int(val[0]["gt"].shape[0]), n_bins=15, scale=SCALE,
                    n_novel=30, n_classes=core.NUM_CLASSES, fit_set="seen-class train masks",
                    n_boot=a.n_boot),
        summary=summary, per_seed=per_seed, fits=fits,
        bootstrap_ci_seed0=boot_seed0, edge_ablation=edge,
    )
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(result, open(os.path.join(a.out_dir, "metrics_calib.json"), "w"), indent=2)
    write_tables(result, a.out_dir)

    # verdict (honest)
    d = summary["diffusion"]; tg = summary["temp_global"]
    pc = summary["temp_perclass"]; vs = summary["vector_scaling"]
    print("\n[calib] ===== VERDICT (mean over seeds) =====", flush=True)
    print(f"  overall  ECE: diffusion={d['ECE']['mean']}  temp_global={tg['ECE']['mean']}  "
          f"perclass={pc['ECE']['mean']}  vector={vs['ECE']['mean']}", flush=True)
    print(f"  NOVEL    ECE: diffusion={d['ECE_novel']['mean']}  temp_global={tg['ECE_novel']['mean']}  "
          f"perclass={pc['ECE_novel']['mean']}  vector={vs['ECE_novel']['mean']}", flush=True)
    print(f"  mIoU       : diffusion={d['mIoU']['mean']}  temp_global={tg['mIoU']['mean']}  "
          f"baseline={summary['baseline']['mIoU']['mean']}", flush=True)
    diff_wins_novel = d["ECE_novel"]["mean"] < min(tg["ECE_novel"]["mean"],
                                                   pc["ECE_novel"]["mean"], vs["ECE_novel"]["mean"])
    diff_wins_overall = d["ECE"]["mean"] < min(tg["ECE"]["mean"], pc["ECE"]["mean"], vs["ECE"]["mean"])
    print(f"  diffusion best on NOVEL ECE? {diff_wins_novel};  best on overall ECE? {diff_wins_overall}",
          flush=True)
    json.dump(dict(diff_best_novel_ece=bool(diff_wins_novel),
                   diff_best_overall_ece=bool(diff_wins_overall)),
              open(os.path.join(a.out_dir, "calib_verdict.json"), "w"), indent=2)
    print("DONE_CALIB", flush=True)


def write_tables(result, out_dir):
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
    s = result["summary"]; cfg = result["config"]

    def cell(m, k):
        v = s[m][k]
        return f"{v['mean']:.1f}\\,$\\pm$\\,{v['std']:.1f}"

    pretty = {"baseline": "Uncalibrated baseline", "temp_global": "Temperature scaling (global $T$)",
              "temp_perclass": "Per-class temperature", "vector_scaling": "Vector/Platt scaling",
              "diffusion": "Graph diffusion"}
    order = ["baseline", "temp_global", "temp_perclass", "vector_scaling", "diffusion"]
    L = [f"% AUTO-GENERATED by scripts/run_calib_baselines.py -- ADE20K-150 val "
         f"(N={cfg['n_val']} imgs, {cfg['eval_res']}px), mean$\\pm$std over {len(cfg['seeds'])} "
         f"random seen/novel splits; calibrators fit on seen-class train masks.",
         "\\centering\\small",
         "\\caption{\\textbf{Real} calibration-baseline comparison on ADE20K-150 val "
         f"($N{{=}}{cfg['n_val']}$ images at {cfg['eval_res']}px; mean$\\pm$std over "
         f"{len(cfg['seeds'])} random seen/novel splits, $30$ novel classes each; post-hoc "
         "calibrators fit on seen-class train masks). All methods start from the identical frozen-CLIP "
         "per-mask logits. A global temperature is the best calibrator overall, while the two per-class "
         "calibrators collapse novel classes.}",
         "\\label{tab:real_calib}",
         "\\begin{tabular}{lccccc}", "\\toprule",
         "Method & mIoU & novel-IoU & ECE$\\downarrow$ & novel ECE$\\downarrow$ & seen ECE$\\downarrow$ \\\\",
         "\\midrule"]
    for m in order:
        L.append(f"{pretty[m]} & {cell(m,'mIoU')} & {cell(m,'novelIoU')} & "
                 f"{cell(m,'ECE')} & {cell(m,'ECE_novel')} & {cell(m,'ECE_seen')} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(out_dir, "tables", "tab_real_calib.tex"), "w").write("\n".join(L) + "\n")

    # edge-source ablation table
    e = result["edge_ablation"]
    em = {"all": "all edges (IoU+boundary+cos)", "iou": "IoU only", "boundary": "boundary only",
          "cos": "CLIP-cosine only"}
    A = ["% AUTO-GENERATED by scripts/run_calib_baselines.py -- edge-source ablation, ADE20K-150 val.",
         "\\centering\\small",
         "\\caption{\\textbf{Real} edge-source ablation for graph diffusion "
         f"($\\alpha{{=}}{cfg['alpha']}, k{{=}}{cfg['k']}$; first split; ADE20K-150 val). Each row "
         "restricts the diffusion edge weight to a single evidence source.}",
         "\\label{tab:real_edge}",
         "\\begin{tabular}{lcccc}", "\\toprule",
         "Edge source & mIoU & novel-IoU & ECE$\\downarrow$ & novel ECE$\\downarrow$ \\\\", "\\midrule"]
    for k in ("all", "iou", "boundary", "cos"):
        v = e[k]
        A.append(f"{em[k]} & {v['mIoU']:.1f} & {v['novelIoU']:.1f} & {v['ECE']:.1f} & {v['ECE_novel']:.1f} \\\\")
    A += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(out_dir, "tables", "tab_real_edge.tex"), "w").write("\n".join(A) + "\n")
    print("[calib] wrote tab_real_calib.tex + tab_real_edge.tex", flush=True)


if __name__ == "__main__":
    main()
