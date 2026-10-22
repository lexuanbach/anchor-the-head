"""Principled investigation: does GRAPH DIFFUSION of the frozen-CLIP baseline logits help?

The trained feature-residual decoder overfits seen classes and collapses novel-class IoU. Here we
test the cleanest 'overlapping masks share evidence' mechanism that avoids the seen-only collapse
observed for trained heads (in these experiments):
smooth the frozen Proposal+CLIP per-mask class logits over the proposal graph (edge weight =
IoU + boundary-overlap + positive CLIP-region cosine). alpha=0 recovers the training-free baseline;
alpha>0 lets confident neighbours inform uncertain masks. Because diffusion is symmetric over all
classes, novel (unsupervised) classes are treated symmetrically with seen classes, so the operation
does not single out novel classes for collapse the way a seen-only fit does. Reads the cache (no training,
no GPU). Sweeps alpha x k-steps and reports mIoU/hIoU/novel/ECE. Writes results/tables/tab_diffuse.tex
+ results/metrics_diffuse.json.
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


def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(-1, keepdims=True) + 1e-9)


def load_val(cache):
    return [dict(np.load(f)) for f in sorted(glob.glob(os.path.join(cache, "val", "*.npz")))]


def diffuse(logit0, ei, ef, alpha, k):
    n = logit0.shape[0]
    if ei.shape[1] == 0 or alpha == 0:
        return logit0
    w = ef[:, 0] + ef[:, 1] + np.clip(ef[:, 2], 0, None)          # edge weight: IoU+boundary+pos-cos
    A = np.zeros((n, n), np.float32)
    np.add.at(A, (ei[0], ei[1]), w)
    A /= (A.sum(1, keepdims=True) + 1e-6)                          # row-stochastic
    h = logit0.copy()
    for _ in range(k):
        h = (1 - alpha) * logit0 + alpha * (A @ h)                # anchored diffusion
    return h


def evaluate(val, protos, seen, novel, alpha, k):
    inter = np.zeros(core.NUM_CLASSES, np.int64); union = np.zeros(core.NUM_CLASSES, np.int64)
    ec = []; ecorr = []
    for rec in val:
        z = rec["region_emb"].astype(np.float32)
        logit0 = 100.0 * (z @ protos.T)
        h = diffuse(logit0, rec["edge_index"].astype(np.int64), rec["edge_feat"].astype(np.float32),
                    alpha, k)
        prob = softmax(h); cls = h.argmax(1); conf = prob.max(1)
        masks = rec["masks"]; gt = rec["gt"].astype(np.int64)
        pred = core.paint(list(masks), cls, conf, gt.shape[0], gt.shape[1])
        i, u = core.confusion(pred, gt); inter += i; union += u
        glab = rec["gt_label"].astype(np.int64); v = glab != IGNORE
        ec.extend(conf[v].tolist()); ecorr.extend((cls[v] == glab[v]).tolist())
    miou = core.miou_from(inter, union); h_, s, nv = core.hiou(inter, union, seen, novel)
    return {"mIoU": round(miou, 1), "hIoU": round(h_, 1), "seen": round(s, 1),
            "novel": round(nv, 1), "ECE": round(core.ece_bins(ec, ecorr), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache"); ap.add_argument("--out-dir", default="results")
    a = ap.parse_args()
    protos = np.load(os.path.join(a.cache, "text_protos.npy")).astype(np.float32)
    spl = json.load(open(os.path.join(a.cache, "split.json")))
    seen, novel = spl["seen"], spl["novel"]
    val = load_val(a.cache)
    print(f"[diffuse] val={len(val)} seen={len(seen)} novel={len(novel)}", flush=True)
    res = {}
    base = evaluate(val, protos, seen, novel, 0.0, 0)
    res["baseline (alpha=0)"] = base
    print(f"[diffuse] baseline: {base}", flush=True)
    best = ("baseline (alpha=0)", base)
    for alpha in (0.2, 0.4, 0.6, 0.8):
        for k in (1, 2, 3):
            m = evaluate(val, protos, seen, novel, alpha, k)
            tag = f"alpha={alpha},k={k}"
            res[tag] = m
            if m["mIoU"] > best[1]["mIoU"]:
                best = (tag, m)
            print(f"[diffuse] {tag}: {m}", flush=True)
    verdict = ("graph diffusion BEATS baseline" if best[1]["mIoU"] > base["mIoU"]
               and best[1]["novel"] >= base["novel"] - 0.5
               else "graph diffusion does NOT beat baseline")
    print(f"[diffuse] VERDICT: {verdict} (best {best[0]} mIoU={best[1]['mIoU']} vs baseline {base['mIoU']})",
          flush=True)
    res["_verdict"] = verdict; res["_best"] = best[0]
    os.makedirs(os.path.join(a.out_dir, "tables"), exist_ok=True)
    json.dump(res, open(os.path.join(a.out_dir, "metrics_diffuse.json"), "w"), indent=2)
    print("DONE_DIFFUSE", flush=True)


if __name__ == "__main__":
    main()
