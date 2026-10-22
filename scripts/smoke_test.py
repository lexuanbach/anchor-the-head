"""Synthetic smoke test: validates the ProtoMask graph/decoder/metrics + train+eval path end-to-end
without SAM/CLIP/datasets. Builds fake per-image records and runs baseline eval, a tiny training run,
and the table writer. Catches shape/logic bugs before the GPU caching run.
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from protomask import core  # noqa: E402

IGNORE = 255


def fake_rec(rng, C, D, R=64, val=False):
    n = int(rng.integers(4, 10))
    re = rng.normal(size=(n, D)).astype(np.float32)
    re /= (np.linalg.norm(re, axis=1, keepdims=True) + 1e-6)
    geom = rng.random((n, 5)).astype(np.float32)
    lab = rng.integers(0, C, n).astype(np.int16)
    lab[rng.random(n) < 0.2] = IGNORE
    # a few random edges
    e = int(rng.integers(2, 2 * n))
    ei = rng.integers(0, n, size=(2, e)).astype(np.int32)
    ef = rng.random((e, 3)).astype(np.float16)
    rec = dict(region_emb=re.astype(np.float16), geom=geom, gt_label=lab,
               edge_index=ei, edge_feat=ef)
    if val:
        masks = (rng.random((n, R, R)) < 0.25)
        masks[0] = True  # ensure coverage
        gt = rng.integers(0, C, (R, R)).astype(np.int16)
        gt[rng.random((R, R)) < 0.1] = IGNORE
        rec["masks"] = masks.astype(bool); rec["gt"] = gt
    return rec


def main():
    import torch
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    import run  # the driver (reuse evaluate/train/_write_tables)
    dev = "cpu"
    rng = np.random.default_rng(0)
    C = core.NUM_CLASSES; D = 768
    protos = rng.normal(size=(C, D)).astype(np.float32)
    protos /= (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-6)
    protos_t = torch.from_numpy(protos).to(dev)
    seen = list(range(C - 30)); novel = list(range(C - 30, C))
    tr = [fake_rec(rng, C, D) for _ in range(12)]
    va = [fake_rec(rng, C, D, val=True) for _ in range(5)]

    res = {}
    res["baseline"] = run.evaluate(None, va, protos_t, seen, novel, dev, baseline=True)
    assert all(np.isfinite(v) for v in res["baseline"].values()), res["baseline"]
    print("baseline OK:", res["baseline"])

    cfgs = {"full": dict(use_graph=True, use_uncertainty=True, refine_protos=True),
            "no_graph": dict(use_graph=False, use_uncertainty=True, refine_protos=True),
            "no_uncertainty": dict(use_graph=True, use_uncertainty=False, refine_protos=True),
            "fixed_protos": dict(use_graph=True, use_uncertainty=True, refine_protos=False)}
    for nm, cfg in cfgs.items():
        net = run.train(cfg, tr, protos_t, set(seen), dev, epochs=2, lr=1e-3, ms_img=None)
        m = run.evaluate(net, va, protos_t, seen, novel, dev)
        assert all(np.isfinite(v) for v in m.values()), (nm, m)
        res[nm] = m
        print(f"{nm} OK:", m)
    res["mask_independent_clip"] = res["baseline"]
    with tempfile.TemporaryDirectory(prefix="anchor-head-smoke-") as out_dir:
        os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
        run._write_tables(res, out_dir)
        assert os.path.exists(os.path.join(out_dir, "tables", "tab_main.tex"))
        assert os.path.exists(os.path.join(out_dir, "tables", "tab_ablation.tex"))
    print("SMOKE_OK: model + metrics + train/eval + table writer all run end-to-end")


if __name__ == "__main__":
    main()
