#!/usr/bin/env python3
"""Materialise the exact seen/novel split files used in the paper.

The splits are deterministic draws from protocol-specific RNG streams; this script writes them to
``splits/`` so the released artifact contains the protocol files themselves, not just the
generator. The RNG calls below are byte-for-byte identical to the ones the experiment
scripts use, so the dumped files match every run:

  * ADE20K-150 : ``np.random.default_rng(seed).choice(150, 30, replace=False)`` for seed in 0..4
                 (identical to ``make_split`` in ``run_calib_baselines.py`` and to the single
                 ``default_rng(0)`` split in ``cache_features.py``).
  * ADE dense  : one RandomState(0), five sequential permutation(150) draws; first 30 novel.
  * PASCAL VOC : a single ``np.random.RandomState(0)`` drawing 5 sequential
                 ``permutation(20)``; the first 6 indices of each permutation are novel
                 (identical to the split loop in ``voc_collapse.py``).

Usage:  python scripts/dump_splits.py            # writes splits/*.json + splits/SHA256SUMS
Only dependency is numpy; no datasets, models, or GPU are required.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

ADE_NUM_CLASSES = 150  # == len(protomask.core.ADE150)
ADE_NOVEL = 30
VOC_NUM_CLASSES = 20
VOC_NOVEL = 6
N_SEEDS = 5

OUT = os.path.join(os.path.dirname(__file__), "..", "splits")


def ade_split(seed: int):
    rng = np.random.default_rng(seed)
    novel = sorted(rng.choice(ADE_NUM_CLASSES, ADE_NOVEL, replace=False).tolist())
    seen = [c for c in range(ADE_NUM_CLASSES) if c not in set(novel)]
    return seen, novel


def ade_dense_splits():
    rng = np.random.RandomState(0)
    out = []
    for _ in range(N_SEEDS):
        perm = rng.permutation(ADE_NUM_CLASSES)
        out.append((sorted(perm[ADE_NOVEL:].tolist()), sorted(perm[:ADE_NOVEL].tolist())))
    return out


def voc_splits():
    rng = np.random.RandomState(0)  # one state, 5 sequential permutations
    out = []
    for _ in range(N_SEEDS):
        perm = rng.permutation(VOC_NUM_CLASSES)
        novel = sorted(perm[:VOC_NOVEL].tolist())
        seen = sorted(perm[VOC_NOVEL:].tolist())
        out.append((seen, novel))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    written = []

    for seed in range(N_SEEDS):
        seen, novel = ade_split(seed)
        p = os.path.join(OUT, f"ade20k_seed{seed}.json")
        json.dump({"dataset": "ADE20K-150", "seed": seed, "num_classes": ADE_NUM_CLASSES,
                   "seen": seen, "novel": novel}, open(p, "w"), indent=1)
        written.append(p)

    for split_index, (seen, novel) in enumerate(ade_dense_splits()):
        p = os.path.join(OUT, f"ade20k_dense_seed{split_index}.json")
        with open(p, "w") as fh:
            json.dump({"dataset": "ADE20K-150", "protocol": "dense-CLIP",
                       "rng": "RandomState", "rng_seed": 0, "split_index": split_index,
                       "num_classes": ADE_NUM_CLASSES, "seen": seen, "novel": novel}, fh, indent=1)
        written.append(p)

    for seed, (seen, novel) in enumerate(voc_splits()):
        p = os.path.join(OUT, f"voc_seed{seed}.json")
        json.dump({"dataset": "PASCAL VOC2012", "seed": seed, "num_classes": VOC_NUM_CLASSES,
                   "seen": seen, "novel": novel}, open(p, "w"), indent=1)
        written.append(p)

    lines = []
    for p in sorted(written):
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        lines.append(f"{h}  {os.path.basename(p)}")
    open(os.path.join(OUT, "SHA256SUMS"), "w").write("\n".join(lines) + "\n")
    print(f"[dump_splits] wrote {len(written)} split files + SHA256SUMS to {os.path.normpath(OUT)}")


if __name__ == "__main__":
    main()
