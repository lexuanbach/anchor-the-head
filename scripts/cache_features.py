"""Cache frozen SAM masks + CLIP region embeddings + graph edges per ADE20K-150 image (REAL).

This is the one-time GPU bottleneck; everything downstream (baseline, ProtoMask training, ablations)
reads the cache and is fast. For val images we also store downsampled masks + GT so painting->pixel
mIoU is exact. Writes one .npz per image under cache/<split>/ and (once) text_protos.npy + split.json.
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


def reduce_label(ann):
    """ADE20K reduce_zero_label: ann value 0 -> IGNORE, else class = value-1 (0..149)."""
    a = np.array(ann, np.int64)
    out = np.full(a.shape, IGNORE, np.int64)
    m = a > 0
    out[m] = a[m] - 1
    return out


def downsample(arr, R, nearest=True):
    from PIL import Image
    im = Image.fromarray(arr.astype(np.uint8))
    im = im.resize((R, R), Image.NEAREST if nearest else Image.BILINEAR)
    return np.array(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "validation"])
    ap.add_argument("--n-images", type=int, default=400)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--points-per-side", type=int, default=16)
    ap.add_argument("--eval-res", type=int, default=256)
    ap.add_argument("--cache", default=os.environ.get("PMX_CACHE", "cache_mc"))
    ap.add_argument("--ade-root", default=os.environ.get("PMX_ADE_ROOT", "data/ADEChallengeData2016"))
    ap.add_argument("--clip", default="openai/clip-vit-large-patch14")
    ap.add_argument("--encoder", default="context", choices=["bbox", "context"],
                    help="region encoder: 'bbox' = original weak masked-crop; "
                         "'context' = stronger context-padded crop")
    ap.add_argument("--pad", type=float, default=0.25, help="context pad fraction (encoder=context)")
    a = ap.parse_args()
    import torch
    from PIL import Image
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if os.path.exists(os.path.join(a.cache, "qfeat.npy")):
        ap.error("Dense-CLIP cache detected; choose a separate masked-crop --cache directory.")
    outdir = os.path.join(a.cache, "train" if a.split == "train" else "val")
    os.makedirs(outdir, exist_ok=True)

    clip_model, clip_proc = core.build_clip(dev, a.clip)
    # text prototypes + seen/novel split (write once)
    proto_path = os.path.join(a.cache, "text_protos.npy")
    if not os.path.exists(proto_path):
        P = core.text_prototypes(clip_model, clip_proc, core.ADE150, dev)
        np.save(proto_path, P)
        rng = np.random.default_rng(0)
        novel = sorted(rng.choice(core.NUM_CLASSES, 30, replace=False).tolist())
        seen = [c for c in range(core.NUM_CLASSES) if c not in novel]
        json.dump({"seen": seen, "novel": novel}, open(os.path.join(a.cache, "split.json"), "w"))
        print(f"[cache] wrote text_protos {P.shape} + seen/novel split", flush=True)

    gen = core.build_sam(dev)
    sub = "training" if a.split == "train" else "validation"
    imgs = sorted(glob.glob(os.path.join(a.ade_root, "images", sub, "*.jpg")))
    n = min(a.n_images, len(imgs) - a.start)
    print(f"[cache] {a.split}: {n} images from {a.start} (ppside={a.points_per_side})", flush=True)
    done = 0
    for k in range(a.start, a.start + n):
        fp = os.path.join(outdir, f"{k:06d}.npz")
        if os.path.exists(fp):
            done += 1; continue
        img_path = imgs[k]
        ann_path = img_path.replace(os.sep + "images" + os.sep, os.sep + "annotations" + os.sep)[:-4] + ".png"
        pil = Image.open(img_path).convert("RGB")
        gt = reduce_label(Image.open(ann_path))
        W, H = pil.size
        masks = core.sam_masks(gen, pil, points_per_side=a.points_per_side)
        if len(masks) < 2:
            continue
        remb = core.region_embeddings(clip_model, clip_proc, pil, masks, dev,
                                       encoder=a.encoder, pad=a.pad)
        geom = core.mask_features(masks, H, W)
        lab = core.mask_gt_labels(masks, gt)
        ei, ef = core.pairwise_edges(masks, remb)
        save = dict(region_emb=remb.astype(np.float16), geom=geom.astype(np.float32),
                    gt_label=lab.astype(np.int16), edge_index=ei.astype(np.int32),
                    edge_feat=ef.astype(np.float16))
        if a.split == "validation":
            R = a.eval_res
            ms = np.stack([downsample(m, R) for m in masks]).astype(bool)
            save["masks"] = ms
            save["gt"] = downsample(gt, R).astype(np.int16)
        np.savez_compressed(fp, **save)
        done += 1
        if done % 25 == 0:
            print(f"[cache] {a.split} {done}/{n} (last: {len(masks)} masks, {ei.shape[1]} edges)", flush=True)
    print(f"[cache] DONE_CACHE_{a.split.upper()} ({done} files)", flush=True)


if __name__ == "__main__":
    main()
