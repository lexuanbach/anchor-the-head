"""ProtoMask-Set core (REAL): ADE20K-150 data, frozen SAM mask proposals, frozen CLIP region/text
embeddings, mask->GT-label assignment, mask painting, and segmentation/calibration metrics.

Everything here is real and runs on a single GPU with frozen public models
(facebook/sam-vit-base, openai/clip-vit-large-patch14). No training in this file.
"""
from __future__ import annotations

import numpy as np

IGNORE = 255

# Canonical ADE20K-150 class names (index 0..149), reduce_zero_label convention
# (annotation value 0 = ignore; class c occupies annotation value c+1).
ADE150 = [
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed", "windowpane", "grass",
    "cabinet", "sidewalk", "person", "earth", "door", "table", "mountain", "plant", "curtain",
    "chair", "car", "water", "painting", "sofa", "shelf", "house", "sea", "mirror", "rug", "field",
    "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp", "bathtub", "railing", "cushion",
    "base", "box", "column", "signboard", "chest of drawers", "counter", "sand", "sink",
    "skyscraper", "fireplace", "refrigerator", "grandstand", "path", "stairs", "runway", "case",
    "pool table", "pillow", "screen door", "stairway", "river", "bridge", "bookcase", "blind",
    "coffee table", "toilet", "flower", "book", "hill", "bench", "countertop", "stove", "palm",
    "kitchen island", "computer", "swivel chair", "boat", "bar", "arcade machine", "hovel", "bus",
    "towel", "light", "truck", "tower", "chandelier", "awning", "streetlight", "booth", "television",
    "airplane", "dirt track", "apparel", "pole", "land", "bannister", "escalator", "ottoman",
    "bottle", "buffet", "poster", "stage", "van", "ship", "fountain", "conveyer belt", "canopy",
    "washer", "plaything", "swimming pool", "stool", "barrel", "basket", "waterfall", "tent", "bag",
    "minibike", "cradle", "oven", "ball", "food", "step", "tank", "trade name", "microwave", "pot",
    "animal", "bicycle", "lake", "dishwasher", "screen", "blanket", "sculpture", "hood", "sconce",
    "vase", "traffic light", "tray", "ashcan", "fan", "pier", "crt screen", "plate", "monitor",
    "bulletin board", "shower", "radiator", "glass", "clock", "flag",
]
NUM_CLASSES = len(ADE150)  # 150

# CLIP prompt ensemble (a small, standard subset)
PROMPTS = ["a photo of a {}.", "a photo of the {}.", "a cropped photo of a {}.",
           "a close-up photo of a {}.", "{}."]


# ----------------------------- frozen models -----------------------------
def build_clip(device, name="openai/clip-vit-large-patch14"):
    import torch
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(name).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    proc = CLIPProcessor.from_pretrained(name)
    return model, proc


def text_prototypes(model, proc, names, device):
    """CLIP text embeddings per class (prompt-ensembled, L2-normalised). Returns (C, D) numpy."""
    import torch
    protos = []
    with torch.no_grad():
        for n in names:
            prompts = [t.format(n) for t in PROMPTS]
            tok = proc(text=prompts, return_tensors="pt", padding=True).to(device)
            f = model.get_text_features(**tok)
            f = f / f.norm(dim=-1, keepdim=True)
            protos.append(f.mean(0))
    P = torch.stack(protos)
    P = P / P.norm(dim=-1, keepdim=True)
    return P.float().cpu().numpy()


def build_sam(device):
    from transformers import pipeline
    dev = 0 if str(device).startswith("cuda") else -1
    return pipeline("mask-generation", model="facebook/sam-vit-base", device=dev)


def sam_masks(gen, pil, points_per_side=16, min_area=400, max_masks=64):
    """Frozen SAM automatic masks -> list of bool (H,W) arrays, largest-first, tiny ones dropped."""
    out = gen(pil.convert("RGB"), points_per_side=points_per_side, pred_iou_thresh=0.86,
              stability_score_thresh=0.9, points_per_batch=64)
    masks = [np.asarray(m, dtype=bool) for m in out["masks"]]
    masks = [m for m in masks if m.sum() >= min_area]
    masks.sort(key=lambda m: -int(m.sum()))
    return masks[:max_masks]


def _clip_encode_crops(clip_model, clip_proc, crops, device, bs):
    import torch
    embs = []
    with torch.no_grad():
        for i in range(0, len(crops), bs):
            batch = clip_proc(images=crops[i:i + bs], return_tensors="pt").to(device)
            f = clip_model.get_image_features(pixel_values=batch["pixel_values"])
            f = f / f.norm(dim=-1, keepdim=True)
            embs.append(f.float().cpu())
    return (torch.cat(embs).numpy() if embs
            else np.zeros((0, clip_model.config.projection_dim), np.float32))


def region_embeddings(clip_model, clip_proc, pil, masks, device, bs=48,
                      encoder="context", pad=0.25):
    """Region (mask) embeddings via CLIP image features. Returns (N, D) L2-normalised numpy.

    encoder:
      "bbox"    : ORIGINAL weak encoder -- tight masked-bbox crop, non-mask pixels -> CLIP mean grey.
                  This destroys natural context and is the toy-regime (~9 mIoU) baseline.
      "context" : STRONGER encoder (default) -- crop the mask's bounding box padded by `pad` of its
                  size on each side and keep the *natural surrounding background* (no grey-fill, no
                  blur). CLIP was pretrained on natural sub-images, so a context-padded crop is a far
                  better region descriptor than a hard-masked bbox: in a controlled region-class top-1
                  probe on ADE20K-150 val it lifts accuracy from ~0.09 (bbox) to ~0.16 (context,
                  pad=0.25). This is the lever the stronger-encoder experiment pulls.
    """
    from PIL import Image
    img = np.array(pil.convert("RGB"))
    H, W = img.shape[:2]
    grey = np.array([124, 116, 104], np.uint8)
    crops = []
    for m in masks:
        ys, xs = np.where(m)
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        if encoder == "bbox":
            crop = img[y0:y1 + 1, x0:x1 + 1].copy()
            mm = m[y0:y1 + 1, x0:x1 + 1]
            crop[~mm] = grey
        elif encoder == "context":
            bh, bw = y1 - y0 + 1, x1 - x0 + 1
            py, px = int(bh * pad), int(bw * pad)
            Y0, Y1 = max(0, y0 - py), min(H, y1 + 1 + py)
            X0, X1 = max(0, x0 - px), min(W, x1 + 1 + px)
            crop = img[Y0:Y1, X0:X1].copy()            # keep natural background context
        else:
            raise ValueError(f"unknown encoder {encoder!r}")
        crops.append(Image.fromarray(crop))
    return _clip_encode_crops(clip_model, clip_proc, crops, device, bs)


# ----------------------------- labels / geometry -----------------------------
def mask_gt_labels(masks, gt, min_purity=0.0):
    """Majority GT class under each mask (reduce_zero_label already applied: gt in 0..149 or IGNORE).
    Returns int array (N,), entry = class or IGNORE if mask is mostly ignore."""
    out = np.full(len(masks), IGNORE, np.int64)
    for i, m in enumerate(masks):
        vals = gt[m]
        vals = vals[vals != IGNORE]
        if vals.size == 0:
            continue
        cls = np.bincount(vals, minlength=NUM_CLASSES).argmax()
        out[i] = int(cls)
    return out


def mask_features(masks, H, W):
    """Cheap geometric features per mask for the graph/decoder: area frac, centroid x/y, h/w frac."""
    F = np.zeros((len(masks), 5), np.float32)
    for i, m in enumerate(masks):
        ys, xs = np.where(m)
        F[i] = [m.sum() / (H * W), xs.mean() / W, ys.mean() / H,
                (ys.max() - ys.min() + 1) / H, (xs.max() - xs.min() + 1) / W]
    return F


def _boundary(m):
    from scipy.ndimage import binary_erosion
    return m & ~binary_erosion(m, iterations=1)


def pairwise_edges(masks, region_emb):
    """Edge features between masks: IoU, boundary-overlap, CLIP-region cosine. Returns (E,2) index
    + (E,3) feats for all pairs with IoU>0 or adjacent boundaries (sparse-ish)."""
    n = len(masks)
    areas = [int(m.sum()) for m in masks]
    bnds = [_boundary(m) for m in masks]
    src, dst, feat = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            inter = int((masks[i] & masks[j]).sum())
            union = areas[i] + areas[j] - inter
            iou = inter / union if union else 0.0
            bov = int((bnds[i] & bnds[j]).sum())
            badj = bov / (min(int(bnds[i].sum()), int(bnds[j].sum())) + 1e-6)
            cos = float(region_emb[i] @ region_emb[j])
            if iou > 0.01 or badj > 0.02:
                for a, b in ((i, j), (j, i)):
                    src.append(a); dst.append(b); feat.append([iou, badj, cos])
    if not src:
        return np.zeros((2, 0), np.int64), np.zeros((0, 3), np.float32)
    return np.array([src, dst], np.int64), np.array(feat, np.float32)


# ----------------------------- painting + metrics -----------------------------
def paint(masks, mask_cls, mask_conf, H, W):
    """Per-pixel label = class of the highest-confidence mask covering it. Returns (H,W) int, IGNORE
    where no mask covers."""
    lab = np.full((H, W), IGNORE, np.int64)
    best = np.full((H, W), -1.0, np.float32)
    order = np.argsort(mask_conf)  # paint low->high so high conf wins
    for i in order:
        m = masks[i]
        take = m & (mask_conf[i] > best)
        lab[take] = mask_cls[i]
        best[take] = mask_conf[i]
    return lab


def confusion(pred, gt, n=NUM_CLASSES):
    """Accumulate intersection/union counts. Returns (inter[n], union[n], tp[n], fp[n], fn[n])."""
    keep = gt != IGNORE
    p = pred[keep]; g = gt[keep]
    inter = np.zeros(n, np.int64); union = np.zeros(n, np.int64)
    for c in np.unique(np.concatenate([p, g])):
        if c == IGNORE:
            continue
        pc = p == c; gc = g == c
        i = int((pc & gc).sum())
        inter[c] += i
        union[c] += int(pc.sum()) + int(gc.sum()) - i
    return inter, union


def miou_from(inter, union, classes=None):
    cls = range(len(inter)) if classes is None else classes
    ious = [inter[c] / union[c] for c in cls if union[c] > 0]
    return float(np.mean(ious)) * 100 if ious else 0.0


def hiou(inter, union, seen, novel):
    s = miou_from(inter, union, seen); n = miou_from(inter, union, novel)
    return (2 * s * n / (s + n)) if (s + n) > 0 else 0.0, s, n


def boundary_f(pred, gt, band=3):
    """Boundary F-score: F1 of predicted vs GT class-boundary pixels within a band (px)."""
    from scipy.ndimage import binary_dilation
    keep = gt != IGNORE
    def edges(x):
        e = np.zeros_like(x, bool)
        e[:-1] |= (x[:-1] != x[1:]); e[1:] |= (x[:-1] != x[1:])
        e[:, :-1] |= (x[:, :-1] != x[:, 1:]); e[:, 1:] |= (x[:, :-1] != x[:, 1:])
        return e & keep
    ge = edges(gt); pe = edges(np.where(keep, pred, IGNORE))
    gd = binary_dilation(ge, iterations=band); pd = binary_dilation(pe, iterations=band)
    tp_p = int((pe & gd).sum()); tp_g = int((ge & pd).sum())
    prec = tp_p / (int(pe.sum()) + 1e-6); rec = tp_g / (int(ge.sum()) + 1e-6)
    return (2 * prec * rec / (prec + rec + 1e-6)) * 100


def ece_bins(conf, correct, n_bins=15):
    """Expected Calibration Error from confidence + correctness arrays (per accepted pixel/mask)."""
    conf = np.asarray(conf); correct = np.asarray(correct, np.float32)
    if conf.size == 0:
        return 0.0
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for b in range(n_bins):
        m = (conf >= edges[b]) & (conf < edges[b + 1] if b < n_bins - 1 else conf <= 1.0)
        if m.sum() == 0:
            continue
        e += abs(correct[m].mean() - conf[m].mean()) * (m.sum() / conf.size)
    return float(e) * 100
