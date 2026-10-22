# Stage 8: direct comparison against LPOSS (Stojnic et al., CVPR'25) under our protocol, with the
# same backbone, data and evaluation, AND the stronger-base audit: does a seen-only head stacked on
# the LPOSS-corrected logits still collapse novel classes at this stronger operating point?
#
# We run the AUTHORS' OWN propagation code, extracted verbatim by scripts/lposs_extract.py, on our
# cached dense-CLIP logits with DINO ViT-B/16 affinities built exactly as in their LPOSS class
# (torch.hub dino_vitb16, hook on the last block's attn.qkv, 'v' features, CLS dropped, L2-normalised).
# Their call site propagates RAW cosine scores (clip_feats @ clf.T), which is exactly what our
# cache stores in `clslogit`, so the input to the propagation is faithful.
#
# Config from the authors' configs/lposs.yaml: alpha=0.95 gamma=3.0 k=400 sigma=0.01
# pix_dist_pow=1.0 patch=16, pixel_refine=false (LPOSS, not LPOSS+).
#
# Rows, over the same five seen/novel splits used everywhere else:
#   uncal / diffusion / laplacian / LPOSS            (method comparison, matched protocol)
#   base+head / LPOSS+head                           (seen-only residual head stacked on each base)
#
# Writes --out (default $PMX_CACHE/metrics_lposs.json)
import os, sys, glob, json, time, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- environment shim (does NOT touch the authors' code) -------------------------------------
# The prebuilt faiss GPU wheels ship no sm_90 kernels (CUDA error 209 on an H100) and their bundled
# cublas conflicts with the container's torch. We therefore run faiss-cpu for the module import and
# replace faiss.knn_gpu with an EXACT torch equivalent, leaving lposs_core_auto.py byte-identical to
# the authors' released source. Both compute exact (brute-force) inner-product k-NN over
# L2-normalised features and return (similarities, indices) sorted descending, so the resulting
# graph is identical up to tie-ordering.
import faiss  # noqa: E402


def _knn_exact_torch(res, xq, xb, k, metric=None):
    sims = xq @ xb.T
    d, i = torch.topk(sims, k, dim=1, largest=True, sorted=True)
    return d, i


class _DummyGpuRes:
    """Stands in for faiss.StandardGpuResources (absent in faiss-cpu); unused by the torch kNN."""

    def setDefaultNullStreamAllDevices(self):
        pass

    def setTempMemory(self, n):
        pass


from lposs_core_auto import get_lposs_laplacian, perform_lp   # authors' verbatim code

# Applied AFTER the import: lposs_core_auto imports faiss.contrib.torch_utils, which installs its
# own knn_gpu replacement at import time and would otherwise clobber these. The authors' functions
# resolve faiss.knn_gpu at call time, so patching afterwards is what takes effect.
faiss.knn_gpu = _knn_exact_torch
faiss.StandardGpuResources = _DummyGpuRes
print('[shim] faiss.knn_gpu -> exact torch top-k; StandardGpuResources -> no-op', flush=True)

dev = 'cuda'
CACHE = os.environ.get('PMX_CACHE', 'dcache')
ADE = os.environ.get('PMX_ADE_ROOT', 'data/ADEChallengeData2016')
NCLS, LS, BINS = 150, 40.0, 15
NOVEL = int(os.environ.get('PMX_NOVEL', 30))
SEEDS = int(os.environ.get('PMX_SEEDS', 5))
EPOCHS, BS = int(os.environ.get('PMX_EPOCHS', 12)), 8192
# authors' configs/lposs.yaml
ALPHA, GAMMA, KNN, SIGMA, PIX_POW, PATCH = 0.95, 3.0, 400, 0.01, 1.0, 16
# pm_stage1 image geometry (must match, or DINO patches will not align with the cached CLIP patches)
SHORT, MAXL = 336, 2048
IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---------------------------------------------------------------- DINO (as in LPOSS.get_dino_features)
class Dino:
    def __init__(self):
        self.net = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16').to(dev).eval()
        self.hook = {}
        self.net._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(
            lambda m, i, o: self.hook.__setitem__("qkv", o))

    @torch.no_grad()
    def feats(self, img):                       # img [1,3,H,W] normalised, H/W divisible by 16
        _ = self.net(img)
        nh = self.net.blocks[-1].attn.num_heads
        nb_im, nb_tokens, C_qkv = self.hook["qkv"].shape
        qkv = (self.hook["qkv"].reshape(nb_im, nb_tokens, 3, nh, C_qkv // nh // 3)
               .permute(2, 0, 3, 1, 4))
        v = qkv[2].transpose(1, 2).float()      # enc_type_feats='v'
        v = v[:, 1:, :, :].flatten(-2, -1).permute(0, 2, 1)   # drop CLS -> [B, C, N]
        v = v / v.norm(dim=1, keepdim=True)
        return v[0].T.contiguous()              # [N, C]


def load_img_for_dino(path, gh, gw):
    """Same resize as pm_stage1.load_img, cropped to the cached patch grid so DINO patches align 1:1."""
    im = Image.open(path).convert('RGB')
    w, h = im.size
    s = SHORT / min(w, h)
    if max(w, h) * s > MAXL:
        s = MAXL / max(w, h)
    im = im.resize((round(w * s), round(h * s)), Image.BILINEAR)
    t = torch.from_numpy(np.array(im)).permute(2, 0, 1).float().unsqueeze(0) / 255.
    t = t[:, :, :gh * PATCH, :gw * PATCH]       # pm_stage1 uses floor division for the grid
    return ((t - IMNET_MEAN) / IMNET_STD).to(dev)


# ---------------------------------------------------------------- our cache / metrics
def load_cache(split):
    out = []
    for f in sorted(glob.glob(f'{CACHE}/{split}/*.npz')):
        d = np.load(f)
        out.append((torch.from_numpy(d['feat'].astype(np.float32)),
                    torch.from_numpy(d['clslogit'].astype(np.float32)),
                    torch.from_numpy(d['gt'].astype(np.int64)),
                    tuple(int(x) for x in d['grid'])))
    return out


def metrics(probs, novel, seen, base):
    p = probs[VV]; g = VGC[VV]
    conf, pred = p.max(1); correct = (pred == g).float()

    def ece(sel):
        c, a = conf[sel], correct[sel]
        if c.numel() == 0:
            return float('nan')
        e = 0.
        for b in range(BINS):
            m = (c > b / BINS) & (c <= (b + 1) / BINS)
            if m.any():
                e += (m.float().mean() * (a[m].mean() - c[m].mean()).abs()).item()
        return 100 * e

    nov_t = torch.tensor(novel, device=dev); seen_t = torch.tensor(seen, device=dev)
    isn = torch.zeros_like(g, dtype=torch.bool)
    for c in novel:
        isn |= (g == c)
    inter = torch.zeros(NCLS, device=dev); union = torch.zeros(NCLS, device=dev)
    for c in range(NCLS):
        inter[c] = ((g == c) & (pred == c)).sum(); union[c] = ((g == c) | (pred == c)).sum()
    iou = torch.where(union > 0, inter / union, torch.full_like(inter, float('nan')))
    nv = nov_t[~torch.isnan(iou[nov_t])]; sv = seen_t[~torch.isnan(iou[seen_t])]
    niou = 100 * iou[nv].mean().item() if nv.numel() else float('nan')
    siou = 100 * iou[sv].mean().item() if sv.numel() else float('nan')
    bp = base[VV]
    return dict(mIoU=100 * iou[~torch.isnan(iou)].mean().item(), seenIoU=siou, novelIoU=niou,
                hIoU=(2 * siou * niou / (siou + niou)) if (siou + niou) > 0 else 0.0,
                ECE=ece(torch.ones_like(isn)), ECE_novel=ece(isn),
                drift=100 * (1 - F.cosine_similarity(p, bp, dim=1)).mean().item(),
                novel_rate=100 * torch.isin(pred, nov_t).float().mean().item())


@torch.no_grad()
def graph(feat, knn=8):
    f = F.normalize(feat.to(dev).float(), dim=1)
    sim = f @ f.T; n = f.shape[0]; kk = min(knn, n - 1)
    if kk < 1:
        return None
    val, idx = sim.topk(kk + 1, dim=1); val, idx = val[:, 1:], idx[:, 1:]
    W = torch.zeros(n, n, device=dev).scatter_(1, idx, val.clamp(min=0))
    W = (W + W.T) / 2
    return W / W.sum(1, keepdim=True).clamp(min=1e-6)


@torch.no_grad()
def diffuse(feat, logit, alpha=0.4, k=2, anchored=True):
    W = graph(feat); P = F.softmax(LS * logit.to(dev).float(), 1)
    if W is None:
        return P
    out = P
    for _ in range(k):
        out = ((1 - alpha) * P + alpha * (W @ out)) if anchored else (W @ out)
    return out


class ResHead(nn.Module):
    """The audited seen-only residual head (identical architecture to pm_stage3/pm_stage5)."""

    def __init__(self, d, hid=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hid), nn.ReLU(), nn.Linear(hid, NCLS))

    def forward(self, x):
        return self.net(x)


def train_head(X, Z, Y, epochs=EPOCHS):
    """CE on seen-class patches only; Z are the base logits ALREADY scaled (LS*...)."""
    h = ResHead(X.shape[1]).to(dev)
    op = torch.optim.Adam(h.parameters(), lr=1e-3)
    for _ in range(epochs):
        for i in range(0, X.shape[0], BS):
            op.zero_grad()
            x = X[i:i + BS].to(dev)
            loss = F.cross_entropy(Z[i:i + BS] + h(x), Y[i:i + BS])
            loss.backward(); op.step(); del x
    return h


@torch.no_grad()
def apply_head(h, feats, logits_scaled):
    outs = []
    for a in range(0, feats.shape[0], 200_000):
        b = min(a + 200_000, feats.shape[0])
        outs.append(F.softmax(logits_scaled[a:b] + h(feats[a:b].to(dev)), 1))
    return torch.cat(outs)


def propagate_split(items, img_paths, dino, tag):
    """Authors' verbatim LPOSS over every image of a cached split; returns raw propagated logits."""
    lp = []
    t0 = time.time()
    for i, (feat, logit, gt, (gh, gw)) in enumerate(items):
        n = feat.shape[0]
        assert n == gh * gw, f'grid mismatch {tag} img{i}: {n} != {gh}x{gw}'
        d = dino.feats(load_img_for_dino(img_paths[i], gh, gw))
        if d.shape[0] != n:
            raise SystemExit(f'DINO/CLIP patch mismatch on {tag} img {i}: {d.shape[0]} vs {n}')
        preds = logit.to(dev).float()            # RAW cosine scores, as at the authors' call site
        L = get_lposs_laplacian(d, torch.zeros((1, 4), device=dev), [(gh, gw)],
                                sigma=SIGMA, pix_dist_pow=PIX_POW, k=min(KNN, n - 1), gamma=GAMMA,
                                alpha=ALPHA, patch_size=PATCH)
        lp.append(perform_lp(L, preds).float().cpu())
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f'  [{tag} {i+1}/{len(items)}] {el:.0f}s ({el/(i+1):.2f}s/img)', flush=True)
    return torch.cat(lp), (time.time() - t0) / max(1, len(items))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0, help='pilot on the first N val images (0 = all)')
    ap.add_argument('--out', default=os.path.join(CACHE, 'metrics_lposs.json'))
    a = ap.parse_args()

    global VGC, VV
    print('loading cache...', flush=True)
    VAL = load_cache('val')
    TR = load_cache('train')
    if a.limit:
        VAL, TR = VAL[:a.limit], TR[:max(10, a.limit // 2)]
    val_imgs = sorted(glob.glob(f'{ADE}/images/validation/*.jpg'))
    tr_imgs = sorted(glob.glob(f'{ADE}/images/training/*.jpg'))[:len(TR)]
    print(f'val imgs={len(VAL)} train imgs={len(TR)}', flush=True)

    # Propagated logits are expensive (~4.5 s/img); persist them so a rerun never repeats the work.
    lpv_p = os.path.join(CACHE, f'lposs_val_{len(VAL)}.npy')
    lpt_p = os.path.join(CACHE, f'lposs_train_{len(TR)}.npy')
    if os.path.exists(lpv_p) and os.path.exists(lpt_p):
        print(f'reusing persisted LPOSS logits: {lpv_p}, {lpt_p}', flush=True)
        LPv = torch.from_numpy(np.load(lpv_p)); LPt = torch.from_numpy(np.load(lpt_p))
        spimg_v = spimg_t = 0.0
    else:
        dino = Dino()
        LPv, spimg_v = propagate_split(VAL, val_imgs, dino, 'val')       # [Nval,150] raw
        np.save(lpv_p, LPv.numpy().astype(np.float16))
        LPt, spimg_t = propagate_split(TR, tr_imgs, dino, 'train')       # [Ntr,150] raw
        np.save(lpt_p, LPt.numpy().astype(np.float16))
        del dino
    LPv = LPv.float(); LPt = LPt.float()
    torch.cuda.empty_cache()

    vf = torch.cat([x[0] for x in VAL])
    vl = torch.cat([x[1] for x in VAL]).to(dev)
    VG = torch.cat([x[2] for x in VAL])
    VGC = (VG - 1).clamp(min=0).to(dev).long()
    VV = (VG > 0).to(dev)
    LPv = LPv.to(dev)

    tf = torch.cat([x[0] for x in TR])
    tl = torch.cat([x[1] for x in TR])
    tg = torch.cat([x[2] for x in TR])
    tm = tg > 0
    tf_, tl_, tgc = tf[tm], tl[tm].to(dev), (tg[tm] - 1).to(dev).long()
    LPt_ = LPt[tm.cpu()].to(dev)

    P_base = F.softmax(LS * vl, 1)
    P_lp = F.softmax(LS * LPv, 1)
    P_diff = torch.cat([diffuse(f, l, anchored=True) for (f, l, g, _) in VAL])
    P_lap = torch.cat([diffuse(f, l, anchored=False) for (f, l, g, _) in VAL])

    rng = np.random.RandomState(0)               # identical split RNG to every other stage
    rows = {}
    for s in range(SEEDS):
        perm = rng.permutation(NCLS)
        novel = sorted(perm[:NOVEL].tolist()); seen = sorted(perm[NOVEL:].tolist())
        tseen = torch.isin(tgc, torch.tensor(seen, device=dev))
        Xs, Ys = tf_[tseen.cpu()], tgc[tseen]

        for name, P in [('uncal', P_base), ('diffusion', P_diff),
                        ('laplacian', P_lap), ('LPOSS', P_lp)]:
            rows.setdefault(name, []).append(metrics(P, novel, seen, P_base))

        # seen-only residual head stacked on each base (the audit, at two operating points)
        h_raw = train_head(Xs, LS * tl_[tseen], Ys)
        rows.setdefault('base+head', []).append(
            metrics(apply_head(h_raw, vf, LS * vl), novel, seen, P_base))
        del h_raw
        h_lp = train_head(Xs, LS * LPt_[tseen], Ys)
        rows.setdefault('LPOSS+head', []).append(
            metrics(apply_head(h_lp, vf, LS * LPv), novel, seen, P_lp))
        del h_lp
        torch.cuda.empty_cache()

        r = {k: rows[k][-1] for k in rows}
        print(f"[split {s}] LPOSS mIoU={r['LPOSS']['mIoU']:.1f} novel={r['LPOSS']['novelIoU']:.1f} | "
              f"LPOSS+head novel={r['LPOSS+head']['novelIoU']:.1f} | "
              f"base+head novel={r['base+head']['novelIoU']:.1f}", flush=True)

    def agg(v):
        return {k: {'mean': round(float(np.nanmean([r[k] for r in v])), 2),
                    'std': round(float(np.nanstd([r[k] for r in v], ddof=1)), 2)} for k in v[0]}

    res = {k: agg(v) for k, v in rows.items()}
    b = res['uncal']['novelIoU']['mean']
    for k in res:
        res[k]['collapse_index'] = round(res[k]['novelIoU']['mean'] / b, 3) if b else None
    if res.get('LPOSS', {}).get('novelIoU', {}).get('mean'):
        res['LPOSS+head']['collapse_index_vs_LPOSS'] = round(
            res['LPOSS+head']['novelIoU']['mean'] / res['LPOSS']['novelIoU']['mean'], 3)
    json.dump({'config': dict(alpha=ALPHA, gamma=GAMMA, k=KNN, sigma=SIGMA, pix_dist_pow=PIX_POW,
                              patch=PATCH, n_val=len(VAL), n_train=len(TR), seeds=SEEDS, novel=NOVEL,
                              sec_per_img_val=round(spimg_v, 2), sec_per_img_train=round(spimg_t, 2),
                              source='authors verbatim (scripts/lposs_extract.py)'),
               'summary': res}, open(a.out, 'w'), indent=1)
    print('\n=== STAGE8 (LPOSS, authors’ code, our protocol) ===')
    print(f"{'method':12s} {'mIoU':>6s} {'seen':>6s} {'novel':>6s} {'hIoU':>6s} {'ECE':>6s} {'drift':>6s} {'CI':>5s}")
    for k, r in res.items():
        print(f"{k:12s} {r['mIoU']['mean']:6.1f} {r['seenIoU']['mean']:6.1f} {r['novelIoU']['mean']:6.1f} "
              f"{r['hIoU']['mean']:6.1f} {r['ECE']['mean']:6.1f} {r['drift']['mean']:6.1f} {r['collapse_index']}")
    print(f'WROTE {a.out}', flush=True)


if __name__ == '__main__':
    main()
