# Stage 7: MECHANISM DIAGNOSTICS + confound ablation.
#
# Measures the seen-versus-novel logit-margin shift and the change in frozen CLIP alignment, and
# separates seen-only supervision from class imbalance, mixed masks, objective design and
# logit-scale mismatch.
#
# We hold the head fixed and vary ONE confound at a time. If the collapse survives every ablation,
# seen-only supervision is the cause; if any ablation removes it, that confound is the cause.
#
#   C1 class imbalance     inverse-frequency class weighting of the seen-class CE
#   C2 objective design    CE  vs  focal  vs  cosine-margin
#   C3 logit scale         fixed LS=40  vs  learned logit scale
#   C4 residual bound      trust region tau in {0.5, 1, 2, 4, inf}
#   C5 supervision breadth n_seen in {30, 60, 90, 120}  -- does MORE diverse seen data recover novel?
#   C6 label reliability   train only on spatially PURE patches (proxy for mixed/noisy regions)
#
# Writes $PMX_CACHE/metrics_diag.json
import os, glob, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

dev = 'cuda'
CACHE = os.environ.get('PMX_CACHE', 'cache')
NCLS, LS, BINS = 150, 40.0, 15
NOVEL = int(os.environ.get('PMX_NOVEL', 30))
SEEDS = int(os.environ.get('PMX_SEEDS', 5))
EPOCHS = int(os.environ.get('PMX_EPOCHS', 12))
BS = 8192


def load(split, with_grid=False):
    out = []
    for f in sorted(glob.glob(f'{CACHE}/{split}/*.npz')):
        d = np.load(f)
        rec = [torch.from_numpy(d['feat'].astype(np.float32)),
               torch.from_numpy(d['clslogit'].astype(np.float32)),
               torch.from_numpy(d['gt'].astype(np.int64))]
        if with_grid:
            rec.append(tuple(int(x) for x in d['grid']))
        out.append(tuple(rec))
    return out


print('loading cache...', flush=True)
VAL = load('val')
TR = load('train', with_grid=True)


def purity_mask(gt_flat, grid):
    """Fraction of the 3x3 patch neighbourhood sharing this patch's label (>=0.75 == 'pure')."""
    gh, gw = grid
    g = gt_flat.view(gh, gw)
    pad = F.pad(g.unsqueeze(0).unsqueeze(0).float(), (1, 1, 1, 1), mode='replicate')[0, 0]
    agree = torch.zeros_like(g, dtype=torch.float32)
    for dy in range(3):
        for dx in range(3):
            agree += (pad[dy:dy + gh, dx:dx + gw] == g).float()
    return (agree / 9.0).view(-1)


vf = torch.cat([x[0] for x in VAL]); vl = torch.cat([x[1] for x in VAL]).to(dev)
vg = torch.cat([x[2] for x in VAL])
vgc = (vg - 1).clamp(min=0).to(dev).long(); vvalid = (vg > 0).to(dev)

tf = torch.cat([x[0] for x in TR]); tl = torch.cat([x[1] for x in TR])
tg = torch.cat([x[2] for x in TR])
tpur = torch.cat([purity_mask(x[2], x[3]) for x in TR])
tmask = tg > 0
tl_ = tl[tmask].to(dev); tgc = (tg[tmask] - 1).to(dev).long()
tf_ = tf[tmask]; tpur_ = tpur[tmask].to(dev)
D = tf_.shape[1]
BASE_VAL = F.softmax(LS * vl, 1)
print(f'val={vl.shape[0]} train={tl_.shape[0]} pure_frac={100*(tpur_>=0.75).float().mean():.1f}%', flush=True)


def metrics(probs, novel, seen):
    p = probs[vvalid]; g = vgc[vvalid]
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
    bp = BASE_VAL[vvalid]
    drift = 100 * (1 - F.cosine_similarity(p, bp, dim=1)).mean().item()
    return dict(mIoU=100 * iou[~torch.isnan(iou)].mean().item(), seenIoU=siou, novelIoU=niou,
                hIoU=(2 * siou * niou / (siou + niou)) if (siou + niou) > 0 else 0.0,
                ECE=ece(torch.ones_like(isn)), ECE_novel=ece(isn), drift=drift,
                novel_rate=100 * torch.isin(pred, nov_t).float().mean().item())


class ResHead(nn.Module):
    def __init__(self, d, hid=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hid), nn.ReLU(), nn.Linear(hid, NCLS))

    def forward(self, x):
        return self.net(x)


def train(X, Z, Y, objective='ce', weights=None, tau=None, learn_scale=False, epochs=EPOCHS):
    head = ResHead(D).to(dev)
    params = list(head.parameters())
    logit_scale = torch.tensor([float(np.log(LS))], device=dev, requires_grad=learn_scale)
    if learn_scale:
        params.append(logit_scale)
    op = torch.optim.Adam(params, lr=1e-3)
    for _ in range(epochs):
        for i in range(0, X.shape[0], BS):
            op.zero_grad()
            x = X[i:i + BS].to(dev)
            sc = logit_scale.exp() if learn_scale else LS
            r = head(x)
            if tau is not None:
                r = torch.clamp(r, -tau, tau)
            z = sc * Z[i:i + BS] / LS + r if learn_scale else Z[i:i + BS] + r
            y = Y[i:i + BS]
            if objective == 'ce':
                loss = F.cross_entropy(z, y, weight=weights)
            elif objective == 'focal':
                ce = F.cross_entropy(z, y, weight=weights, reduction='none')
                loss = ((1 - torch.exp(-ce)) ** 2 * ce).mean()
            elif objective == 'cosmargin':
                zz = z.clone()
                zz[torch.arange(len(y), device=dev), y] -= 4.0
                loss = F.cross_entropy(zz, y, weight=weights)
            loss.backward(); op.step(); del x
    return head, (logit_scale.exp().item() if learn_scale else LS)


@torch.no_grad()
def apply_head(head, scale=LS, tau=None):
    outs = []
    for a in range(0, vf.shape[0], 200_000):
        b = min(a + 200_000, vf.shape[0])
        x = vf[a:b].to(dev)
        r = head(x)
        if tau is not None:
            r = torch.clamp(r, -tau, tau)
        outs.append(F.softmax(scale * vl[a:b] + r, 1))
        del x
    return torch.cat(outs)


rng = np.random.RandomState(0)
rows = {}


def rec(n, m):
    rows.setdefault(n, []).append(m)


for s in range(SEEDS):
    perm = rng.permutation(NCLS)
    novel = sorted(perm[:NOVEL].tolist()); seen = sorted(perm[NOVEL:].tolist())
    tseen = torch.isin(tgc, torch.tensor(seen, device=dev))
    Xs, Zs, Ys = tf_[tseen.cpu()], LS * tl_[tseen], tgc[tseen]
    print(f'\n===== split {s} =====', flush=True)
    rec('uncal', metrics(F.softmax(LS * vl, 1), novel, seen))

    # baseline seen-only head (the audited collapse)
    h, _ = train(Xs, Zs, Ys, 'ce')
    rec('head_ce', metrics(apply_head(h), novel, seen)); del h

    # C1 class imbalance
    cnt = torch.bincount(Ys, minlength=NCLS).float()
    w = torch.where(cnt > 0, 1.0 / cnt.clamp(min=1), torch.zeros_like(cnt))
    w = (w / w[w > 0].mean()).to(dev)
    h, _ = train(Xs, Zs, Ys, 'ce', weights=w)
    rec('C1_class_balanced', metrics(apply_head(h), novel, seen)); del h

    # C2 objective design
    for obj in ['focal', 'cosmargin']:
        h, _ = train(Xs, Zs, Ys, obj)
        rec(f'C2_{obj}', metrics(apply_head(h), novel, seen)); del h

    # C3 learned logit scale
    h, sc = train(Xs, Zs, Ys, 'ce', learn_scale=True)
    rec('C3_learned_scale', metrics(apply_head(h, scale=sc), novel, seen)); del h

    # C4 trust region
    for tau in [0.5, 1.0, 2.0, 4.0]:
        h, _ = train(Xs, Zs, Ys, 'ce', tau=tau)
        rec(f'C4_tau{tau}', metrics(apply_head(h, tau=tau), novel, seen)); del h

    # C5 supervision breadth: how many seen classes does the head get to learn from?
    for nseen in [30, 60, 90]:
        sub = sorted(np.random.RandomState(100 + s).choice(seen, nseen, replace=False).tolist())
        m = torch.isin(tgc, torch.tensor(sub, device=dev))
        h, _ = train(tf_[m.cpu()], LS * tl_[m], tgc[m], 'ce')
        rec(f'C5_nseen{nseen}', metrics(apply_head(h), novel, seen)); del h
    rec('C5_nseen120', rows['head_ce'][-1])

    # C6 label reliability: pure patches only
    mpure = tseen & (tpur_ >= 0.75)
    h, _ = train(tf_[mpure.cpu()], LS * tl_[mpure], tgc[mpure], 'ce')
    rec('C6_pure_only', metrics(apply_head(h), novel, seen)); del h
    torch.cuda.empty_cache()

    for k in ['uncal', 'head_ce', 'C1_class_balanced', 'C5_nseen30', 'C5_nseen120', 'C6_pure_only']:
        r = rows[k][-1]
        print(f"  {k:20s} novelIoU={r['novelIoU']:5.1f} seenIoU={r['seenIoU']:5.1f} "
              f"drift={r['drift']:5.1f} novel%={r['novel_rate']:5.2f}", flush=True)


def agg(v):
    return {k: {'mean': round(float(np.nanmean([r[k] for r in v])), 2),
                'std': round(float(np.nanstd([r[k] for r in v], ddof=1)), 2)} for k in v[0]}


res = {k: agg(v) for k, v in rows.items()}
base = res['uncal']['novelIoU']['mean']
for k in res:
    res[k]['collapse_index'] = round(res[k]['novelIoU']['mean'] / base, 3) if base else None
json.dump({'config': {'dataset': 'ADE20K-150 dense-CLIP', 'seeds': SEEDS, 'novel': NOVEL},
           'summary': res}, open(f'{CACHE}/metrics_diag.json', 'w'), indent=1)
print('\n=== STAGE7 CONFOUND ABLATION ===')
for k, r in res.items():
    print(f"{k:20s} mIoU={r['mIoU']['mean']:5.1f} seen={r['seenIoU']['mean']:5.1f} "
          f"novel={r['novelIoU']['mean']:5.1f} drift={r['drift']['mean']:5.1f} CI={r['collapse_index']}")
print('WROTE metrics_diag.json', flush=True)
