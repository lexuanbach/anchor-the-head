# Stage 5: ALIGNMENT-PRESERVING LEARNED BASELINES + the novel-class frontier.
#
# Learned corrections explicitly constrained to preserve the frozen CLIP alignment.
# This script runs them under the IDENTICAL frozen-stack protocol as pm_stage3.py (same cache, same
# 5 seen/novel splits, same seen-only supervision, same metrics), so the new rows are directly
# comparable to the other rows.
#
#   B1  frozen-LOGIT distillation head   CE_seen + lam * KL(frozen || head) over ALL classes/patches
#   B2  EMBEDDING-space distillation     class-agnostic: head corrects the feature, classification
#                                        stays cos(e', text_c); CE_seen + lam*(1-cos(e', e_clip))
#   B3  output-space GATED mixture       p = (1-g) p_frozen + g p_head   (g grid + learned per-patch)
#   B4  weight-space interpolation       last-layer weights scaled toward the zero-residual head
#                                        (the WiSE-FT analogue when "zero-shot" == no correction)
#   B5  pseudo-novel EPISODIC training   hide 30 of the 120 seen classes per episode, preserve them
#   B6  learned CLASS-AGNOSTIC affinity  learn the graph edge weights, never the class logits
#
# Reference rows re-run here so every number in the table comes from one process:
#   uncalibrated / global T / anchored diffusion / generic Laplacian / current feature-space decoder.
#
# Diagnostics reported for EVERY row: seen-novel margin shift, CLIP-alignment drift,
# novel prediction rate, collapse index.
#
# Writes $PMX_CACHE/metrics_align.json
import os, glob, json, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

dev = 'cuda'
CACHE = os.environ.get('PMX_CACHE', 'cache')
NCLS, LS, BINS = 150, 40.0, 15
NOVEL = int(os.environ.get('PMX_NOVEL', 30))
SEEDS = int(os.environ.get('PMX_SEEDS', 5))
LAMBDAS = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]     # distillation strength sweep (B1/B2)
GATES   = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]            # output-gate sweep (B3)
ALPHAS  = [0.0, 0.25, 0.5, 0.75, 1.0]               # weight-interp sweep (B4)
EPOCHS = int(os.environ.get('PMX_EPOCHS', 12))
BS, TAU = 8192, 2.0
AFFINITY_IMGS = int(os.environ.get('PMX_AFFIMGS', 150))                                   # train images used for the B6 affinity head


def load(split):
    out = []
    for f in sorted(glob.glob(f'{CACHE}/{split}/*.npz')):
        d = np.load(f)
        out.append((torch.from_numpy(d['feat'].astype(np.float32)),
                    torch.from_numpy(d['clslogit'].astype(np.float32)),
                    torch.from_numpy(d['gt'].astype(np.int64))))
    return out


print('loading cache...', flush=True)
VAL, TR = load('val'), load('train')
print(f'val imgs={len(VAL)} train imgs={len(TR)}', flush=True)
qfeat = torch.from_numpy(np.load(f'{CACHE}/qfeat.npy')).float().to(dev)          # [Nq, D] text protos


def cat(a):
    return (torch.cat([x[0] for x in a]), torch.cat([x[1] for x in a]), torch.cat([x[2] for x in a]))


vf, vl, vg = cat(VAL)
tf, tl, tg = cat(TR)
vl = vl.to(dev)
vgc = (vg - 1).clamp(min=0).to(dev).long()
vvalid = (vg > 0).to(dev)
tmask = tg > 0
tl_ = tl[tmask].to(dev)                 # ALL valid train patches (label-free KD term uses these)
tgc = (tg[tmask] - 1).to(dev).long()
tf_ = tf[tmask]                         # kept on CPU; moved in chunks
D = tf_.shape[1]
print(f'val patches={vl.shape[0]} train patches={tl_.shape[0]} D={D}', flush=True)

BASE_VAL = F.softmax(LS * vl, 1)        # frozen reference distribution, for drift/margin diagnostics


# ---------------------------------------------------------------- metrics + diagnostics
def metrics(probs, novel, seen, base=BASE_VAL):
    p = probs[vvalid]
    g = vgc[vvalid]
    conf, pred = p.max(1)
    correct = (pred == g).float()

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

    nov_t = torch.tensor(novel, device=dev)
    seen_t = torch.tensor(seen, device=dev)
    isn = torch.zeros_like(g, dtype=torch.bool)
    for c in novel:
        isn |= (g == c)

    inter = torch.zeros(NCLS, device=dev); union = torch.zeros(NCLS, device=dev)
    for c in range(NCLS):
        inter[c] = ((g == c) & (pred == c)).sum()
        union[c] = ((g == c) | (pred == c)).sum()
    iou = torch.where(union > 0, inter / union, torch.full_like(inter, float('nan')))
    ok = ~torch.isnan(iou)
    miou = 100 * iou[ok].mean().item()
    nv = nov_t[~torch.isnan(iou[nov_t])]
    sv = seen_t[~torch.isnan(iou[seen_t])]
    niou = 100 * iou[nv].mean().item() if nv.numel() else float('nan')
    siou = 100 * iou[sv].mean().item() if sv.numel() else float('nan')
    hiou = (2 * siou * niou / (siou + niou)) if (niou == niou and siou == siou and siou + niou > 0) else 0.0

    # --- diagnostics
    bp = base[vvalid]
    drift = (1 - F.cosine_similarity(p, bp, dim=1))                    # alignment drift vs frozen
    drift_all = 100 * drift.mean().item()
    drift_nov = 100 * drift[isn].mean().item() if isn.any() else float('nan')
    novel_rate = 100 * torch.isin(pred, nov_t).float().mean().item()   # novel prediction mass
    base_rate = 100 * torch.isin(bp.argmax(1), nov_t).float().mean().item()
    # seen-minus-novel max-logit margin, before vs after
    m_after = (p[:, seen_t].max(1).values - p[:, nov_t].max(1).values).mean().item()
    m_before = (bp[:, seen_t].max(1).values - bp[:, nov_t].max(1).values).mean().item()

    return dict(mIoU=miou, seenIoU=siou, novelIoU=niou, hIoU=hiou,
                ECE=ece(torch.ones_like(isn)), ECE_novel=ece(isn), ECE_seen=ece(~isn),
                drift=drift_all, drift_novel=drift_nov,
                novel_rate=novel_rate, novel_rate_base=base_rate,
                margin_shift=100 * (m_after - m_before))


# ---------------------------------------------------------------- reference methods
def fit_gT(logit, gc):
    T = torch.ones(1, device=dev, requires_grad=True)
    o = torch.optim.LBFGS([T], lr=0.1, max_iter=50)
    def cl():
        o.zero_grad(); l = F.cross_entropy(LS * logit / T.clamp(min=0.05), gc); l.backward(); return l
    o.step(cl)
    return T.detach().clamp(min=0.05).item()


@torch.no_grad()
def graph(feat, knn=8, wmlp=None):
    f = F.normalize(feat.to(dev).float(), dim=1)
    sim = f @ f.T
    n = f.shape[0]
    kk = min(knn, n - 1)
    if kk < 1:
        return None
    val, idx = sim.topk(kk + 1, dim=1)
    val, idx = val[:, 1:], idx[:, 1:]
    if wmlp is not None:                      # B6: learned CLASS-AGNOSTIC edge weights
        fi = f.unsqueeze(1).expand(-1, kk, -1)
        fj = f[idx]
        pair = torch.stack([(fi * fj).sum(-1), (fi - fj).norm(dim=-1)], dim=-1)   # [n,kk,2]
        val = F.softplus(wmlp(pair).squeeze(-1))
    W = torch.zeros(n, n, device=dev)
    W.scatter_(1, idx, val.clamp(min=0))
    W = (W + W.T) / 2
    return W / W.sum(1, keepdim=True).clamp(min=1e-6)


@torch.no_grad()
def diffuse(feat, logit, alpha=0.4, k=2, anchored=True, wmlp=None):
    W = graph(feat, wmlp=wmlp)
    P = F.softmax(LS * logit.to(dev).float(), 1)
    if W is None:
        return P
    out = P
    for _ in range(k):
        out = ((1 - alpha) * P + alpha * (W @ out)) if anchored else (W @ out)
    return out


def val_diffuse(anchored=True, wmlp=None):
    return torch.cat([diffuse(f, l, anchored=anchored, wmlp=wmlp) for (f, l, g) in VAL])


# ---------------------------------------------------------------- heads
class ResHead(nn.Module):
    """Residual correction in logit space (the architecture audited in the paper)."""
    def __init__(self, d, out=NCLS, hid=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hid), nn.ReLU(), nn.Linear(hid, out))

    def forward(self, x):
        return self.net(x)


def chunks(n, bs=BS):
    for i in range(0, n, bs):
        yield i, min(i + bs, n)


@torch.no_grad()
def apply_head(head, scale=1.0, mode='logit'):
    """Run a trained head over the val set in chunks; return probabilities."""
    outs = []
    for a, b in chunks(vf.shape[0], 200_000):
        x = vf[a:b].to(dev)
        z = LS * vl[a:b]
        if mode == 'logit':
            outs.append(F.softmax(z + scale * head(x), 1))
        elif mode == 'emb':                              # B2: correct the feature, classify by text
            e = F.normalize(x + scale * head(x), dim=1)
            outs.append(F.softmax(LS * (e @ F.normalize(qfeat, dim=1).T), 1))
        del x
    return torch.cat(outs)


def train_kd(tl_seen, tg_seen, tf_seen, lam, mode='logit', epochs=EPOCHS):
    """B1 (mode='logit') / B2 (mode='emb'): CE on seen + label-free distillation toward frozen."""
    out_dim = NCLS if mode == 'logit' else D
    head = ResHead(D, out_dim).to(dev)
    op = torch.optim.Adam(head.parameters(), lr=1e-3)
    Xs, Zs, Ys = tf_seen, LS * tl_seen, tg_seen                     # supervised (seen only)
    Xa, Za = tf_, LS * tl_                                          # ALL patches (KD, label-free)
    qn = F.normalize(qfeat, dim=1)
    for _ in range(epochs):
        perm_a = torch.randperm(Xa.shape[0])[:Xs.shape[0]]          # match KD batch budget
        for (a, b) in chunks(Xs.shape[0]):
            op.zero_grad()
            x = Xs[a:b].to(dev); z = Zs[a:b]
            if mode == 'logit':
                zc = z + head(x)
                loss = F.cross_entropy(zc, Ys[a:b])
            else:
                e = F.normalize(x + head(x), dim=1)
                loss = F.cross_entropy(LS * (e @ qn.T), Ys[a:b])
            if lam > 0:
                ia = perm_a[a:b]
                xa = Xa[ia].to(dev); za = Za[ia]
                if mode == 'logit':
                    zk = za + head(xa)
                    loss = loss + lam * F.kl_div(F.log_softmax(zk / TAU, 1),
                                                 F.softmax(za / TAU, 1),
                                                 reduction='batchmean') * (TAU ** 2)
                else:
                    ek = xa + head(xa)
                    loss = loss + lam * (1 - F.cosine_similarity(ek, xa, dim=1)).mean()
                del xa
            loss.backward(); op.step(); del x
    return head


def train_episodic(tl_seen, tg_seen, tf_seen, seen, beta=1.0, epochs=EPOCHS, n_pseudo=30):
    """B5: each epoch hide n_pseudo of the SEEN classes as pseudo-novel and penalise deviation
    from the frozen prediction on their logit columns. Real novel classes are never touched."""
    head = ResHead(D, NCLS).to(dev)
    op = torch.optim.Adam(head.parameters(), lr=1e-3)
    rng = np.random.RandomState(1234)
    seen_arr = np.array(sorted(seen))
    for ep in range(epochs):
        pn = torch.tensor(sorted(rng.choice(seen_arr, n_pseudo, replace=False).tolist()), device=dev)
        keep = ~torch.isin(tg_seen, pn)                       # train only on pseudo-seen patches
        Xe, Ze, Ye = tf_seen[keep.cpu()], LS * tl_seen[keep], tg_seen[keep]
        for (a, b) in chunks(Xe.shape[0]):
            op.zero_grad()
            x = Xe[a:b].to(dev); z = Ze[a:b]
            zc = z + head(x)
            loss = F.cross_entropy(zc, Ye[a:b])
            # preservation penalty on the held-out (pseudo-novel) columns
            loss = loss + beta * F.kl_div(F.log_softmax(zc[:, pn] / TAU, 1),
                                          F.softmax(z[:, pn] / TAU, 1),
                                          reduction='batchmean') * (TAU ** 2)
            loss.backward(); op.step(); del x
    return head


def train_affinity(seen_set, epochs=3):
    """B6: learn the graph edge weights (class-agnostic) on seen-class patches only."""
    wmlp = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 1)).to(dev)
    op = torch.optim.Adam(wmlp.parameters(), lr=1e-3)
    sub = TR[:AFFINITY_IMGS]
    for _ in range(epochs):
        for (f, l, g) in sub:
            m = g > 0
            if m.sum() < 16:
                continue
            gc = (g[m] - 1).to(dev).long()
            keep = torch.isin(gc, torch.tensor(sorted(seen_set), device=dev))
            if keep.sum() < 8:
                continue
            fdev = f.to(dev).float()
            fn = F.normalize(fdev, dim=1)
            sim = fn @ fn.T
            n = fn.shape[0]; kk = min(8, n - 1)
            if kk < 1:
                continue
            _, idx = sim.topk(kk + 1, dim=1); idx = idx[:, 1:]
            fi = fn.unsqueeze(1).expand(-1, kk, -1); fj = fn[idx]
            pair = torch.stack([(fi * fj).sum(-1), (fi - fj).norm(dim=-1)], dim=-1)
            w = F.softplus(wmlp(pair).squeeze(-1))
            W = torch.zeros(n, n, device=dev).scatter_(1, idx, w)
            W = (W + W.T) / 2
            W = W / W.sum(1, keepdim=True).clamp(min=1e-6)
            P = F.softmax(LS * l.to(dev).float(), 1)
            out = 0.6 * P + 0.4 * (W @ P)
            op.zero_grad()
            loss = F.nll_loss(torch.log(out[m][keep].clamp(min=1e-8)), gc[keep])
            loss.backward(); op.step()
    return wmlp


# ---------------------------------------------------------------- experiment loop
rng = np.random.RandomState(0)          # IDENTICAL split RNG to pm_stage3.py
rows = {}


def rec(name, m):
    rows.setdefault(name, []).append(m)


for s in range(SEEDS):
    perm = rng.permutation(NCLS)
    novel = sorted(perm[:NOVEL].tolist())
    seen = sorted(perm[NOVEL:].tolist())
    seen_set = set(seen)
    tseen = torch.isin(tgc, torch.tensor(seen, device=dev))
    tl_s, tg_s = tl_[tseen], tgc[tseen]
    tf_s = tf_[tseen.cpu()]
    print(f'\n===== split {s}: {len(seen)} seen / {len(novel)} novel, '
          f'{tl_s.shape[0]} seen train patches =====', flush=True)

    # ---- reference rows
    rec('uncal', metrics(F.softmax(LS * vl, 1), novel, seen))
    T = fit_gT(tl_s, tg_s)
    rec('globalT', metrics(F.softmax(LS * vl / T, 1), novel, seen))
    rec('diffusion', metrics(val_diffuse(True), novel, seen))
    rec('laplacian', metrics(val_diffuse(False), novel, seen))
    h0 = train_kd(tl_s, tg_s, tf_s, 0.0, 'logit')                     # == current audited decoder
    rec('decoder_seen_only', metrics(apply_head(h0, 1.0, 'logit'), novel, seen))

    # ---- B1 frozen-logit distillation, frontier over lambda
    for lam in LAMBDAS:
        if lam == 0.0:
            continue
        h = train_kd(tl_s, tg_s, tf_s, lam, 'logit')
        rec(f'B1_logitKD_lam{lam}', metrics(apply_head(h, 1.0, 'logit'), novel, seen))
        del h; torch.cuda.empty_cache()

    # ---- B2 embedding-space distillation (class-agnostic head)
    for lam in [0.0, 1.0, 10.0]:
        h = train_kd(tl_s, tg_s, tf_s, lam, 'emb')
        rec(f'B2_embKD_lam{lam}', metrics(apply_head(h, 1.0, 'emb'), novel, seen))
        del h; torch.cuda.empty_cache()

    # ---- B3 output-space gated mixture (uses the seen-only head h0: does gating alone rescue it?)
    with torch.no_grad():
        p_head = apply_head(h0, 1.0, 'logit')
        p_frozen = F.softmax(LS * vl, 1)
        for gt_ in GATES:
            rec(f'B3_gate{gt_}', metrics((1 - gt_) * p_frozen + gt_ * p_head, novel, seen))
        del p_head, p_frozen; torch.cuda.empty_cache()

    # ---- B4 weight-space interpolation toward the zero-residual head
    for al in ALPHAS:
        rec(f'B4_winterp{al}', metrics(apply_head(h0, al, 'logit'), novel, seen))

    # ---- B5 pseudo-novel episodic training
    h5 = train_episodic(tl_s, tg_s, tf_s, seen)
    rec('B5_episodic', metrics(apply_head(h5, 1.0, 'logit'), novel, seen))
    del h5; torch.cuda.empty_cache()

    # ---- B6 learned class-agnostic affinity
    w6 = train_affinity(seen_set)
    rec('B6_learned_affinity', metrics(val_diffuse(True, wmlp=w6), novel, seen))
    del w6, h0; torch.cuda.empty_cache()

    for k in ['uncal', 'decoder_seen_only', 'B1_logitKD_lam1.0', 'B1_logitKD_lam10.0',
              'B2_embKD_lam1.0', 'B5_episodic', 'B6_learned_affinity']:
        if k in rows:
            r = rows[k][-1]
            print(f"  {k:24s} mIoU={r['mIoU']:5.1f} seen={r['seenIoU']:5.1f} novel={r['novelIoU']:5.1f} "
                  f"hIoU={r['hIoU']:5.1f} ECE={r['ECE']:5.1f} drift={r['drift']:5.1f} "
                  f"novel_rate={r['novel_rate']:5.2f}", flush=True)


def agg(v):
    return {k: {'mean': round(float(np.nanmean([r[k] for r in v])), 2),
                'std': round(float(np.nanstd([r[k] for r in v], ddof=1)), 2)} for k in v[0]}


res = {k: agg(v) for k, v in rows.items()}
base_novel = res['uncal']['novelIoU']['mean']
for k in res:
    res[k]['collapse_index'] = round(res[k]['novelIoU']['mean'] / base_novel, 3) if base_novel else None

json.dump({'config': {'dataset': 'ADE20K-150 dense-CLIP', 'novel': NOVEL, 'seeds': SEEDS,
                      'lambdas': LAMBDAS, 'gates': GATES, 'alphas': ALPHAS,
                      'epochs': EPOCHS, 'tau': TAU},
           'summary': res}, open(f'{CACHE}/metrics_align.json', 'w'), indent=1)

print('\n=== STAGE5 SUMMARY (mean over %d splits) ===' % SEEDS)
print(f"{'method':26s} {'mIoU':>6s} {'seen':>6s} {'novel':>6s} {'hIoU':>6s} {'ECE':>6s} "
      f"{'drift':>6s} {'nov%':>6s} {'CI':>5s}")
for k, r in res.items():
    print(f"{k:26s} {r['mIoU']['mean']:6.1f} {r['seenIoU']['mean']:6.1f} {r['novelIoU']['mean']:6.1f} "
          f"{r['hIoU']['mean']:6.1f} {r['ECE']['mean']:6.1f} {r['drift']['mean']:6.1f} "
          f"{r['novel_rate']['mean']:6.2f} {r['collapse_index']}")
print('WROTE metrics_align.json', flush=True)
