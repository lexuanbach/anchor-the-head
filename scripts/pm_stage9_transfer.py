# Stage 9: COCO-Stuff -> ADE20K cross-vocabulary transfer audit.
#
# Protocol: every correction is FIT on COCO-Stuff-171 patches (stage 9a cache, same frozen
# SCLIP-CSA ViT-B/16 backbone) and EVALUATED on the full ADE20K-150 val cache. The entire ADE
# vocabulary is unseen at fitting time, so this is the transfer analogue of our seen/novel split
# with the split moved to the dataset boundary.
#
# A structural finding falls out of the design itself: class-SPECIFIC corrections (per-class
# temperature, logit-space residual heads over the 171 COCO classes) are NOT APPLICABLE to a new
# vocabulary at all -- their parameters are indexed by training classes. Only class-symmetric or
# text-conditioned corrections can even be transported:
#   globalT_coco     scalar temperature fit on COCO           (class-symmetric)
#   textcal_coco     T_c = softplus(MLP(text_c)) fit on COCO, applied to ADE class texts
#   embhead_coco     class-agnostic feature-space residual, CE vs COCO text protos
#                    (lam \in {0,1}: without / with the cosine anchor of B2)
#   diffusion        parameter-free reference (never fit on anything)
#
# Writes $PMX_OUT/metrics_transfer.json
import os, glob, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

dev = 'cuda'
ADE_CACHE = os.environ.get('PMX_CACHE', 'cache')
COCO_CACHE = os.environ.get('PMX_COCO_CACHE', 'coco_cache')
OUT = os.environ.get('PMX_OUT', ADE_CACHE)
NCLS_ADE, NCLS_COCO, LS, BINS = 150, 171, 40.0, 15
EPOCHS, BS = int(os.environ.get('PMX_EPOCHS', 12)), 8192


def load(cache, split):
    out = []
    for f in sorted(glob.glob(f'{cache}/{split}/*.npz')):
        d = np.load(f)
        out.append((torch.from_numpy(d['feat'].astype(np.float32)),
                    torch.from_numpy(d['clslogit'].astype(np.float32)),
                    torch.from_numpy(d['gt'].astype(np.int64))))
    return out


print('loading caches...', flush=True)
ADE = load(ADE_CACHE, 'val')
COCO = load(COCO_CACHE, 'train')
q_ade = F.normalize(torch.from_numpy(np.load(f'{ADE_CACHE}/qfeat.npy')).float().to(dev), dim=1)
q_coco = F.normalize(torch.from_numpy(np.load(f'{COCO_CACHE}/qfeat_coco.npy')).float().to(dev), dim=1)
print(f'ADE val imgs={len(ADE)} COCO fit imgs={len(COCO)}', flush=True)

vf = torch.cat([x[0] for x in ADE])
vl = torch.cat([x[1] for x in ADE]).to(dev)
vg = torch.cat([x[2] for x in ADE])
vgc = (vg - 1).clamp(min=0).to(dev).long()
vv = (vg > 0).to(dev)
cf = torch.cat([x[0] for x in COCO])
cl = torch.cat([x[1] for x in COCO])
cg = torch.cat([x[2] for x in COCO])
cm = cg > 0
cf_, cl_, cgc = cf[cm], cl[cm].to(dev), (cg[cm] - 1).to(dev).long()
D = cf_.shape[1]
P_base = F.softmax(LS * vl, 1)
print(f'ADE patches={vl.shape[0]} COCO fit patches={cl_.shape[0]}', flush=True)


def metrics(probs):
    p = probs[vv]; g = vgc[vv]
    conf, pred = p.max(1); correct = (pred == g).float()
    e = 0.
    for b in range(BINS):
        m = (conf > b / BINS) & (conf <= (b + 1) / BINS)
        if m.any():
            e += (m.float().mean() * (correct[m].mean() - conf[m].mean()).abs()).item()
    inter = torch.zeros(NCLS_ADE, device=dev); union = torch.zeros(NCLS_ADE, device=dev)
    for c in range(NCLS_ADE):
        inter[c] = ((g == c) & (pred == c)).sum(); union[c] = ((g == c) | (pred == c)).sum()
    iou = torch.where(union > 0, inter / union, torch.full_like(inter, float('nan')))
    bp = P_base[vv]
    return dict(mIoU=100 * iou[~torch.isnan(iou)].mean().item(), ECE=100 * e,
                acc=100 * correct.mean().item(),
                drift=100 * (1 - F.cosine_similarity(p, bp, dim=1)).mean().item())


def fit_globalT():
    T = torch.ones(1, device=dev, requires_grad=True)
    o = torch.optim.LBFGS([T], lr=0.1, max_iter=50)
    def c():
        o.zero_grad(); l = F.cross_entropy(LS * cl_ / T.clamp(min=0.05), cgc); l.backward(); return l
    o.step(c)
    return T.detach().clamp(min=0.05).item()


def fit_textcal(epochs=200):
    mlp = nn.Sequential(nn.Linear(D, 128), nn.ReLU(), nn.Linear(128, 1)).to(dev)
    op = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        op.zero_grad()
        T = F.softplus(mlp(q_coco)).squeeze(1) + 0.3
        loss = F.cross_entropy(LS * cl_ / T.view(1, -1), cgc)
        loss.backward(); op.step()
    with torch.no_grad():
        return (F.softplus(mlp(q_ade)).squeeze(1) + 0.3).detach()      # ADE temps from ADE texts


def fit_embhead(lam):
    h = nn.Sequential(nn.Linear(D, 256), nn.ReLU(), nn.Linear(256, D)).to(dev)
    op = torch.optim.Adam(h.parameters(), lr=1e-3)
    for _ in range(EPOCHS):
        for i in range(0, cf_.shape[0], BS):
            op.zero_grad()
            x = cf_[i:i + BS].to(dev)
            e = F.normalize(x + h(x), dim=1)
            loss = F.cross_entropy(LS * (e @ q_coco.T), cgc[i:i + BS])
            if lam > 0:
                loss = loss + lam * (1 - F.cosine_similarity(x + h(x), x, dim=1)).mean()
            loss.backward(); op.step(); del x
    return h


@torch.no_grad()
def apply_embhead(h):
    outs = []
    for a in range(0, vf.shape[0], 200_000):
        b = min(a + 200_000, vf.shape[0])
        x = vf[a:b].to(dev)
        e = F.normalize(x + h(x), dim=1)
        outs.append(F.softmax(LS * (e @ q_ade.T), 1))
        del x
    return torch.cat(outs)


@torch.no_grad()
def diffuse_all(alpha=0.4, k=2, knn=8):
    outs = []
    for (feat, logit, gt) in ADE:
        f = F.normalize(feat.to(dev).float(), dim=1)
        P = F.softmax(LS * logit.to(dev).float(), 1)
        n = f.shape[0]; kk = min(knn, n - 1)
        if kk < 1:
            outs.append(P); continue
        val, idx = f @ f.T, None
        val, idx = val.topk(kk + 1, dim=1)
        val, idx = val[:, 1:], idx[:, 1:]
        W = torch.zeros(n, n, device=dev).scatter_(1, idx, val.clamp(min=0))
        W = (W + W.T) / 2
        W = W / W.sum(1, keepdim=True).clamp(min=1e-6)
        out = P
        for _ in range(k):
            out = (1 - alpha) * P + alpha * (W @ out)
        outs.append(out)
    return torch.cat(outs)


res = {}
res['baseline'] = metrics(P_base)
T = fit_globalT()
res['globalT_coco'] = metrics(F.softmax(LS * vl / T, 1))
res['globalT_coco']['fit_T'] = round(T, 3)
Tc = fit_textcal()
res['textcal_coco'] = metrics(F.softmax(LS * vl / Tc.view(1, -1), 1))
for lam in [0.0, 1.0]:
    h = fit_embhead(lam)
    res[f'embhead_coco_lam{lam}'] = metrics(apply_embhead(h))
    del h; torch.cuda.empty_cache()
res['diffusion'] = metrics(diffuse_all())
res['perclassT_coco'] = {'note': 'STRUCTURALLY INAPPLICABLE: 171 per-class temperatures are indexed '
                                 'by COCO-Stuff classes and cannot be transported to the ADE vocabulary'}
res['logit_head_coco'] = {'note': 'STRUCTURALLY INAPPLICABLE: residual head outputs 171 COCO logits'}

json.dump({'config': dict(fit='COCO-Stuff-171 (val2017 patches, stage9a)', eval='ADE20K-150 val',
                          n_ade=len(ADE), n_coco=len(COCO), LS=LS, epochs=EPOCHS),
           'summary': res}, open(os.path.join(OUT, 'metrics_transfer.json'), 'w'), indent=1)
print('\n=== STAGE9 TRANSFER (fit COCO-Stuff -> eval ADE20K) ===')
for k, r in res.items():
    if 'note' in r:
        print(f'{k:22s} {r["note"][:60]}...')
    else:
        print(f"{k:22s} mIoU={r['mIoU']:5.1f} acc={r['acc']:5.1f} ECE={r['ECE']:5.1f} drift={r['drift']:5.1f}")
print('WROTE metrics_transfer.json', flush=True)
