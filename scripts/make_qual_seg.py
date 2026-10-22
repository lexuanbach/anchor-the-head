"""Qualitative collapse figure for ProtoMask (dense-CLIP ADE20K regime, from the cached patches).

For a few ADE20K val images that contain a held-out NOVEL class, we paint per-pixel predictions for
(1) parameter-free graph diffusion and (2) a per-class temperature fit on seen classes, alongside the
ground truth. Diffusion preserves the novel-class region; the seen-only per-class temperature collapses
it (the novel object is repainted as a seen class), visualising the collapse law. Reuses the Stage-1
dense-CLIP cache (feat/clslogit/gt/grid). Loading is selective so only the chosen images decompress feat.

Outputs colorised PNGs and a manifest to $PMX_CACHE/qual_seg/.
"""
import os, glob, json, numpy as np, torch, torch.nn.functional as F
from PIL import Image
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
CACHE = os.environ.get('PMX_CACHE', 'cache')
ADE = os.environ.get('PMX_ADE_ROOT', 'data/ADEChallengeData2016')
QDIR = os.path.join(CACHE, 'qual_seg'); os.makedirs(QDIR, exist_ok=True)
NCLS = 150; LS = 40.0; NOVEL = 30
qfeat = torch.from_numpy(np.load(f'{CACHE}/qfeat.npy')).float().to(dev)

perm = np.random.RandomState(0).permutation(NCLS)        # seed-0 dense-CLIP split (matches pm_stage2)
novel = sorted(perm[:NOVEL].tolist()); seen = sorted(perm[NOVEL:].tolist())
novelt = torch.tensor(novel)
_rng = np.random.RandomState(42); PAL = _rng.randint(40, 230, size=(NCLS, 3), dtype=np.uint8)


@torch.no_grad()
def diffuse(feat, logit, alpha=0.4, k=2, knn=8):
    f = F.normalize(feat.to(dev), dim=1); P = F.softmax(LS * logit.to(dev), 1)
    sim = f @ f.T; n = f.shape[0]; kk = min(knn, n - 1)
    val, idx = sim.topk(kk + 1, dim=1); val = val[:, 1:]; idx = idx[:, 1:]
    W = torch.zeros(n, n, device=dev); W.scatter_(1, idx, val.clamp(min=0)); W = (W + W.T) / 2
    W = W / W.sum(1, keepdim=True).clamp(min=1e-6)
    out = P
    for _ in range(k): out = (1 - alpha) * P + alpha * (W @ out)
    return out


def fit_perclass_T(logit, gc):
    sm = torch.zeros(NCLS, device=dev, dtype=torch.bool); sm[torch.tensor(seen, device=dev)] = True
    T = torch.ones(NCLS, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=80)
    def cl():
        opt.zero_grad(); Tc = torch.where(sm, T.clamp(min=0.05), torch.ones_like(T))
        loss = F.cross_entropy(LS * logit / Tc.view(1, -1), gc); loss.backward(); return loss
    opt.step(cl)
    Tf = T.detach().clamp(min=0.05); return torch.where(sm, Tf, torch.ones_like(Tf))


def colorize(pred_grid, gh, gw, size):
    flat = np.asarray(pred_grid).reshape(-1)
    rgb = np.zeros((gh * gw, 3), np.uint8); val = flat >= 0; rgb[val] = PAL[flat[val]]
    return Image.fromarray(rgb.reshape(gh, gw, 3)).resize(size, Image.NEAREST)


def main():
    val_files = sorted(glob.glob(f'{CACHE}/val/*.npz'))
    tr_files = sorted(glob.glob(f'{CACHE}/train/*.npz'))
    print(f'val={len(val_files)} train={len(tr_files)}', flush=True)

    # per-class temperature: read only clslogit+gt from train (skip feat)
    TL, TG = [], []
    for f in tr_files:
        d = np.load(f); TL.append(d['clslogit'].astype(np.float32)); TG.append(d['gt'].astype(np.int64))
    tl = torch.from_numpy(np.concatenate(TL)); tg = torch.from_numpy(np.concatenate(TG)); tmask = tg > 0
    Tc = fit_perclass_T(tl[tmask].to(dev), (tg[tmask] - 1).to(dev))
    print('fit per-class T', flush=True)

    # pass 1: rank val by novel-pixel count, reading only gt (npz members are lazy: feat not decompressed)
    cand = []
    for k, f in enumerate(val_files):
        gt = np.load(f)['gt'].astype(np.int64)
        nov = int(np.isin(gt - 1, novel).sum())
        if nov >= 40: cand.append((nov, k, f))
    cand.sort(key=lambda z: -z[0]); cand = cand[:25]
    print(f'  {len(cand)} candidates with a novel class', flush=True)

    # pass 2: diffusion + per-class-T only on candidates (now decompress feat for these)
    scored = []
    for _, k, f in cand:
        d = np.load(f); gh, gw = (int(x) for x in d['grid'])
        feat = torch.from_numpy(d['feat'].astype(np.float32)); logit = torch.from_numpy(d['clslogit'].astype(np.float32))
        gtc = torch.from_numpy(d['gt'].astype(np.int64)) - 1
        pd = diffuse(feat, logit).argmax(1).cpu(); pp = F.softmax(LS * logit.to(dev) / Tc.view(1, -1), 1).argmax(1).cpu()
        nov_mask = torch.isin(gtc, novelt)
        gain = int(((pd == gtc) & nov_mask).sum()) - int(((pp == gtc) & nov_mask).sum())
        scored.append((gain, k, f, gh, gw, pd.numpy(), pp.numpy(), gtc.numpy()))
    scored.sort(key=lambda z: -z[0])

    ade_imgs = sorted(glob.glob(f'{ADE}/images/validation/*.jpg'))
    manifest = []
    for rank, (gain, k, f, gh, gw, pd, pp, gtc) in enumerate(scored[:4]):
        idx = int(os.path.basename(f).split('.')[0])
        Image.open(ade_imgs[idx]).convert('RGB').resize((320, 320)).save(os.path.join(QDIR, f'ex{rank}_input.jpg'), quality=92)
        colorize(gtc, gh, gw, (320, 320)).save(os.path.join(QDIR, f'ex{rank}_gt.png'))
        colorize(pd, gh, gw, (320, 320)).save(os.path.join(QDIR, f'ex{rank}_diffusion.png'))
        colorize(pp, gh, gw, (320, 320)).save(os.path.join(QDIR, f'ex{rank}_perclassT.png'))
        manifest.append({'rank': rank, 'ade_index': idx, 'novel_correct_gain_diffusion_minus_perclassT': gain})
        print(f'  ex{rank}: ade#{idx} gain={gain}', flush=True)
    json.dump({'novel_classes': novel, 'examples': manifest}, open(os.path.join(QDIR, 'qual_seg.json'), 'w'), indent=1)
    print(f'[qual_seg] wrote {len(manifest)} examples to {QDIR}', flush=True)


if __name__ == '__main__':
    main()
