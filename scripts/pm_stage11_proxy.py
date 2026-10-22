# Stage 11: a MATERIALLY STRONGER frozen base for the head audit.
#
# Runs the identical collapse/anchoring audit on a more competitive ADE20K base. FC-CLIP/CAT-Seg-scale checkpoints exceed our MIG slice, so we
# build the strongest base our stack supports: ProxyCLIP-style DINO-proxy attention (Lan et al.,
# ECCV 2024) applied to the cached dense-CLIP logits -- CLIP's noisy patch affinities are replaced
# by DINO ViT-B/16 affinities, which is the mechanism behind ProxyCLIP's ~20 mIoU ADE numbers.
# Ours is an independent implementation on the SCLIP backbone and is labeled as such.
#
# Variant selection (full replacement vs anchored mix, temperature) is done on the TRAIN cache
# only; the chosen variant is then frozen and applied to build cache_proxy/{val,train}, after which
# the UNCHANGED pm_stage2.py / pm_stage5_align.py run the identical audit at the higher operating
# point. Writes $PMX_OUT/proxy_base.json with the selection log and base mIoU.
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

dev = 'cuda'
CACHE = os.environ.get('PMX_CACHE', 'cache')
OUTC = os.environ.get('PMX_PROXY_CACHE', 'cache_proxy')
ADE = os.environ.get('PMX_ADE_ROOT', 'data/ADEChallengeData2016')
NCLS, LS = 150, 40.0
SHORT, MAXL, PATCH = 336, 2048, 16
IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Dino:
    def __init__(self):
        self.net = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16').to(dev).eval()
        self.hook = {}
        self.net._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(
            lambda m, i, o: self.hook.__setitem__("qkv", o))

    @torch.no_grad()
    def feats(self, img):
        _ = self.net(img)
        nh = self.net.blocks[-1].attn.num_heads
        nb, nt, cq = self.hook["qkv"].shape
        qkv = self.hook["qkv"].reshape(nb, nt, 3, nh, cq // nh // 3).permute(2, 0, 3, 1, 4)
        v = qkv[2].transpose(1, 2).float()[:, 1:, :, :].flatten(-2, -1).permute(0, 2, 1)
        v = v / v.norm(dim=1, keepdim=True)
        return v[0].T.contiguous()


def load_img(path, gh, gw):
    im = Image.open(path).convert('RGB')
    w, h = im.size
    s = SHORT / min(w, h)
    if max(w, h) * s > MAXL:
        s = MAXL / max(w, h)
    im = im.resize((round(w * s), round(h * s)), Image.BILINEAR)
    t = torch.from_numpy(np.array(im)).permute(2, 0, 1).float().unsqueeze(0) / 255.
    return ((t[:, :, :gh * PATCH, :gw * PATCH] - IMNET_MEAN) / IMNET_STD).to(dev)


@torch.no_grad()
def proxy_logits(z, d, variant):
    """z [N,150] raw cosine logits; d [N,Cd] L2-normalised DINO feats."""
    sim = (d @ d.T).clamp(min=0)                     # ProxyCLIP keeps non-negative affinities
    name, tau, mix = variant
    A = torch.softmax(sim / tau, dim=1)
    zp = A @ z
    return (1 - mix) * z + mix * zp


def miou_of(zs, gts, novelset=None):
    inter = torch.zeros(NCLS, device=dev)
    union = torch.zeros(NCLS, device=dev)
    for z, g in zip(zs, gts):
        pred = z.argmax(1)
        v = g > 0
        gc = (g - 1).clamp(min=0)
        for c in torch.unique(gc[v]):
            pc = (pred == c) & v
            gcc = (gc == c) & v
            inter[c] += (pc & gcc).sum()
            union[c] += (pc | gcc).sum()
    iou = torch.where(union > 0, inter / union, torch.full_like(inter, float('nan')))
    return 100 * iou[~torch.isnan(iou)].mean().item()


def main():
    os.makedirs(os.path.join(OUTC, 'val'), exist_ok=True)
    os.makedirs(os.path.join(OUTC, 'train'), exist_ok=True)
    for aux in ('qfeat.npy',):
        src = os.path.join(CACHE, aux)
        if os.path.exists(src) and not os.path.exists(os.path.join(OUTC, aux)):
            import shutil
            shutil.copy(src, os.path.join(OUTC, aux))

    dino = Dino()
    tr_imgs = sorted(glob.glob(f'{ADE}/images/training/*.jpg'))[:600]
    val_imgs = sorted(glob.glob(f'{ADE}/images/validation/*.jpg'))

    # ---- variant selection on the first 100 TRAIN images ----
    variants = [('replace_t007', 0.07, 1.0), ('replace_t02', 0.2, 1.0),
                ('mix05_t007', 0.07, 0.5), ('mix05_t02', 0.2, 0.5)]
    sel_z = {v[0]: [] for v in variants}
    sel_z['base'] = []
    sel_g = []
    tfiles = sorted(glob.glob(f'{CACHE}/train/*.npz'))[:100]
    for i, f in enumerate(tfiles):
        dd = np.load(f)
        gh, gw = [int(x) for x in dd['grid']]
        z = torch.from_numpy(dd['clslogit'].astype(np.float32)).to(dev)
        g = torch.from_numpy(dd['gt'].astype(np.int64)).to(dev)
        dfe = dino.feats(load_img(tr_imgs[i], gh, gw))
        sel_g.append(g)
        sel_z['base'].append(z)
        for v in variants:
            sel_z[v[0]].append(proxy_logits(z, dfe, v))
    scores = {k: round(miou_of(zs, sel_g), 2) for k, zs in sel_z.items()}
    print('[select] train-100 mIoU:', scores, flush=True)
    best = max(variants, key=lambda v: scores[v[0]])
    print('[select] chosen:', best, flush=True)

    # ---- build the proxy cache with the frozen choice ----
    log = {'selection': scores, 'chosen': best[0]}
    for split, files, imgs in [('val', sorted(glob.glob(f'{CACHE}/val/*.npz')), val_imgs),
                               ('train', sorted(glob.glob(f'{CACHE}/train/*.npz')), tr_imgs)]:
        for i, f in enumerate(files):
            out = os.path.join(OUTC, split, os.path.basename(f))
            if os.path.exists(out):
                continue
            dd = np.load(f)
            gh, gw = [int(x) for x in dd['grid']]
            z = torch.from_numpy(dd['clslogit'].astype(np.float32)).to(dev)
            dfe = dino.feats(load_img(imgs[i], gh, gw))
            zp = proxy_logits(z, dfe, best)
            np.savez(out, feat=dd['feat'], clslogit=zp.cpu().numpy().astype(np.float16),
                     gt=dd['gt'], grid=dd['grid'])
            if (i + 1) % 200 == 0:
                print(f'  [{split} {i+1}/{len(files)}]', flush=True)

    # base mIoU on the full val proxy cache
    zs, gs = [], []
    for f in sorted(glob.glob(f'{OUTC}/val/*.npz')):
        dd = np.load(f)
        zs.append(torch.from_numpy(dd['clslogit'].astype(np.float32)).to(dev))
        gs.append(torch.from_numpy(dd['gt'].astype(np.int64)).to(dev))
    log['val_mIoU_proxy'] = round(miou_of(zs, gs), 2)
    zs2 = [torch.from_numpy(np.load(f)['clslogit'].astype(np.float32)).to(dev)
           for f in sorted(glob.glob(f'{CACHE}/val/*.npz'))]
    log['val_mIoU_base'] = round(miou_of(zs2, gs), 2)
    outp = os.path.join(os.environ.get('PMX_OUT', OUTC), 'proxy_base.json')
    json.dump(log, open(outp, 'w'), indent=1)
    print('[stage11]', json.dumps(log['selection']), 'chosen', best[0],
          'valBase', log['val_mIoU_base'], 'valProxy', log['val_mIoU_proxy'], flush=True)
    print('WROTE', outp, flush=True)


if __name__ == '__main__':
    main()
