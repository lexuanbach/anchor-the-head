"""Qualitative collapse figure on PASCAL VOC2012 (training-free dense CLIP, ~82 mIoU, object-centric).

For a few VOC val images that contain a held-out NOVEL object class, paint per-pixel predictions for
(1) parameter-free graph diffusion and (2) a per-class temperature fit on seen classes, beside the
ground truth. Predictions are shown on the labelled foreground (background/void -> light grey) so the
object silhouettes are clean. Diffusion keeps the novel object's class; the seen-only per-class
temperature collapses it (the object is repainted as a seen class). Reuses voc_collapse machinery.

Outputs PNGs + manifest to $PMX_CACHE/qual_voc/.  GPU + SCLIP + VOC required.
"""
import os, sys, json, numpy as np, torch, torch.nn.functional as F
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
import voc_collapse as V

dev = V.dev; VOC = V.VOC; NCLS = V.NCLS; NOVEL = V.NOVEL; LS = V.LS
QDIR = os.path.join(V.CACHE, 'qual_voc'); os.makedirs(QDIR, exist_ok=True)
GREY = np.array([235, 235, 235], np.uint8)


def voc_cmap():
    def bitget(v, i): return (v >> i) & 1
    cmap = np.zeros((256, 3), np.uint8)
    for k in range(256):
        r = g = b = 0; c = k
        for j in range(8):
            r |= bitget(c, 0) << (7 - j); g |= bitget(c, 1) << (7 - j); b |= bitget(c, 2) << (7 - j); c >>= 3
        cmap[k] = [r, g, b]
    return cmap
CMAP = voc_cmap()


def paint(pred_full, gt):                 # pred_full: HxW class 0..19; show on GT foreground only
    out = np.tile(GREY, (*gt.shape, 1))
    fg = (gt >= 1) & (gt <= NCLS)
    out[fg] = CMAP[pred_full[fg] + 1]     # object class c -> VOC colour c+1
    return Image.fromarray(out)


def gt_paint(gt):
    out = np.tile(GREY, (*gt.shape, 1)); fg = (gt >= 1) & (gt <= NCLS)
    out[fg] = CMAP[gt[fg]]                # gt already 1..20
    return Image.fromarray(out)


@torch.no_grad()
def predict(net, ps, qfeat, qidx, iid):
    gt = np.array(Image.open(f'{VOC}/SegmentationClass/{iid}.png')).astype(np.int64)
    img = V.load_img(f'{VOC}/JPEGImages/{iid}.jpg').to(dev).type(net.dtype)
    logit, feat, gh, gw = V.slide_feat(net, img, ps, qfeat)
    cl = torch.full((NCLS, gh, gw), -1e4, device=dev)
    for c in range(NCLS):
        m = (qidx == c)
        if m.any(): cl[c] = logit[m].max(0).values
    return gt, cl, feat, gh, gw


def uparg(grid_logits, size):             # grid_logits: [NCLS,gh,gw] -> argmax HxW
    up = F.interpolate(grid_logits.unsqueeze(0).float(), size=size, mode='bilinear', align_corners=False)[0]
    return up.argmax(0).cpu().numpy()


def main():
    import clip
    net, _ = clip.load('ViT-B/16', device=dev, jit=False); net.eval(); ps = net.visual.patch_size
    qfeat, qidx, _ = V.build_text(net)
    rng = np.random.RandomState(0); perm = rng.permutation(NCLS)
    novel = sorted(perm[:NOVEL].tolist()); seen = sorted(perm[NOVEL:].tolist())
    print('novel object classes:', novel, flush=True)

    val = [l.strip() for l in open(f'{VOC}/ImageSets/Segmentation/val.txt')]
    cand = []
    for iid in val:
        gt = np.array(Image.open(f'{VOC}/SegmentationClass/{iid}.png')).astype(np.int64)
        nov = sum(int((gt == (c + 1)).sum()) for c in novel)
        if nov > 3000: cand.append((nov, iid))
    cand.sort(key=lambda z: -z[0]); cand = cand[:12]
    print(f'{len(cand)} val candidates with a novel object', flush=True)

    # per-class temperature on seen pixels from 200 train images
    tr = [l.strip() for l in open(f'{VOC}/ImageSets/Segmentation/train.txt')][:60]
    TL, TG = [], []
    for iid in tr:
        gt, cl, feat, gh, gw = predict(net, ps, qfeat, qidx, iid)
        up = F.interpolate(cl.unsqueeze(0).float(), size=gt.shape, mode='bilinear', align_corners=False)[0]
        valid = (gt >= 1) & (gt <= NCLS)
        lab = gt[valid] - 1
        keepseen = np.isin(lab, seen)
        TL.append(up.permute(1, 2, 0)[valid].cpu()[keepseen]); TG.append(torch.from_numpy(lab[keepseen]))
    tl = torch.cat(TL).to(dev); tg = torch.cat(TG).to(dev).long()
    Tc = V.fit_pcT(tl, tg, seen)
    print('fit per-class T on', tl.shape[0], 'seen pixels', flush=True)

    scored = []
    for nov, iid in cand:
        gt, cl, feat, gh, gw = predict(net, ps, qfeat, qidx, iid)
        n = gh * gw
        fl = cl.reshape(NCLS, n).T; ff = feat.reshape(feat.shape[0], n).T
        Pd = V.diffuse(ff, fl)                                   # [n,NCLS]
        diff_grid = Pd.argmax(1).reshape(gh, gw)
        pcT_grid = F.softmax(LS * cl / Tc.view(NCLS, 1, 1), 0).argmax(0)
        diff_full = F.interpolate(diff_grid[None, None].float(), size=gt.shape, mode='nearest')[0, 0].cpu().numpy().astype(int)
        pcT_full = F.interpolate(pcT_grid[None, None].float(), size=gt.shape, mode='nearest')[0, 0].cpu().numpy().astype(int)
        novmask = np.isin(gt - 1, novel) & (gt >= 1)
        gain = int(((diff_full == (gt - 1)) & novmask).sum()) - int(((pcT_full == (gt - 1)) & novmask).sum())
        scored.append((gain, iid, gt, diff_full, pcT_full))
    scored.sort(key=lambda z: -z[0])

    manifest = []
    for rank, (gain, iid, gt, diff_full, pcT_full) in enumerate(scored[:4]):
        Image.open(f'{VOC}/JPEGImages/{iid}.jpg').convert('RGB').save(os.path.join(QDIR, f'ex{rank}_input.jpg'), quality=92)
        gt_paint(gt).save(os.path.join(QDIR, f'ex{rank}_gt.png'))
        paint(diff_full, gt).save(os.path.join(QDIR, f'ex{rank}_diffusion.png'))
        paint(pcT_full, gt).save(os.path.join(QDIR, f'ex{rank}_perclassT.png'))
        manifest.append({'rank': rank, 'voc_id': iid, 'novel_correct_gain': gain})
        print(f'  ex{rank}: {iid} gain={gain}', flush=True)
    json.dump({'novel_classes': novel, 'examples': manifest}, open(os.path.join(QDIR, 'qual_voc.json'), 'w'), indent=1)
    print(f'[qual_voc] wrote {len(manifest)} examples to {QDIR}', flush=True)


if __name__ == '__main__':
    main()
