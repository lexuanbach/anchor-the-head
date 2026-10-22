# Stage 9a: cache dense-CLIP patches for COCO-Stuff-171 (val2017 images +
# stuffthingmaps annotations) with the SAME frozen SCLIP-CSA ViT-B/16 backbone as ADE20K.
# These patches are used ONLY to FIT corrections (the transfer audit evaluates on ADE20K), so the
# val2017 split is fine as a fitting set and much cheaper than train2017.
#
# stuffthingmaps convention (cocostuff repo): png values are original label ids MINUS ONE
# (0..181), with 255 = unlabeled. Original ids: 1..91 things (11 ids unused), 92..182 stuff.
# COCO-Stuff-171 drops the 11 unused thing ids; classes are the remaining ids in sorted order,
# matching the 171-line cls_coco_stuff.txt used by SCLIP/mmseg.
import os, sys, glob, numpy as np, torch, torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pm_stage1 as s1   # reuse build_text / load_img / slide_feat and the SCLIP clip import

dev = 'cuda'
OUT = os.environ.get('PMX_CACHE')            # e.g. coco_cache
COCO_IMG = os.environ['PMX_COCO_IMG']        # .../coco/val2017
COCO_ANN = os.environ['PMX_COCO_ANN']        # .../coco/stuffthingmaps/val2017
N_IMGS = int(os.environ.get('PMX_COCO_N', 600))

UNUSED_THINGS = {12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91}
VALID = [i for i in range(1, 183) if i not in UNUSED_THINGS]        # 171 original ids
assert len(VALID) == 171
LUT = np.zeros(256, np.int16)                                        # png value -> class+1 (0=ignore)
for ci, orig in enumerate(VALID):
    LUT[orig - 1] = ci + 1                                           # png stores orig-1


def main():
    names = [l.strip() for l in open(os.environ['PMX_ADE_NAMES'])]   # points at cls_coco_stuff.txt
    assert len(names) == 171, f'expected 171 COCO-Stuff classes, got {len(names)}'
    net, _ = s1.clip.load('ViT-B/16', device=dev, jit=False)
    net.eval()
    ps = net.visual.patch_size
    qfeat, qidx, ncls = s1.build_text(net)
    assert ncls == 171
    # per-class prototypes (synonym mean, renormalised) for the transfer heads
    qc = torch.stack([F.normalize(qfeat[qidx == c].mean(0), dim=0) for c in range(ncls)])
    np.save(os.path.join(OUT, 'qfeat_coco.npy'), qc.half().cpu().numpy())
    od = os.path.join(OUT, 'train')
    os.makedirs(od, exist_ok=True)
    imgs = sorted(glob.glob(os.path.join(COCO_IMG, '*.jpg')))[:N_IMGS]
    print(f'caching {len(imgs)} COCO val2017 images (fitting set only)', flush=True)
    for k, ip in enumerate(imgs):
        fp = os.path.join(od, f'{k:05d}.npz')
        if os.path.exists(fp):
            continue
        ann = os.path.join(COCO_ANN, os.path.basename(ip).replace('.jpg', '.png'))
        gt_raw = np.array(Image.open(ann))
        gt = LUT[np.minimum(gt_raw, 255)]                            # 0=ignore, 1..171
        img = s1.load_img(ip).to(dev).type(net.dtype)
        logit, feat, gh, gw = s1.slide_feat(net, img, ps, qfeat)
        cl = torch.full((ncls, gh, gw), -1e4, device=dev)
        for c in range(ncls):
            m = (qidx == c)
            if m.any():
                cl[c] = logit[m].max(0).values
        gtt = torch.from_numpy(gt).float().view(1, 1, *gt.shape)
        gtp = F.interpolate(gtt, size=(gh, gw), mode='nearest')[0, 0].numpy().astype(np.int16)
        np.savez(fp, feat=feat.permute(1, 2, 0).reshape(-1, feat.shape[0]).half().cpu().numpy(),
                 clslogit=cl.permute(1, 2, 0).reshape(-1, ncls).half().cpu().numpy(),
                 gt=gtp.reshape(-1), grid=np.array([gh, gw]))
        if (k + 1) % 100 == 0:
            print(f'  [{k+1}/{len(imgs)}]', flush=True)
    print('DONE_COCO_CACHE', flush=True)


if __name__ == '__main__':
    main()
