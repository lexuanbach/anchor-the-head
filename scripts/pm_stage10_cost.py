# Stage 10: runtime, memory, and graph-size numbers for each correction.
# Dense-CLIP regime on the cached patches; per-component wall-clock + peak GPU memory.
# Writes $PMX_OUT/metrics_cost.json
import os, glob, json, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

dev = 'cuda'
CACHE = os.environ.get('PMX_CACHE', 'cache_dense')
OUT = os.environ.get('PMX_OUT', CACHE)
NCLS, LS = 150, 40.0


def load(split):
    out = []
    for f in sorted(glob.glob(f'{CACHE}/{split}/*.npz')):
        d = np.load(f)
        out.append((torch.from_numpy(d['feat'].astype(np.float32)),
                    torch.from_numpy(d['clslogit'].astype(np.float32)),
                    torch.from_numpy(d['gt'].astype(np.int64))))
    return out


VAL, TR = load('val'), load('train')
tl = torch.cat([x[1] for x in TR]); tg = torch.cat([x[2] for x in TR])
tm = tg > 0
tf_ = torch.cat([x[0] for x in TR])[tm]
tl_, tgc = tl[tm].to(dev), (tg[tm] - 1).to(dev).long()
res = {'graph': {'nodes_per_img_mean': round(float(np.mean([x[0].shape[0] for x in VAL])), 1),
                 'knn': 8, 'edges_per_img_mean': round(float(np.mean([x[0].shape[0] * 8 for x in VAL])), 1),
                 'n_val_imgs': len(VAL), 'n_train_patches': int(tl_.shape[0])}}


def timed(name, fn, *a, **kw):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    out = fn(*a, **kw)
    torch.cuda.synchronize()
    res[name] = {'wall_s': round(time.time() - t0, 2),
                 'peak_gpu_MB': round(torch.cuda.max_memory_allocated() / 1e6, 1)}
    return out


@torch.no_grad()
def diffusion_gpu():
    for (feat, logit, gt) in VAL:
        f = F.normalize(feat.to(dev), dim=1)
        P = F.softmax(LS * logit.to(dev), 1)
        n = f.shape[0]; kk = min(8, n - 1)
        val, idx = (f @ f.T).topk(kk + 1, dim=1)
        val, idx = val[:, 1:], idx[:, 1:]
        W = torch.zeros(n, n, device=dev).scatter_(1, idx, val.clamp(min=0))
        W = (W + W.T) / 2; W = W / W.sum(1, keepdim=True).clamp(min=1e-6)
        out = P
        for _ in range(2):
            out = 0.6 * P + 0.4 * (W @ out)


@torch.no_grad()
def diffusion_cpu(n_imgs=100):
    for (feat, logit, gt) in VAL[:n_imgs]:
        f = F.normalize(feat, dim=1)
        P = F.softmax(LS * logit, 1)
        n = f.shape[0]; kk = min(8, n - 1)
        val, idx = (f @ f.T).topk(kk + 1, dim=1)
        val, idx = val[:, 1:], idx[:, 1:]
        W = torch.zeros(n, n).scatter_(1, idx, val.clamp(min=0))
        W = (W + W.T) / 2; W = W / W.sum(1, keepdim=True).clamp(min=1e-6)
        out = P
        for _ in range(2):
            out = 0.6 * P + 0.4 * (W @ out)


def globalT():
    T = torch.ones(1, device=dev, requires_grad=True)
    o = torch.optim.LBFGS([T], lr=0.1, max_iter=50)
    def c():
        o.zero_grad(); l = F.cross_entropy(LS * tl_ / T.clamp(min=0.05), tgc); l.backward(); return l
    o.step(c)


def decoder():
    h = nn.Sequential(nn.Linear(tf_.shape[1], 256), nn.ReLU(), nn.Linear(256, NCLS)).to(dev)
    op = torch.optim.Adam(h.parameters(), lr=1e-3)
    for _ in range(12):
        for i in range(0, tf_.shape[0], 8192):
            op.zero_grad()
            loss = F.cross_entropy(tl_[i:i + 8192] * LS + h(tf_[i:i + 8192].to(dev)), tgc[i:i + 8192])
            loss.backward(); op.step()


timed('diffusion_gpu_full_val', diffusion_gpu)
res['diffusion_gpu_full_val']['s_per_img'] = round(res['diffusion_gpu_full_val']['wall_s'] / len(VAL), 4)
timed('diffusion_cpu_100imgs', diffusion_cpu)
res['diffusion_cpu_100imgs']['s_per_img'] = round(res['diffusion_cpu_100imgs']['wall_s'] / 100, 4)
timed('globalT_fit', globalT)
timed('decoder_train_12ep_1split', decoder)
res['decoder_params'] = sum(p.numel() for p in
                            nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, NCLS)).parameters())
json.dump(res, open(os.path.join(OUT, 'metrics_cost.json'), 'w'), indent=1)
print(json.dumps(res, indent=1))
print('WROTE metrics_cost.json', flush=True)
