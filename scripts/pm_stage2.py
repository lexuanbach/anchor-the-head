# Stage 2: replicate ProtoMask's calibration + decoder-collapse study on the COMPETITIVE
# training-free dense-CLIP regime (cached patches from Stage 1). 5 seen/novel splits.
import os, glob, numpy as np, torch, torch.nn.functional as F
dev='cuda'; CACHE=os.environ.get('PMX_CACHE','cache_dense'); NCLS=150; LS=40.0; NOVEL=30; SEEDS=5; BINS=15
qfeat=torch.from_numpy(np.load(f'{CACHE}/qfeat.npy')).float().to(dev)   # [150,512]

def load(split, cap=None):
    fs=sorted(glob.glob(f'{CACHE}/{split}/*.npz')); fs=fs[:cap] if cap else fs
    imgs=[]
    for f in fs:
        d=np.load(f)
        imgs.append((torch.from_numpy(d['feat'].astype(np.float32)),
                     torch.from_numpy(d['clslogit'].astype(np.float32)),
                     torch.from_numpy(d['gt'].astype(np.int64))))
    return imgs
print('loading cache...',flush=True)
VAL=load('val'); TR=load('train')
print(f'val imgs={len(VAL)} train imgs={len(TR)}',flush=True)

def cat(imgs):
    return (torch.cat([x[0] for x in imgs]), torch.cat([x[1] for x in imgs]), torch.cat([x[2] for x in imgs]))
vf,vl,vg = cat(VAL)                    # val feat,logit,gt(0=ignore,1..150)
tf,tl,tg = cat(TR)
vmask=vg>0; tmask=tg>0
vl=vl.to(dev); vgc=(vg-1).to(dev); vvalid=vmask.to(dev)
tl_=tl[tmask].to(dev); tgc=(tg[tmask]-1).to(dev)

def ece_iou(probs, gtc, valid, novel_set):
    # probs [N,150] on dev; gtc [N]; valid [N] bool
    p=probs[valid]; g=gtc[valid]
    conf,pred=p.max(1)
    correct=(pred==g).float()
    def ece(sel):
        c=conf[sel]; a=correct[sel]
        if c.numel()==0: return float('nan')
        e=0.0; 
        for b in range(BINS):
            lo,hi=b/BINS,(b+1)/BINS; m=(c>lo)&(c<=hi)
            if m.any(): e+=(m.float().mean()*(a[m].mean()-c[m].mean()).abs()).item()
        return 100*e
    isnov=torch.zeros_like(g,dtype=torch.bool)
    for c in novel_set: isnov|=(g==c)
    eo=ece(torch.ones_like(isnov)); en=ece(isnov); es=ece(~isnov)
    # IoU
    inter=torch.zeros(NCLS,device=dev); union=torch.zeros(NCLS,device=dev)
    for c in range(NCLS):
        gi=(g==c); pi=(pred==c); inter[c]=(gi&pi).sum(); union[c]=(gi|pi).sum()
    iou=torch.where(union>0,inter/union,torch.full_like(inter,float('nan')))
    miou=100*iou[~torch.isnan(iou)].mean().item()
    nv=[c for c in novel_set if not torch.isnan(iou[c])]
    niou=100*iou[torch.tensor(nv,device=dev)].mean().item() if nv else float('nan')
    return eo,en,es,miou,niou

def fit_global_T(logit,gc):
    T=torch.ones(1,device=dev,requires_grad=True); opt=torch.optim.LBFGS([T],lr=0.1,max_iter=50)
    def cl():
        opt.zero_grad(); loss=F.cross_entropy(LS*logit/T.clamp(min=0.05),gc); loss.backward(); return loss
    opt.step(cl); return T.detach().clamp(min=0.05).item()

def fit_perclass_T(logit,gc,seen):
    # joint per-class temperature on seen classes (novel T frozen at 1); no inplace ops
    seen_mask=torch.zeros(NCLS,device=dev,dtype=torch.bool); seen_mask[torch.tensor(list(seen),device=dev)]=True
    T=torch.ones(NCLS,device=dev,requires_grad=True)
    opt=torch.optim.LBFGS([T],lr=0.1,max_iter=80)
    def cl():
        opt.zero_grad()
        Tc=torch.where(seen_mask,T.clamp(min=0.05),torch.ones_like(T))
        loss=F.cross_entropy(LS*logit/Tc.view(1,-1),gc); loss.backward(); return loss
    opt.step(cl)
    Tf=T.detach().clamp(min=0.05)
    return torch.where(seen_mask,Tf,torch.ones_like(Tf))

def train_decoder(tf_seen, tl_seen, tg_seen, epochs=12):
    D=tf_seen.shape[1]; head=torch.nn.Sequential(torch.nn.Linear(D,256),torch.nn.ReLU(),torch.nn.Linear(256,NCLS)).to(dev)
    opt=torch.optim.Adam(head.parameters(),lr=1e-3); X=tf_seen.to(dev); base=LS*tl_seen.to(dev); y=tg_seen.to(dev)
    for e in range(epochs):
        for i in range(0,len(X),8192):
            opt.zero_grad(); sl=base[i:i+8192]+0.5*head(X[i:i+8192])   # bounded residual on CLIP logits
            loss=F.cross_entropy(sl,y[i:i+8192]); loss.backward(); opt.step()
    return head

@torch.no_grad()
def diffuse_image(feat,logit,alpha=0.4,k=2,knn=8):
    f=F.normalize(feat.to(dev),dim=1); P=F.softmax(LS*logit.to(dev),1)
    sim=f@f.T; n=f.shape[0]; kk=min(knn,n-1)
    val,idx=sim.topk(kk+1,dim=1); val=val[:,1:]; idx=idx[:,1:]
    W=torch.zeros(n,n,device=dev); W.scatter_(1,idx,val.clamp(min=0)); W=(W+W.T)/2
    W=W/W.sum(1,keepdim=True).clamp(min=1e-6)
    out=P
    for _ in range(k): out=(1-alpha)*P+alpha*(W@out)
    return out

rng=np.random.RandomState(0)
rows={m:[] for m in ['uncal','globalT','perclassT','diffusion','decoder']}
for s in range(SEEDS):
    perm=rng.permutation(NCLS); novel=sorted(perm[:NOVEL].tolist()); seen=sorted(perm[NOVEL:].tolist())
    seenset=set(seen)
    tseen=torch.tensor([g.item() in seenset for g in tgc],device=dev)
    tl_s=tl_[tseen]; tg_s=tgc[tseen]
    # uncal
    Pu=F.softmax(LS*vl,1); rows['uncal'].append(ece_iou(Pu,vgc,vvalid,novel))
    # global T
    T=fit_global_T(tl_s,tg_s); Pg=F.softmax(LS*vl/T,1); rows['globalT'].append(ece_iou(Pg,vgc,vvalid,novel))
    # per-class T (novel T=1)
    Tc=fit_perclass_T(tl_s,tg_s,seen); Pp=F.softmax(LS*vl/Tc.view(1,-1),1); rows['perclassT'].append(ece_iou(Pp,vgc,vvalid,novel))
    # diffusion (per image, parameter-free)
    eo=en=es=mi=ni=0; accs=[]
    dprobs=[]; 
    for (feat,logit,gt) in VAL: dprobs.append(diffuse_image(feat,logit))
    Pd=torch.cat(dprobs); rows['diffusion'].append(ece_iou(Pd,vgc,vvalid,novel))
    # trained decoder
    head=train_decoder(tf[tmask][tseen.cpu()], tl_[tseen], tg_s)
    with torch.no_grad(): Pde=F.softmax(LS*vl+0.5*head(vf.to(dev)),1)
    rows['decoder'].append(ece_iou(Pde,vgc,vvalid,novel))
    print(f'[seed {s}] novelIoU uncal={rows["uncal"][-1][4]:.1f} gT={rows["globalT"][-1][4]:.1f} pcT={rows["perclassT"][-1][4]:.1f} diff={rows["diffusion"][-1][4]:.1f} dec={rows["decoder"][-1][4]:.1f}',flush=True)

import json
def agg(v):
    a=np.array(v); m=np.nanmean(a,0); sd=np.nanstd(a,0,ddof=1)
    keys=['ECE','ECE_novel','ECE_seen','mIoU','novelIoU']
    return {k:{'mean':round(float(m[i]),2),'std':round(float(sd[i]),2)} for i,k in enumerate(keys)}
res={m:agg(v) for m,v in rows.items()}
json.dump({'config':{'regime':'training-free dense CLIP ViT-B/16 (SCLIP-CSA)','novel':NOVEL,'seeds':SEEDS},'summary':res},
          open(f'{CACHE}/metrics_regime.json','w'),indent=1)
print('\n=== SUMMARY (5-split mean) ===')
for m,r in res.items():
    print(f"{m:11s} mIoU={r['mIoU']['mean']:.1f} ECE={r['ECE']['mean']:.1f} ECE_novel={r['ECE_novel']['mean']:.1f} ECE_seen={r['ECE_seen']['mean']:.1f} novelIoU={r['novelIoU']['mean']:.1f}")
print('WROTE metrics_regime.json')
