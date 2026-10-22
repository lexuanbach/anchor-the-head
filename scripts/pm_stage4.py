# Stage 4 (round-4 follow-up): does an OV-AWARE, class-SYMMETRIC TRAINED calibrator avoid the collapse?
# Text-side calibrator: per-class temperature predicted from the class TEXT embedding, T_c=softplus(MLP(text_c)),
# trained on SEEN patches only; novel classes get a temperature from their own text (the map generalises).
# If it collapses too, the seen-only-supervision pattern holds here as well; if not, a positive trained method exists. 5 splits, ADE dense-CLIP cache.
import os,glob,numpy as np,torch,torch.nn as nn,torch.nn.functional as F,math,json
dev='cuda';CACHE=os.environ.get('PMX_CACHE','cache_dense');NCLS=150;LS=40.0;NOVEL=30;SEEDS=5;BINS=15
qfeat=torch.from_numpy(np.load(f'{CACHE}/qfeat.npy')).float().to(dev)   # [150,512] class text embeddings
def load(sp):
    out=[]
    for f in sorted(glob.glob(f'{CACHE}/{sp}/*.npz')):
        d=np.load(f); out.append((torch.from_numpy(d['clslogit'].astype(np.float32)),torch.from_numpy(d['gt'].astype(np.int64))))
    return out
VAL=load('val');TR=load('train')
vl=torch.cat([x[0] for x in VAL]).to(dev); vg=torch.cat([x[1] for x in VAL]); vgc=(vg-1).clamp(min=0).to(dev).long(); vv=(vg>0).to(dev)
tl=torch.cat([x[0] for x in TR]); tg=torch.cat([x[1] for x in TR]); tm=tg>0; tl_=tl[tm].to(dev); tgc=(tg[tm]-1).to(dev).long()
def metr(probs,novel):
    p=probs[vv];g=vgc[vv];conf,pred=p.max(1);correct=(pred==g).float()
    def ece(sel):
        c=conf[sel];a=correct[sel]
        if c.numel()==0:return float('nan')
        e=0.
        for b in range(BINS):
            m=(c>b/BINS)&(c<=(b+1)/BINS)
            if m.any():e+=(m.float().mean()*(a[m].mean()-c[m].mean()).abs()).item()
        return 100*e
    _o=conf.argsort(descending=True);_e=(1-correct)[_o];aurc=100*(torch.cumsum(_e,0)/torch.arange(1,len(_e)+1,device=dev)).mean().item()
    isn=torch.zeros_like(g,dtype=torch.bool)
    for c in novel:isn|=(g==c)
    inter=torch.zeros(NCLS,device=dev);union=torch.zeros(NCLS,device=dev)
    for c in range(NCLS):inter[c]=((g==c)&(pred==c)).sum();union[c]=((g==c)|(pred==c)).sum()
    iou=torch.where(union>0,inter/union,torch.full_like(inter,float('nan')))
    nv=[c for c in novel if not torch.isnan(iou[c])]
    return dict(ECE=ece(torch.ones_like(isn)),ECE_novel=ece(isn),mIoU=100*iou[~torch.isnan(iou)].mean().item(),
                novelIoU=100*iou[torch.tensor(nv,device=dev)].mean().item() if nv else float('nan'),AURC=aurc)
def fit_gT(logit,gc):
    T=torch.ones(1,device=dev,requires_grad=True);o=torch.optim.LBFGS([T],lr=0.1,max_iter=50)
    def cl():o.zero_grad();l=F.cross_entropy(LS*logit/T.clamp(min=0.05),gc);l.backward();return l
    o.step(cl);return T.detach().clamp(min=0.05).item()
def fit_pcT(logit,gc,seen):
    sm=torch.zeros(NCLS,device=dev,dtype=torch.bool);sm[torch.tensor(list(seen),device=dev)]=True
    T=torch.ones(NCLS,device=dev,requires_grad=True);o=torch.optim.LBFGS([T],lr=0.1,max_iter=80)
    def cl():o.zero_grad();Tc=torch.where(sm,T.clamp(min=0.05),torch.ones_like(T));l=F.cross_entropy(LS*logit/Tc.view(1,-1),gc);l.backward();return l
    o.step(cl);Tf=T.detach().clamp(min=0.05);return torch.where(sm,Tf,torch.ones_like(Tf))
def fit_textcal(logit,gc,epochs=200):       # T_c = softplus(MLP(text_c))+0.3, trained on seen patches
    mlp=nn.Sequential(nn.Linear(512,128),nn.ReLU(),nn.Linear(128,1)).to(dev)
    op=torch.optim.Adam(mlp.parameters(),lr=1e-3,weight_decay=1e-4)
    for e in range(epochs):
        op.zero_grad(); T=F.softplus(mlp(qfeat)).squeeze(1)+0.3   # [150]
        loss=F.cross_entropy(LS*logit/T.view(1,-1),gc); loss.backward(); op.step()
    with torch.no_grad(): return (F.softplus(mlp(qfeat)).squeeze(1)+0.3).detach()
rng=np.random.RandomState(0);rows={k:[] for k in ['globalT','perclassT','textcal']}
for s in range(SEEDS):
    perm=rng.permutation(NCLS);novel=sorted(perm[:NOVEL].tolist());seen=sorted(perm[NOVEL:].tolist());ss=set(seen)
    tseen=torch.tensor([g.item() in ss for g in tgc],device=dev);tl_s=tl_[tseen];tg_s=tgc[tseen]
    T=fit_gT(tl_s,tg_s);rows['globalT'].append(metr(F.softmax(LS*vl/T,1),novel))
    Tc=fit_pcT(tl_s,tg_s,seen);rows['perclassT'].append(metr(F.softmax(LS*vl/Tc.view(1,-1),1),novel))
    Tt=fit_textcal(tl_s,tg_s);rows['textcal'].append(metr(F.softmax(LS*vl/Tt.view(1,-1),1),novel))
    print(f"[seed {s}] novelIoU globalT={rows['globalT'][-1]['novelIoU']:.1f} perclassT={rows['perclassT'][-1]['novelIoU']:.1f} textcal={rows['textcal'][-1]['novelIoU']:.1f} | textcal ECEnov={rows['textcal'][-1]['ECE_novel']:.1f}",flush=True)
def agg(v):
    return {k:{'mean':round(float(np.nanmean([r[k] for r in v])),2),'std':round(float(np.nanstd([r[k] for r in v],ddof=1)),2)} for k in v[0]}
res={k:agg(v) for k,v in rows.items()};json.dump({'summary':res},open(f'{CACHE}/metrics_stage4.json','w'),indent=1)
print('\n=== STAGE4 ===')
for k,r in res.items():print(f"{k:10s} mIoU={r['mIoU']['mean']:.1f} ECE={r['ECE']['mean']:.1f} ECEnov={r['ECE_novel']['mean']:.1f} novelIoU={r['novelIoU']['mean']:.1f}")
print('WROTE metrics_stage4.json',flush=True)
