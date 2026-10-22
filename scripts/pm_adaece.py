# Adaptive-bin and class-balanced ECE: confirm the calibration ranking is not a
# binning artifact. Recompute ECE three ways for uncal / global-T / diffusion on the cached dense-CLIP logits.
import os,glob,numpy as np,torch,torch.nn.functional as F,json
dev='cuda';CACHE=os.environ.get('PMX_CACHE','cache_dense');NCLS=150;LS=40.0;NOVEL=30;SEEDS=5;BINS=15
qfeat=torch.from_numpy(np.load(f'{CACHE}/qfeat.npy')).float().to(dev)
def load(sp):
    out=[]
    for f in sorted(glob.glob(f'{CACHE}/{sp}/*.npz')):
        d=np.load(f); out.append((torch.from_numpy(d['feat'].astype(np.float32)),torch.from_numpy(d['clslogit'].astype(np.float32)),torch.from_numpy(d['gt'].astype(np.int64))))
    return out
VAL=load('val');TR=load('train')
def cat(a): return torch.cat([x[1] for x in a]),torch.cat([x[2] for x in a])
vl,vg=cat(VAL); tl,tg=cat(TR)
vl=vl.to(dev); vgc=(vg-1).clamp(min=0).to(dev).long(); vv=(vg>0).to(dev)
tm=tg>0; tl_=tl[tm].to(dev); tgc=(tg[tm]-1).to(dev).long()
def ece_equalwidth(conf,corr,nb=BINS):
    e=0.
    for b in range(nb):
        m=(conf>b/nb)&(conf<=(b+1)/nb)
        if m.any(): e+=(m.float().mean()*(corr[m].mean()-conf[m].mean()).abs()).item()
    return 100*e
def ece_adaptive(conf,corr,nb=BINS):                  # equal-mass bins
    n=conf.numel(); order=conf.argsort(); c=conf[order]; a=corr[order]; e=0.
    for b in range(nb):
        lo=b*n//nb; hi=(b+1)*n//nb
        if hi>lo: e+=((hi-lo)/n)*(a[lo:hi].mean()-c[lo:hi].mean()).abs().item()
    return 100*e
def ece_classbalanced(conf,corr,pred,gt,nb=BINS):     # mean over classes of per-class equal-width ECE
    es=[]
    for cl in torch.unique(gt):
        m=(gt==cl)
        if m.sum()>=20: es.append(ece_equalwidth(conf[m],corr[m],nb))
    return float(np.mean(es)) if es else float('nan')
def fit_gT(logit,gc):
    T=torch.ones(1,device=dev,requires_grad=True);o=torch.optim.LBFGS([T],lr=0.1,max_iter=50)
    def cl():o.zero_grad();l=F.cross_entropy(LS*logit/T.clamp(min=0.05),gc);l.backward();return l
    o.step(cl);return T.detach().clamp(min=0.05).item()
@torch.no_grad()
def diffuse(feat,logit,alpha=0.4,k=2,knn=8):
    f=F.normalize(feat.to(dev).float(),dim=1);P=F.softmax(LS*logit.to(dev).float(),1);sim=f@f.T;n=f.shape[0];kk=min(knn,n-1)
    if kk<1: return P
    val,idx=sim.topk(kk+1,dim=1);val=val[:,1:];idx=idx[:,1:];W=torch.zeros(n,n,device=dev);W.scatter_(1,idx,val.clamp(min=0));W=(W+W.T)/2;W=W/W.sum(1,keepdim=True).clamp(min=1e-6)
    out=P
    for _ in range(k): out=(1-alpha)*P+alpha*(W@out)
    return out
def three_ece(probs):
    p=probs[vv];g=vgc[vv];conf,pred=p.max(1);corr=(pred==g).float()
    return dict(ECE=round(ece_equalwidth(conf,corr),1), ECE_adaptive=round(ece_adaptive(conf,corr),1), ECE_classbal=round(ece_classbalanced(conf,corr,pred,g),1))
rng=np.random.RandomState(0); rows={k:[] for k in ['uncal','globalT','diffusion']}
Pd=torch.cat([diffuse(f,l) for (f,l,g) in VAL])
for s in range(SEEDS):
    perm=rng.permutation(NCLS); seen=set(sorted(perm[NOVEL:].tolist()))
    tseen=torch.tensor([g.item() in seen for g in tgc],device=dev)
    rows['uncal'].append(three_ece(F.softmax(LS*vl,1)))
    T=fit_gT(tl_[tseen],tgc[tseen]); rows['globalT'].append(three_ece(F.softmax(LS*vl/T,1)))
    rows['diffusion'].append(three_ece(Pd))
def agg(v): return {k:round(float(np.mean([r[k] for r in v])),1) for k in v[0]}
res={k:agg(v) for k,v in rows.items()}
json.dump(res,open(f'{CACHE}/metrics_adaece.json','w'),indent=1)
print('METHOD     ECE(eqwidth) ECE(adaptive) ECE(classbal)')
for k,r in res.items(): print(f"{k:10s} {r['ECE']:>10} {r['ECE_adaptive']:>12} {r['ECE_classbal']:>12}")
print('WROTE metrics_adaece.json',flush=True)
