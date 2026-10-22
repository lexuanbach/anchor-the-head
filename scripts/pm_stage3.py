# Stage 3: on the cached dense-CLIP logits, add the extra baselines
# and metrics -- (a) generic graph-Laplacian smoothing (is the gain just ANY pairwise smoothing?),
# (b) a logit-space trust-region trained head (does collapse persist?), (c) AURC / risk-coverage, and
# per-pixel ECE (already the unit). 5 seen/novel splits on ADE20K-150 dense-CLIP cache.
import os,glob,numpy as np,torch,torch.nn.functional as F,math,json
dev='cuda';CACHE=os.environ.get('PMX_CACHE','cache');NCLS=150;LS=40.0;NOVEL=30;SEEDS=5;BINS=15
def load(split):
    out=[]
    for f in sorted(glob.glob(f'{CACHE}/{split}/*.npz')):
        d=np.load(f); out.append((torch.from_numpy(d['feat'].astype(np.float32)),
                                  torch.from_numpy(d['clslogit'].astype(np.float32)),
                                  torch.from_numpy(d['gt'].astype(np.int64))))
    return out
print('loading cache...',flush=True); VAL=load('val'); TR=load('train')
def cat(a): return torch.cat([x[0] for x in a]),torch.cat([x[1] for x in a]),torch.cat([x[2] for x in a])
vf,vl,vg=cat(VAL); tf,tl,tg=cat(TR)
vmask=vg>0; vl=vl.to(dev); vgc=(vg-1).clamp(min=0).to(dev).long(); vvalid=vmask.to(dev)
tmask=tg>0; tl_=tl[tmask].to(dev); tgc=(tg[tmask]-1).to(dev).long(); tf_=tf[tmask]

def metrics(probs,novel,with_aurc=False):
    p=probs[vvalid]; g=vgc[vvalid]; conf,pred=p.max(1); correct=(pred==g).float()
    def ece(sel):
        c=conf[sel]; a=correct[sel]
        if c.numel()==0: return float('nan')
        e=0.
        for b in range(BINS):
            m=(c>b/BINS)&(c<=(b+1)/BINS)
            if m.any(): e+=(m.float().mean()*(a[m].mean()-c[m].mean()).abs()).item()
        return 100*e
    isn=torch.zeros_like(g,dtype=torch.bool)
    for c in novel: isn|=(g==c)
    inter=torch.zeros(NCLS,device=dev); union=torch.zeros(NCLS,device=dev)
    for c in range(NCLS): inter[c]=((g==c)&(pred==c)).sum(); union[c]=((g==c)|(pred==c)).sum()
    iou=torch.where(union>0,inter/union,torch.full_like(inter,float('nan')))
    nv=[c for c in novel if not torch.isnan(iou[c])]
    miou=100*iou[~torch.isnan(iou)].mean().item()
    niou=100*iou[torch.tensor(nv,device=dev)].mean().item() if nv else float('nan')
    aurc=float('nan')
    if with_aurc:
        order=conf.argsort(descending=True); err=(1-correct)[order]
        cum=torch.cumsum(err,0)/torch.arange(1,len(err)+1,device=dev)
        aurc=100*cum.mean().item()
    return dict(ECE=ece(torch.ones_like(isn)),ECE_novel=ece(isn),mIoU=miou,novelIoU=niou,AURC=aurc)

def fit_gT(logit,gc):
    T=torch.ones(1,device=dev,requires_grad=True); o=torch.optim.LBFGS([T],lr=0.1,max_iter=50)
    def cl(): o.zero_grad(); l=F.cross_entropy(LS*logit/T.clamp(min=0.05),gc); l.backward(); return l
    o.step(cl); return T.detach().clamp(min=0.05).item()
@torch.no_grad()
def graph(feat,knn=8):
    f=F.normalize(feat.to(dev).float(),dim=1); sim=f@f.T; n=f.shape[0]; kk=min(knn,n-1)
    if kk<1: return None
    val,idx=sim.topk(kk+1,dim=1); val=val[:,1:]; idx=idx[:,1:]
    W=torch.zeros(n,n,device=dev); W.scatter_(1,idx,val.clamp(min=0)); W=(W+W.T)/2
    return W/W.sum(1,keepdim=True).clamp(min=1e-6)
@torch.no_grad()
def diffuse(feat,logit,alpha=0.4,k=2):           # ANCHORED (ours)
    W=graph(feat); P=F.softmax(LS*logit.to(dev).float(),1)
    if W is None: return P
    out=P
    for _ in range(k): out=(1-alpha)*P+alpha*(W@out)
    return out
@torch.no_grad()
def laplacian(feat,logit,k=2):                    # GENERIC smoothing, NO anchor
    W=graph(feat); P=F.softmax(LS*logit.to(dev).float(),1)
    if W is None: return P
    out=P
    for _ in range(k): out=W@out                  # pure propagation, no CLIP prior anchor
    return out
def train_decoder(X,base,y,space='feat',tau=2.0,ep=12):
    h=torch.nn.Sequential(torch.nn.Linear(X.shape[1],256),torch.nn.ReLU(),torch.nn.Linear(256,NCLS)).to(dev)
    op=torch.optim.Adam(h.parameters(),lr=1e-3); X=X.to(dev)
    for e in range(ep):
        for i in range(0,len(X),8192):
            op.zero_grad()
            r=h(X[i:i+8192])
            r=0.5*r if space=='feat' else torch.clamp(r,-tau,tau)   # logit-space trust region
            loss=F.cross_entropy(base[i:i+8192]+r,y[i:i+8192]); loss.backward(); op.step()
    return h,space,tau

rng=np.random.RandomState(0); rows={k:[] for k in ['uncal','globalT','diffusion','laplacian','decoder_feat','decoder_logit_tr']}
for s in range(SEEDS):
    perm=rng.permutation(NCLS); novel=sorted(perm[:NOVEL].tolist()); seen=set(sorted(perm[NOVEL:].tolist()))
    tseen=torch.tensor([g.item() in seen for g in tgc],device=dev)
    tl_s=tl_[tseen]; tg_s=tgc[tseen]; tf_s=tf_[tseen.cpu()]
    rows['uncal'].append(metrics(F.softmax(LS*vl,1),novel,True))
    T=fit_gT(tl_s,tg_s); rows['globalT'].append(metrics(F.softmax(LS*vl/T,1),novel,True))
    rows['diffusion'].append(metrics(torch.cat([diffuse(f,l) for (f,l,g) in VAL]),novel,True))
    rows['laplacian'].append(metrics(torch.cat([laplacian(f,l) for (f,l,g) in VAL]),novel,True))
    hf,_,_=train_decoder(tf_s,LS*tl_s,tg_s,'feat')
    with torch.no_grad(): rows['decoder_feat'].append(metrics(F.softmax(LS*vl+0.5*hf(vf.to(dev)),1),novel,True))
    hl,_,tau=train_decoder(tf_s,LS*tl_s,tg_s,'logit',2.0)
    with torch.no_grad(): rows['decoder_logit_tr'].append(metrics(F.softmax(LS*vl+torch.clamp(hl(vf.to(dev)),-2.0,2.0),1),novel,True))
    print(f"[seed {s}] novelIoU diff={rows['diffusion'][-1]['novelIoU']:.1f} lap={rows['laplacian'][-1]['novelIoU']:.1f} dec_feat={rows['decoder_feat'][-1]['novelIoU']:.1f} dec_logitTR={rows['decoder_logit_tr'][-1]['novelIoU']:.1f}",flush=True)

def agg(v):
    ks=v[0].keys(); out={}
    for k in ks:
        a=np.array([r[k] for r in v]); out[k]={'mean':round(float(np.nanmean(a)),2),'std':round(float(np.nanstd(a,ddof=1)),2)}
    return out
res={k:agg(v) for k,v in rows.items()}
json.dump({'config':{'dataset':'ADE20K-150 dense-CLIP','novel':NOVEL,'seeds':SEEDS,'tau_logit':2.0},'summary':res},open(f'{CACHE}/metrics_stage3.json','w'),indent=1)
print('\n=== STAGE3 SUMMARY ===')
for k,r in res.items(): print(f"{k:17s} mIoU={r['mIoU']['mean']:.1f} ECE={r['ECE']['mean']:.1f} ECEnov={r['ECE_novel']['mean']:.1f} novelIoU={r['novelIoU']['mean']:.1f} AURC={r['AURC']['mean']}")
print('WROTE metrics_stage3.json',flush=True)
