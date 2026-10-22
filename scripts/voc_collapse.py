# Cross-DATASET check of the collapse law: run the dense-CLIP calibration+collapse study on
# PASCAL VOC2012 (20 object classes, different domain from ADE20K-150 scenes). Self-contained.
import os, sys, glob, numpy as np, torch, torch.nn.functional as F, math, json
from PIL import Image
SCLIP=os.environ.get('PMX_SCLIP','SCLIP'); sys.path.append(SCLIP)
import clip
from prompts.imagenet_template import openai_imagenet_template
dev='cuda'; VOC=os.environ.get('PMX_VOC_ROOT','data/VOCdevkit/VOC2012'); NAMES=os.environ.get('PMX_VOC_NAMES',os.path.join(SCLIP,'configs/cls_voc20.txt')); CACHE=os.environ.get('PMX_CACHE','cache')
NCLS=20; NOVEL=6; SEEDS=5; BINS=15; LS=40.0
MEAN=torch.tensor([0.48145466,0.4578275,0.40821073]).view(1,3,1,1); STD=torch.tensor([0.26862954,0.26130258,0.27577711]).view(1,3,1,1)
def mean(x): return sum(x)/len(x)
def sstd(x): m=mean(x); return math.sqrt(sum((v-m)**2 for v in x)/(len(x)-1)) if len(x)>1 else 0.0

def cls_idx():
    lines=[l.strip() for l in open(NAMES)]; w,idx=[],[]
    for i,l in enumerate(lines):
        for x in l.split(', '): w.append(x); idx.append(i)
    return w,idx,len(lines)
@torch.no_grad()
def build_text(net):
    w,idx,n=cls_idx(); fs=[]
    for q in w:
        tok=clip.tokenize([t(q) for t in openai_imagenet_template]).to(dev)
        f=net.encode_text(tok); f/=f.norm(dim=-1,keepdim=True); f=f.mean(0); f/=f.norm(); fs.append(f.unsqueeze(0))
    return torch.cat(fs,0), torch.tensor(idx,device=dev), n
def load_img(p,short=336,maxl=2048):
    im=Image.open(p).convert('RGB'); w,h=im.size; s=short/min(w,h)
    if max(w,h)*s>maxl: s=maxl/max(w,h)
    im=im.resize((round(w*s),round(h*s)),Image.BILINEAR)
    t=torch.from_numpy(np.array(im)).permute(2,0,1).float()/255.
    return (t.unsqueeze(0)-MEAN)/STD
@torch.no_grad()
def slide_feat(net,img,ps,qfeat,crop=224,stride=112):
    _,_,H,W=img.shape; gh=H//ps; gw=W//ps; D=qfeat.shape[1]; Nq=qfeat.shape[0]
    al=torch.zeros(Nq,gh,gw,device=dev); af=torch.zeros(D,gh,gw,device=dev); cnt=torch.zeros(1,gh,gw,device=dev)
    hg=max(H-crop+stride-1,0)//stride+1; wg=max(W-crop+stride-1,0)//stride+1
    for i in range(hg):
        for j in range(wg):
            y1=min(i*stride,H-crop) if H>=crop else 0; x1=min(j*stride,W-crop) if W>=crop else 0
            y2=min(y1+crop,H); x2=min(x1+crop,W); y1=max(y2-crop,0); x1=max(x2-crop,0)
            fc=net.encode_image(img[:,:,y1:y2,x1:x2],return_all=True,csa=True); fc=fc/fc.norm(dim=-1,keepdim=True); fc=fc[0,1:]
            gch=(y2-y1)//ps; gcw=(x2-x1)//ps; fc=fc.reshape(gch,gcw,D).permute(2,0,1)
            lc=(fc.permute(1,2,0)@qfeat.T).permute(2,0,1); py=y1//ps; px=x1//ps
            al[:,py:py+gch,px:px+gcw]+=lc; af[:,py:py+gch,px:px+gcw]+=fc; cnt[:,py:py+gch,px:px+gcw]+=1
    cnt=cnt.clamp(min=1); return al/cnt, af/cnt, gh, gw

def run_split(net,ps,qfeat,qidx,ids,cache):
    inter=np.zeros(NCLS); union=np.zeros(NCLS)
    for k,iid in enumerate(ids):
        ip=f'{VOC}/JPEGImages/{iid}.jpg'; gp=f'{VOC}/SegmentationClass/{iid}.png'
        gt=np.array(Image.open(gp)).astype(np.int64)   # 0=bg,1..20,255=void
        img=load_img(ip).to(dev).type(net.dtype)
        logit,feat,gh,gw=slide_feat(net,img,ps,qfeat)
        cl=torch.full((NCLS,gh,gw),-1e4,device=dev)
        for c in range(NCLS):
            m=(qidx==c)
            if m.any(): cl[c]=logit[m].max(0).values
        up=F.interpolate(cl.unsqueeze(0),size=gt.shape,mode='bilinear',align_corners=False)[0]
        pred=up.argmax(0).cpu().numpy(); valid=(gt>=1)&(gt<=NCLS); g=gt[valid]-1; p=pred[valid]
        for c in range(NCLS):
            inter[c]+=((g==c)&(p==c)).sum(); union[c]+=((g==c)|(p==c)).sum()
        if cache is not None:
            gtt=torch.from_numpy(gt).float().view(1,1,*gt.shape)
            gtp=F.interpolate(gtt,size=(gh,gw),mode='nearest')[0,0].numpy().astype(np.int16)
            cache.append((feat.permute(1,2,0).reshape(-1,feat.shape[0]).half().cpu(),
                          cl.permute(1,2,0).reshape(-1,NCLS).half().cpu(),
                          torch.from_numpy(gtp.reshape(-1))))
    iou=inter[union>0]/union[union>0]; return 100*iou.mean()

# ---- ECE/IoU + calibrators (NCLS=20) ----
def ece_iou(probs,gtc,valid,novel):
    p=probs[valid]; g=gtc[valid]; conf,pred=p.max(1); correct=(pred==g).float()
    def ece(sel):
        c=conf[sel]; a=correct[sel]
        if c.numel()==0: return float('nan')
        e=0.0
        for b in range(BINS):
            m=(c>b/BINS)&(c<=(b+1)/BINS)
            if m.any(): e+=(m.float().mean()*(a[m].mean()-c[m].mean()).abs()).item()
        return 100*e
    isn=torch.zeros_like(g,dtype=torch.bool)
    for c in novel: isn|=(g==c)
    inter=torch.zeros(NCLS,device=dev); union=torch.zeros(NCLS,device=dev)
    for c in range(NCLS):
        inter[c]=((g==c)&(pred==c)).sum(); union[c]=((g==c)|(pred==c)).sum()
    iou=torch.where(union>0,inter/union,torch.full_like(inter,float('nan')))
    nv=[c for c in novel if not torch.isnan(iou[c])]
    miou=100*iou[~torch.isnan(iou)].mean().item()
    niou=100*iou[torch.tensor(nv,device=dev)].mean().item() if nv else float('nan')
    return ece(torch.ones_like(isn)),ece(isn),ece(~isn),miou,niou
def fit_gT(logit,gc):
    T=torch.ones(1,device=dev,requires_grad=True); o=torch.optim.LBFGS([T],lr=0.1,max_iter=50)
    def cl(): o.zero_grad(); l=F.cross_entropy(LS*logit/T.clamp(min=0.05),gc); l.backward(); return l
    o.step(cl); return T.detach().clamp(min=0.05).item()
def fit_pcT(logit,gc,seen):
    sm=torch.zeros(NCLS,device=dev,dtype=torch.bool); sm[torch.tensor(list(seen),device=dev)]=True
    T=torch.ones(NCLS,device=dev,requires_grad=True); o=torch.optim.LBFGS([T],lr=0.1,max_iter=80)
    def cl(): o.zero_grad(); Tc=torch.where(sm,T.clamp(min=0.05),torch.ones_like(T)); l=F.cross_entropy(LS*logit/Tc.view(1,-1),gc); l.backward(); return l
    o.step(cl); Tf=T.detach().clamp(min=0.05); return torch.where(sm,Tf,torch.ones_like(Tf))
def train_dec(X,base,y,ep=12):
    h=torch.nn.Sequential(torch.nn.Linear(X.shape[1],256),torch.nn.ReLU(),torch.nn.Linear(256,NCLS)).to(dev)
    op=torch.optim.Adam(h.parameters(),lr=1e-3)
    for e in range(ep):
        for i in range(0,len(X),8192):
            op.zero_grad(); sl=base[i:i+8192]+0.5*h(X[i:i+8192]); l=F.cross_entropy(sl,y[i:i+8192]); l.backward(); op.step()
    return h
@torch.no_grad()
def diffuse(feat,logit,alpha=0.4,k=2,knn=8):
    f=F.normalize(feat.to(dev).float(),dim=1); P=F.softmax(LS*logit.to(dev).float(),1); sim=f@f.T; n=f.shape[0]; kk=min(knn,n-1)
    if kk<1: return P
    val,idx=sim.topk(kk+1,dim=1); val=val[:,1:]; idx=idx[:,1:]
    W=torch.zeros(n,n,device=dev); W.scatter_(1,idx,val.clamp(min=0)); W=(W+W.T)/2; W=W/W.sum(1,keepdim=True).clamp(min=1e-6)
    out=P
    for _ in range(k): out=(1-alpha)*P+alpha*(W@out)
    return out

def main():
    net,_=clip.load('ViT-B/16',device=dev,jit=False); net.eval(); ps=net.visual.patch_size
    qfeat,qidx,_=build_text(net)
    val=[l.strip() for l in open(f'{VOC}/ImageSets/Segmentation/val.txt')]
    tr =[l.strip() for l in open(f'{VOC}/ImageSets/Segmentation/train.txt')][:400]
    print(f'VOC val={len(val)} train={len(tr)} NCLS={NCLS}',flush=True)
    VC=[]; m=run_split(net,ps,qfeat,qidx,val,VC); print(f'VOC headline mIoU={m:.2f}',flush=True)
    TC=[]; run_split(net,ps,qfeat,qidx,tr,TC)
    vf=torch.cat([x[0] for x in VC]).float(); vl=torch.cat([x[1] for x in VC]).float().to(dev); vg=torch.cat([x[2] for x in VC])
    tf=torch.cat([x[0] for x in TC]).float(); tl=torch.cat([x[1] for x in TC]).float(); tg=torch.cat([x[2] for x in TC])
    vmask=(vg>=1)&(vg<=NCLS); vgc=(vg-1).clamp(min=0).to(dev).long(); vvalid=vmask.to(dev)
    tmask=(tg>=1)&(tg<=NCLS); tl_=tl[tmask].to(dev); tgc=(tg[tmask]-1).to(dev).long()
    rng=np.random.RandomState(0); rows={k:[] for k in ['uncal','globalT','perclassT','diffusion','decoder']}
    for s in range(SEEDS):
        perm=rng.permutation(NCLS); novel=sorted(perm[:NOVEL].tolist()); seen=sorted(perm[NOVEL:].tolist())
        ss=set(seen); tseen=torch.tensor([g.item() in ss for g in tgc],device=dev)
        tl_s=tl_[tseen]; tg_s=tgc[tseen]
        rows['uncal'].append(ece_iou(F.softmax(LS*vl,1),vgc,vvalid,novel))
        T=fit_gT(tl_s,tg_s); rows['globalT'].append(ece_iou(F.softmax(LS*vl/T,1),vgc,vvalid,novel))
        Tc=fit_pcT(tl_s,tg_s,seen); rows['perclassT'].append(ece_iou(F.softmax(LS*vl/Tc.view(1,-1),1),vgc,vvalid,novel))
        Pd=torch.cat([diffuse(f,l) for (f,l,g) in VC]); rows['diffusion'].append(ece_iou(Pd,vgc,vvalid,novel))
        h=train_dec(tf[tmask][tseen.cpu()].to(dev), LS*tl_s, tg_s)
        with torch.no_grad(): Pde=F.softmax(LS*vl+0.5*h(vf.to(dev)),1)
        rows['decoder'].append(ece_iou(Pde,vgc,vvalid,novel))
        print(f'[seed {s}] novelIoU uncal={rows["uncal"][-1][4]:.1f} gT={rows["globalT"][-1][4]:.1f} pcT={rows["perclassT"][-1][4]:.1f} diff={rows["diffusion"][-1][4]:.1f} dec={rows["decoder"][-1][4]:.1f}',flush=True)
    def agg(v):
        a=np.array(v); mn=np.nanmean(a,0); sd=np.nanstd(a,0,ddof=1); ks=['ECE','ECE_novel','ECE_seen','mIoU','novelIoU']
        return {k:{'mean':round(float(mn[i]),2),'std':round(float(sd[i]),2)} for i,k in enumerate(ks)}
    res={k:agg(v) for k,v in rows.items()}
    json.dump({'config':{'dataset':'PASCAL VOC2012','backbone':'training-free dense CLIP ViT-B/16 (SCLIP-CSA)','NCLS':NCLS,'NOVEL':NOVEL,'headline_mIoU':round(m,2)},'summary':res}, open(os.path.join(CACHE,'metrics_voc.json'),'w'),indent=1)
    print('\n=== VOC SUMMARY ===')
    for k,r in res.items(): print(f"{k:11s} mIoU={r['mIoU']['mean']:.1f} ECE={r['ECE']['mean']:.1f} novelIoU={r['novelIoU']['mean']:.1f}")
    print('WROTE metrics_voc.json',flush=True)
if __name__=='__main__': main()
