# Stage 1: training-free dense CLIP (SCLIP CSA) on ADE20K-150.
# Reports val mIoU and caches per-image patch (feature, gt) for the calibration/collapse study.
import os, sys, glob, numpy as np, torch, torch.nn.functional as F
from PIL import Image
SCLIP=os.environ.get('PMX_SCLIP','SCLIP'); sys.path.append(SCLIP)
import clip
from prompts.imagenet_template import openai_imagenet_template
ADE=os.environ.get('PMX_ADE_ROOT','data/ADEChallengeData2016'); NAMES=os.environ.get('PMX_ADE_NAMES',os.path.join(SCLIP,'configs/cls_ade20k.txt'))
OUT=os.environ.get('PMX_CACHE','cache'); os.makedirs(OUT,exist_ok=True)
MEAN=torch.tensor([0.48145466,0.4578275,0.40821073]).view(1,3,1,1)
STD =torch.tensor([0.26862954,0.26130258,0.27577711]).view(1,3,1,1)
dev='cuda'

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

@torch.no_grad()
def feat_grid(net,img,ps):
    # returns per-patch projected features [gh*gw, D] and grid (gh,gw) for one crop-free full pass via sliding accumulation
    f=net.encode_image(img,return_all=True,csa=True); f=f/f.norm(dim=-1,keepdim=True); f=f[:,1:]   # [1,N,D]
    gh=img.shape[-2]//ps; gw=img.shape[-1]//ps
    return f[0], gh, gw

@torch.no_grad()
def slide_feat(net,img,ps,qfeat,crop=224,stride=112):
    # accumulate per-patch class logits over sliding windows at patch grid of the FULL image
    _,_,H,W=img.shape; gh=H//ps; gw=W//ps; Nq=qfeat.shape[0]; D=qfeat.shape[1]
    acc_logit=torch.zeros(Nq,gh,gw,device=dev); acc_feat=torch.zeros(D,gh,gw,device=dev); cnt=torch.zeros(1,gh,gw,device=dev)
    gcrop=crop//ps; gstride=stride//ps
    hg=max(H-crop+stride-1,0)//stride+1; wg=max(W-crop+stride-1,0)//stride+1
    for i in range(hg):
        for j in range(wg):
            y1=min(i*stride,H-crop) if H>=crop else 0; x1=min(j*stride,W-crop) if W>=crop else 0
            y2=min(y1+crop,H); x2=min(x1+crop,W); y1=max(y2-crop,0); x1=max(x2-crop,0)
            fc=net.encode_image(img[:,:,y1:y2,x1:x2],return_all=True,csa=True); fc=fc/fc.norm(dim=-1,keepdim=True); fc=fc[0,1:]  # [gc*gc,D]
            gc_h=(y2-y1)//ps; gc_w=(x2-x1)//ps
            fc=fc.reshape(gc_h,gc_w,D).permute(2,0,1)
            lc=(fc.permute(1,2,0)@qfeat.T).permute(2,0,1)  # [Nq,gc_h,gc_w]
            py1=y1//ps; px1=x1//ps
            acc_logit[:,py1:py1+gc_h,px1:px1+gc_w]+=lc
            acc_feat[:,py1:py1+gc_h,px1:px1+gc_w]+=fc
            cnt[:,py1:py1+gc_h,px1:px1+gc_w]+=1
    cnt=cnt.clamp(min=1)
    return (acc_logit/cnt), (acc_feat/cnt), gh, gw

def load_img(p,short=336,maxl=2048):
    im=Image.open(p).convert('RGB'); w,h=im.size; s=short/min(w,h)
    if max(w,h)*s>maxl: s=maxl/max(w,h)
    im=im.resize((round(w*s),round(h*s)),Image.BILINEAR)
    t=torch.from_numpy(np.array(im)).permute(2,0,1).float()/255.
    return (t.unsqueeze(0)-MEAN)/STD

def run(split, files, qfeat, qidx, ncls, net, ps, cache=True):
    inter=np.zeros(ncls); union=np.zeros(ncls); od=os.path.join(OUT,split); os.makedirs(od,exist_ok=True)
    for k,ip in enumerate(files):
        gp=f"{ADE}/annotations/validation/{os.path.basename(ip).replace('.jpg','.png')}" if split=='val' else \
           f"{ADE}/annotations/training/{os.path.basename(ip).replace('.jpg','.png')}"
        gt=np.array(Image.open(gp)).astype(np.int64)
        img=load_img(ip).to(dev).type(net.dtype)
        logit,feat,gh,gw=slide_feat(net,img,ps,qfeat)            # [Nq,gh,gw],[D,gh,gw]
        # per-class via synonym max
        cl=torch.full((ncls,gh,gw),-1e4,device=dev)
        for c in range(ncls):
            m=(qidx==c)
            if m.any(): cl[c]=logit[m].max(0).values
        # mIoU at full res
        up=F.interpolate(cl.unsqueeze(0),size=gt.shape,mode='bilinear',align_corners=False)[0]
        pred=up.argmax(0).cpu().numpy(); valid=gt>0; g=gt[valid]-1; p=pred[valid]
        for c in range(ncls):
            gi=(g==c); pi=(p==c); inter[c]+=np.logical_and(gi,pi).sum(); union[c]+=np.logical_or(gi,pi).sum()
        if cache:
            # patch-grid GT via nearest downsample
            gtt=torch.from_numpy(gt).float().view(1,1,*gt.shape)
            gtp=F.interpolate(gtt,size=(gh,gw),mode='nearest')[0,0].numpy().astype(np.int16)  # 0=ignore,1..150
            np.savez(os.path.join(od,f'{k:05d}.npz'),
                     feat=feat.permute(1,2,0).reshape(-1,feat.shape[0]).half().cpu().numpy(),
                     clslogit=cl.permute(1,2,0).reshape(-1,ncls).half().cpu().numpy(),
                     gt=gtp.reshape(-1), grid=np.array([gh,gw]))
        if (k+1)%200==0:
            iou=inter[union>0]/union[union>0]; print(f'  [{split} {k+1}/{len(files)}] mIoU={100*iou.mean():.2f}',flush=True)
    iou=inter[union>0]/union[union>0]; print(f'{split} FINAL mIoU={100*iou.mean():.2f} ({(union>0).sum()} cls, {len(files)} imgs)',flush=True)
    return 100*iou.mean()

def main():
    net,_=clip.load('ViT-B/16',device=dev,jit=False); net.eval()
    ps=net.visual.patch_size; qfeat,qidx,ncls=build_text(net)
    np.save(os.path.join(OUT,'qfeat.npy'),qfeat.half().cpu().numpy())
    print(f'ncls={ncls} nq={qfeat.shape[0]} D={qfeat.shape[1]} patch={ps}',flush=True)
    val=sorted(glob.glob(f'{ADE}/images/validation/*.jpg'))
    tr =sorted(glob.glob(f'{ADE}/images/training/*.jpg'))[:600]
    m=run('val',val,qfeat,qidx,ncls,net,ps)
    run('train',tr,qfeat,qidx,ncls,net,ps)
    print(f'DONE headline val mIoU={m:.2f}',flush=True); open(os.path.join(OUT,'DONE'),'w').write(str(m))

if __name__=='__main__': main()
