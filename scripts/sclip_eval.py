import os, sys, glob, argparse, numpy as np, torch, torch.nn.functional as F
from PIL import Image
SCLIP=os.environ.get('PMX_SCLIP','SCLIP'); sys.path.append(SCLIP)
import clip
from prompts.imagenet_template import openai_imagenet_template

ADE=os.environ.get('PMX_ADE_ROOT','data/ADEChallengeData2016')
NAMES=os.environ.get('PMX_ADE_NAMES',os.path.join(SCLIP,'configs/cls_ade20k.txt'))
MEAN=torch.tensor([0.48145466,0.4578275,0.40821073]).view(1,3,1,1)
STD =torch.tensor([0.26862954,0.26130258,0.27577711]).view(1,3,1,1)

def get_cls_idx(path):
    lines=[l.strip() for l in open(path)]
    words,idx=[],[]
    for i,l in enumerate(lines):
        ws=l.split(', ')
        words+=ws; idx+=[i]*len(ws)
    return words, idx, len(lines)

@torch.no_grad()
def build_text(net, device):
    words,idx,ncls=get_cls_idx(NAMES)
    feats=[]
    for w in words:
        tok=clip.tokenize([t(w) for t in openai_imagenet_template]).to(device)
        f=net.encode_text(tok); f/=f.norm(dim=-1,keepdim=True)
        f=f.mean(0); f/=f.norm(); feats.append(f.unsqueeze(0))
    return torch.cat(feats,0), torch.tensor(idx,device=device), ncls

@torch.no_grad()
def forward_feature(net, img, qfeat, ps):
    feat=net.encode_image(img,return_all=True,csa=True)
    feat=feat/feat.norm(dim=-1,keepdim=True)
    feat=feat[:,1:]
    logits=feat@qfeat.T                          # [1, N, Nq]
    w=img.shape[-2]//ps; h=img.shape[-1]//ps
    logits=logits.permute(0,2,1).reshape(1,-1,w,h)
    return F.interpolate(logits,size=img.shape[-2:],mode='bilinear',align_corners=False)

@torch.no_grad()
def slide(net,img,qfeat,ps,crop=224,stride=112):
    _,_,H,W=img.shape
    hg=max(H-crop+stride-1,0)//stride+1; wg=max(W-crop+stride-1,0)//stride+1
    Nq=qfeat.shape[0]
    pred=img.new_zeros((1,Nq,H,W)); cnt=img.new_zeros((1,1,H,W))
    for i in range(hg):
        for j in range(wg):
            y1=min(i*stride,H-crop) if H>=crop else 0; x1=min(j*stride,W-crop) if W>=crop else 0
            y2=min(y1+crop,H); x2=min(x1+crop,W); y1=max(y2-crop,0); x1=max(x2-crop,0)
            cl=forward_feature(net,img[:,:,y1:y2,x1:x2],qfeat,ps)
            pred[:,:,y1:y2,x1:x2]+=cl; cnt[:,:,y1:y2,x1:x2]+=1
    return pred/cnt

def load_img(p,short=336,maxl=2048):
    im=Image.open(p).convert('RGB'); w,h=im.size
    s=short/min(w,h)
    if max(w,h)*s>maxl: s=maxl/max(w,h)
    nw,nh=round(w*s),round(h*s)
    im=im.resize((nw,nh),Image.BILINEAR)
    t=torch.from_numpy(np.array(im)).permute(2,0,1).float()/255.
    return (t.unsqueeze(0)-MEAN)/STD

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--n',type=int,default=200); ap.add_argument('--model',default='ViT-B/16'); ap.add_argument('--ls',type=float,default=40.)
    a=ap.parse_args(); dev='cuda'
    net,_=clip.load(a.model,device=dev,jit=False); net.eval()
    ps=net.visual.patch_size if hasattr(net.visual,'patch_size') else 16
    qfeat,qidx,ncls=build_text(net,dev)
    print(f'model={a.model} ncls={ncls} nq={qfeat.shape[0]} patch={ps}',flush=True)
    imgs=sorted(glob.glob(f'{ADE}/images/validation/*.jpg'))[:a.n]
    inter=np.zeros(ncls); union=np.zeros(ncls)
    for k,ip in enumerate(imgs):
        gp=f"{ADE}/annotations/validation/{os.path.basename(ip).replace('.jpg','.png')}"
        gt=np.array(Image.open(gp)).astype(np.int64)
        img=load_img(ip).to(dev).type(net.dtype)
        ql=slide(net,img,qfeat,ps)                       # [1,Nq,h,w]
        ql=F.interpolate(ql,size=gt.shape,mode='bilinear',align_corners=False)[0]  # [Nq,H,W]
        # group synonyms -> per class (max)
        cls_logits=torch.full((ncls,)+gt.shape,-1e4,device=dev)
        for c in range(ncls):
            m=(qidx==c)
            if m.any(): cls_logits[c]=ql[m].max(0).values
        pred=cls_logits.argmax(0).cpu().numpy()          # 0..ncls-1
        valid=gt>0; g=gt[valid]-1; p=pred[valid]
        for c in range(ncls):
            gi=(g==c); pi=(p==c)
            inter[c]+=np.logical_and(gi,pi).sum(); union[c]+=np.logical_or(gi,pi).sum()
        if (k+1)%50==0:
            iou=inter[union>0]/union[union>0]; print(f'  [{k+1}/{len(imgs)}] running mIoU={100*iou.mean():.2f} ({len(iou)} cls)',flush=True)
    iou=inter[union>0]/union[union>0]
    print(f'FINAL mIoU={100*iou.mean():.2f} over {(union>0).sum()} classes, {len(imgs)} imgs',flush=True)

if __name__=='__main__': main()
