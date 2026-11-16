#!/usr/bin/env python3
"""Route B: bbox/None-supervised entity-to-patch partial matching on GMNER TRAIN/DEV."""
from __future__ import annotations

import argparse, json, math, time, traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT=Path("/home/jyx/yixiao/MINER-My")
EXP=ROOT/"kais_extension/experiment24_literature_guided"
CACHE=ROOT/"kais_extension/experiment10_explicit_grounding/cache"


def geometry():
    centers=[];boxes=[]
    for j in range(196):
        x,y=j%14,j//14;x1,y1=x/14,y/14;x2,y2=(x+1)/14,(y+1)/14
        centers.append([(x1+x2)/2,(y1+y2)/2]);boxes.append([x1,y1,x2,y2])
    c=torch.tensor(centers,dtype=torch.float32);b=torch.tensor(boxes,dtype=torch.float32)
    return c,b,torch.cat([c,b],-1)


CENTERS,PATCH_BOXES,POS=geometry()


class DS(Dataset):
    def __init__(self,p):self.p=p
    def __len__(self):return len(self.p["y_ground"])
    def __getitem__(self,i):return {"h":self.p["h"][i].float(),"v":self.p["v"][i].float(),"y":self.p["y_ground"][i].float(),"box":self.p["gt_xyxy"][i].float()}


class Head(nn.Module):
    """One entity supplies mass to 196 patches or a learned dustbin."""
    def __init__(self,d):
        super().__init__();self.q=nn.Linear(d,d);self.k=nn.Linear(d+6,d);self.dust=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,1))
    def forward(self,h,v):
        pos=POS.to(v.device).unsqueeze(0).expand(v.size(0),-1,-1)
        score=(self.q(h).unsqueeze(1)*self.k(torch.cat([v,pos],-1))).sum(-1)/math.sqrt(v.size(-1))
        return torch.cat([score,self.dust(h)],-1)


def target(box,y,device):
    b=box;pb=PATCH_BOXES.to(device)
    ix1=torch.maximum(b[:,0,None],pb[None,:,0]);iy1=torch.maximum(b[:,1,None],pb[None,:,1])
    ix2=torch.minimum(b[:,2,None],pb[None,:,2]);iy2=torch.minimum(b[:,3,None],pb[None,:,3])
    overlap=(ix2-ix1).clamp_min(0)*(iy2-iy1).clamp_min(0)
    dist=overlap/(overlap.sum(-1,keepdim=True).clamp_min(1e-12))
    empty=overlap.sum(-1)<=0
    if empty.any():
        cx=(b[:,0]+b[:,2])/2;cy=(b[:,1]+b[:,3])/2;c=CENTERS.to(device)
        nearest=((c[None,:,0]-cx[:,None])**2+(c[None,:,1]-cy[:,None])**2).argmin(-1)
        dist[empty]=0;dist[empty,nearest[empty]]=1
    out=torch.zeros(b.size(0),197,device=device);out[:,:196]=dist*y[:,None];out[:,196]=1-y
    return out,(overlap>0).float()


def ranks(x):
    order=np.argsort(x);r=np.empty(len(x),float);i=0
    while i<len(x):
        j=i
        while j+1<len(x) and x[order[j+1]]==x[order[i]]:j+=1
        r[order[i:j+1]]=(i+j+2)/2;i=j+1
    return r


def auroc(y,s):
    y=np.asarray(y,int);s=np.asarray(s,float);p=y.sum();n=len(y)-p
    if p==0 or n==0:return float("nan")
    return float((ranks(s)[y==1].sum()-p*(p+1)/2)/(p*n))


def aupr(y,s):
    y=np.asarray(y,int);s=np.asarray(s,float);p=y.sum()
    if p==0:return float("nan")
    o=np.argsort(-s);yy=y[o];prec=np.cumsum(yy)/np.arange(1,len(y)+1)
    return float((prec*yy).sum()/p)


def prf(y,s,t):
    y=np.asarray(y,int);p=np.asarray(s)>=t;tp=((p==1)&(y==1)).sum();fp=((p==1)&(y==0)).sum();fn=((p==0)&(y==1)).sum()
    P=tp/(tp+fp) if tp+fp else 0;R=tp/(tp+fn) if tp+fn else 0;F=2*P*R/(P+R) if P+R else 0
    return float(P),float(R),float(F)


def evaluate(model,loader,device):
    model.eval();ys=[];scores=[];patch=[];targets=[];gt_boxes=[]
    with torch.no_grad():
        for b in loader:
            h=b["h"].to(device);v=b["v"].to(device);y=b["y"].to(device);box=b["box"].to(device)
            prob=model(h,v).softmax(-1);_,inside=target(box,y,device)
            ys.extend(y.cpu().tolist());scores.extend((1-prob[:,196]).cpu().tolist())
            g=y.bool()
            if g.any():
                patch.append(prob[g,:196].cpu());targets.append(inside[g].cpu());gt_boxes.extend(box[g].cpu().tolist())
    thresholds=[0.3,0.5,0.7];cand=[]
    for t in thresholds:
        P,R,F=prf(ys,scores,t);cand.append((F,t,P,R))
    F,t,P,R=max(cand)
    yy=np.asarray(ys,dtype=int);ss=np.asarray(scores);ung=yy==0
    dust_reject=float(np.mean(ss[ung]<t)) if ung.any() else float("nan")
    false_match=float(np.mean(ss[ung]>=t)) if ung.any() else float("nan")
    probs=torch.cat(patch).numpy();tar=torch.cat(targets).numpy();top=probs.argmax(1)
    pointing=float(np.mean([tar[i,top[i]]>0 for i in range(len(top))]))
    inside=(probs*tar).sum(1)/(probs.sum(1)+1e-12);area=tar.mean(1);lift=inside/(area+1e-12)
    pn=probs/(probs.sum(1,keepdims=True)+1e-12);ent=-(pn*np.log(np.clip(pn,1e-12,1))).sum(1)/math.log(196)
    # DEV-only deterministic heatmap-to-box rule selection.
    rules=[("thr",.3),("thr",.5),("thr",.7),("top",.1),("top",.2),("top",.3)]
    def decode(row,rule):
        kind,val=rule
        idx=np.where(row>=val)[0] if kind=="thr" else np.argsort(row)[-max(1,int(math.ceil(196*val))):]
        if len(idx)==0:idx=np.asarray([int(np.argmax(row))])
        pb=PATCH_BOXES[idx].numpy();return [pb[:,0].min(),pb[:,1].min(),pb[:,2].max(),pb[:,3].max()]
    def iou(a,b):
        ix1=max(a[0],b[0]);iy1=max(a[1],b[1]);ix2=min(a[2],b[2]);iy2=min(a[3],b[3]);inter=max(0,ix2-ix1)*max(0,iy2-iy1)
        aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]);bb=max(0,b[2]-b[0])*max(0,b[3]-b[1]);return inter/(aa+bb-inter+1e-12)
    box_candidates=[]
    for rule in rules:
        pred=[decode(row,rule) for row in probs];ious=[iou(a,b) for a,b in zip(pred,gt_boxes)];box_candidates.append((float(np.mean(ious)),rule,float(np.mean(np.asarray(ious)>=.5))))
    mean_iou,box_rule,acc05=max(box_candidates,key=lambda x:x[0])
    return {"groundability_auroc":auroc(ys,scores),"groundability_aupr":aupr(ys,scores),"groundability_precision":P,"groundability_recall":R,"groundability_f1":F,"threshold":t,
            "dustbin_rejection_accuracy":dust_reject,"false_match_rate":false_match,"groundability_accuracy":float(np.mean((ss>=t)==yy)),
            "matched_transport_mass":float(ss.mean()),"unmatched_dustbin_mass":float((1.0-ss).mean()),
            "patch_auroc":auroc(tar.reshape(-1),probs.reshape(-1)),"patch_aupr":aupr(tar.reshape(-1),probs.reshape(-1)),
            "pointing_accuracy":pointing,"inside_mass":float(inside.mean()),"area_expectation":float(area.mean()),"mass_lift":float(lift.mean()),"normalized_entropy":float(ent.mean()),
            "bbox_decode_rule":f"{box_rule[0]}_{box_rule[1]}","mean_iou":mean_iou,"acc_iou_05":acc05,
            "n_entities":len(ys),"n_groundable":int(sum(ys))}


def run(a):
    out=Path(a.run_dir);out.mkdir(parents=True,exist_ok=True);torch.manual_seed(42);np.random.seed(42)
    train=torch.load(CACHE/"features_train.pt",map_location="cpu",weights_only=False);dev=torch.load(CACHE/"features_dev.pt",map_location="cpu",weights_only=False)
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu");model=Head(int(train["h"].shape[-1])).to(device)
    tr=DataLoader(DS(train),batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(42));dv=DataLoader(DS(dev),batch_size=256,shuffle=False)
    b=next(iter(tr));h=b["h"].to(device);v=b["v"].to(device);y=b["y"].to(device);box=b["box"].to(device);tgt,_=target(box,y,device);logits=model(h,v);loss=-(tgt*logits.log_softmax(-1)).sum(-1).mean()
    loss.backward();grad=float(torch.sqrt(sum((p.grad.float()**2).sum() for p in model.parameters() if p.grad is not None)));finite=bool(torch.isfinite(loss));model.zero_grad(set_to_none=True)
    smoke={"finite":finite,"gradient_norm":grad,"train_entities":len(train["y_ground"]),"dev_entities":len(dev["y_ground"]),"h_shape":list(train["h"].shape),"v_shape":list(train["v"].shape)}
    (out/"smoke.json").write_text(json.dumps(smoke,indent=2)+"\n")
    if not finite or grad<=0:raise RuntimeError(smoke)
    if a.smoke:(out/"status.json").write_text(json.dumps({"status":"smoke_complete",**smoke},indent=2)+"\n");return
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4);best=-1;bad=0;hist=[];t0=time.time();torch.cuda.reset_peak_memory_stats()
    for ep in range(1,6):
        model.train();ls=[]
        for b in tr:
            h=b["h"].to(device);v=b["v"].to(device);y=b["y"].to(device);box=b["box"].to(device);tgt,_=target(box,y,device)
            logits=model(h,v);loss=-(tgt*logits.log_softmax(-1)).sum(-1).mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();ls.append(float(loss))
        met=evaluate(model,dv,device);proxy=.5*met["groundability_f1"]+.5*met["pointing_accuracy"]
        row={"epoch":ep,"train_loss":float(np.mean(ls)),"dev_proxy":proxy,**met};hist.append(row);print(json.dumps(row),flush=True)
        if proxy>best:best=proxy;bad=0;bestrow=row.copy();torch.save(model.state_dict(),out/"best_model.pth")
        else:bad+=1
        if bad>=2:break
    status={"status":"complete","route":"B","seed":42,"dataset":"GMNER TRAIN/DEV","best_epoch":bestrow["epoch"],"best_dev_proxy":bestrow["dev_proxy"],
            **{k:v for k,v in bestrow.items() if k not in ("epoch","dev_proxy","train_loss")},"gradient_norm":grad,"trainable_parameters":sum(p.numel() for p in model.parameters()),
            "gpu_time_seconds":time.time()-t0,"peak_gpu_memory_mb":torch.cuda.max_memory_allocated()/1024**2,"test_evaluated":False}
    (out/"epoch_metrics.json").write_text(json.dumps(hist,indent=2)+"\n");(out/"status.json").write_text(json.dumps(status,indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--run-dir",required=True);p.add_argument("--smoke",action="store_true");a=p.parse_args()
    try:run(a)
    except Exception as e:
        out=Path(a.run_dir);out.mkdir(parents=True,exist_ok=True);(out/"status.json").write_text(json.dumps({"status":"failed","route":"B","error":repr(e),"traceback":traceback.format_exc(),"test_evaluated":False},indent=2)+"\n");raise
