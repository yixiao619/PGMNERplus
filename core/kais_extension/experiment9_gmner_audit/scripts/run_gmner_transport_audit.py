#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,math,os,sys,importlib.util,random
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from PIL import Image

ROOT=Path(__file__).resolve().parents[3]
EXP=ROOT/'kais_extension/experiment9_gmner_audit'; RES=EXP/'results'; CACHE=EXP/'cache'
S_CKPT=0.594940338282

def write_json(p,o): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(o,indent=2,ensure_ascii=False,sort_keys=True)+'\n',encoding='utf-8')
def write_csv(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True); keys=[]
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)

def parse_args():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',default=str(EXP/'cache/gmner_pgmner_format'))
    ap.add_argument('--labels-file',default=str(EXP/'cache/gmner_pgmner_format/labels.txt'))
    ap.add_argument('--checkpoint',default='/data/jyx/MINER-My/outputs/pp_ib_partial_ot_grid_dc10_m05_seed42/best_model.pth')
    ap.add_argument('--split',default='test')
    ap.add_argument('--gpu-id',default='1')
    ap.add_argument('--batch-size',type=int,default=8)
    ap.add_argument('--limit-samples',type=int,default=0)
    return ap.parse_args()

def load_pgmner(cli):
    os.chdir(ROOT); sys.path.insert(0,str(ROOT))
    sys.argv=['main-mner-robert.py','--data_dir',cli.data_dir,'--output_dir',str(RES/'pgmner_side_outputs'),'--model_name_or_path','pretrain/roberta-large','--tokenizer_name','pretrain/roberta-large','--vit_name','pretrain/clip-vit-base-patch16','--gpu_id',cli.gpu_id,'--do_train','false','--do_eval','false','--do_predict','false','--batch_size',str(cli.batch_size),'--max_span_len','4','--max_seq_len','128','--seed','42','--use_pp_ib','--use_got','--got_mode','partial_sinkhorn','--ot_sinkhorn_iter','30','--partial_mass_init','0.5','--dust_cost','1.0','--entropy_eps','0.1','--ot_alpha','0.05','--auto_img_dims','true','--use_hf_clip_vision','true']
    spec=importlib.util.spec_from_file_location('pgmner_main_exp9',ROOT/'main-mner-robert.py')
    mod=importlib.util.module_from_spec(spec); assert spec and spec.loader; spec.loader.exec_module(mod)
    mod.args.labels=cli.labels_file
    return mod

def load_with_remap(model,state):
    try: model.load_state_dict(state,strict=True); return 'strict'
    except RuntimeError:
        old='image_model._hf.encoder.vision_model.'; new='image_model._hf.encoder.'; mapped={}; changed=0
        for k,v in state.items():
            if k.startswith(old): mapped[new+k[len(old):]]=v; changed+=1
            else: mapped[k]=v
        if not changed: raise
        model.load_state_dict(mapped,strict=True); return f'legacy_clip_key_remap:{changed}'

def read_sidecar():
    out={}
    p=EXP/'cache/gmner_pgmner_format/gmner_entity_sidecar.jsonl'
    split_counts=defaultdict(int)
    for line in p.open(encoding='utf-8'):
        r=json.loads(line); split=r['split']
        split_counts[split]+=1
        out[(split,str(r['sample_id']))]=r
        out[(split,f'{split}-{split_counts[split]}')]=r
    return out

def patch_centers(n=196):
    g=int(round(math.sqrt(n)))
    return [((j%g+0.5)/g,(j//g+0.5)/g) for j in range(n)]

def bbox_norm(b, image_path):
    if not b: return None
    try:
        w,h=Image.open(image_path).size
    except Exception:
        return None
    x1,y1,x2,y2=b
    if max(abs(x1),abs(y1),abs(x2),abs(y2)) <= 2.0:
        return [max(0.0,min(1.0,x1)),max(0.0,min(1.0,y1)),max(0.0,min(1.0,x2)),max(0.0,min(1.0,y2))]
    # Match utils.datasets.read_image: Resize(256) preserving aspect ratio, then CenterCrop(224).
    scale=256.0/min(float(w),float(h))
    rw,rh=float(w)*scale,float(h)*scale
    left=(rw-224.0)/2.0; top=(rh-224.0)/2.0
    xx1=(x1*scale-left)/224.0; yy1=(y1*scale-top)/224.0
    xx2=(x2*scale-left)/224.0; yy2=(y2*scale-top)/224.0
    lo_x,hi_x=sorted([xx1,xx2]); lo_y,hi_y=sorted([yy1,yy2])
    lo_x=max(0.0,min(1.0,lo_x)); hi_x=max(0.0,min(1.0,hi_x))
    lo_y=max(0.0,min(1.0,lo_y)); hi_y=max(0.0,min(1.0,hi_y))
    if hi_x<=lo_x or hi_y<=lo_y: return None
    return [lo_x,lo_y,hi_x,hi_y]

def rect_iou(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1=max(ax1,bx1); iy1=max(ay1,by1); ix2=min(ax2,bx2); iy2=min(ay2,by2)
    inter=max(0.0,ix2-ix1)*max(0.0,iy2-iy1)
    aa=max(0.0,ax2-ax1)*max(0.0,ay2-ay1); bb=max(0.0,bx2-bx1)*max(0.0,by2-by1)
    return float(inter/(aa+bb-inter+1e-12))

def top_patch_rect(trow, frac, centers):
    n=len(trow); k=max(1,int(math.ceil(n*frac)))
    idx=np.argsort(np.asarray(trow))[-k:]
    xs=[centers[i][0] for i in idx]; ys=[centers[i][1] for i in idx]
    # Convert selected patch centers to an approximate covering rectangle in normalized 14x14 patch space.
    g=int(round(math.sqrt(n))); half=0.5/g
    return [max(0.0,min(xs)-half), max(0.0,min(ys)-half), min(1.0,max(xs)+half), min(1.0,max(ys)+half)]

def point_in_box(pt,b):
    x,y=pt; x1,y1,x2,y2=b; lo_x,hi_x=sorted([x1,x2]); lo_y,hi_y=sorted([y1,y2]); return lo_x<=x<=hi_x and lo_y<=y<=hi_y

def entropy(p):
    p=np.asarray(p,dtype=float); s=p.sum()
    if s<=0: return 0.0
    q=p/s; q=q[q>0]
    return float(-(q*np.log(q)).sum())

def main():
    cli=parse_args(); RES.mkdir(parents=True,exist_ok=True)
    pg=load_pgmner(cli); labels=pg.get_labels(pg.args.labels); id2label={i:l for i,l in enumerate(labels)}
    config=pg.AutoConfig.from_pretrained(pg.args.config_name if pg.args.config_name else pg.args.model_name_or_path,id2label={str(i):l for i,l in enumerate(labels)},label2id={l:i for i,l in enumerate(labels)},cache_dir=None)
    tokenizer=pg.RobertaTokenizer.from_pretrained(pg.args.tokenizer_name if pg.args.tokenizer_name else pg.args.model_name_or_path)
    model=pg.BertSpanNerBN(config,args=pg.args,num_labels=len(labels))
    state=torch.load(cli.checkpoint,map_location=pg.args.device)
    if isinstance(state,dict) and 'state_dict' in state: state=state['state_dict']
    load_mode=load_with_remap(model,state); model.to(pg.args.device); model.eval(); model.set_partial_ot_warmup(False)
    examples=pg.load_examples(pg.args.data_dir,mode=cli.split,tokenizer=tokenizer,visual_suffix_map=pg.args.visual_suffix_map,aux_knowledge_map=pg.resolve_aux_knowledge_map(pg.args,cli.split),vllm_desc_map=pg.resolve_vllm_desc_map(pg.args,cli.split),max_seq_len=pg.args.max_seq_len,max_aux_words=int(getattr(pg.args,'max_aux_words',12)),aux_dropout=0.0,desc_max_words=int(getattr(pg.args,'desc_max_words',60)),vllm_desc_dropout=0.0,desc_sep_token=str(getattr(pg.args,'desc_sep_token','</s>')))
    if cli.limit_samples: examples=examples[:cli.limit_samples]
    from utils.datasets import SpanNerDataset,collate_fn
    ds=SpanNerDataset(examples,args=pg.args,tokenizer=tokenizer,labels=labels,dev=True)
    dl=DataLoader(ds,sampler=SequentialSampler(ds),batch_size=cli.batch_size,collate_fn=collate_fn)
    side=read_sidecar(); rows=[]; sig=[]; sample_idx=0; rng=random.Random(42)
    with torch.no_grad():
        for bi,batch_cpu in enumerate(dl):
            batch={k:v.to(pg.args.device) for k,v in batch_cpu.items()}
            spans_rep,txt_rep_g,attn_last=model.span_encoding(**batch)
            if getattr(model,'use_pp_ib',False):
                _,_,_,_=model.dist_t(txt_rep_g); text_gate=model.dist_t.last_probe.get('gate')
            img_g,img_l=model.image_model(batch['image'])
            if getattr(model,'use_pp_ib',False):
                _,_,_,_=model.dist_v(img_g); visual_gate=model.dist_v.last_probe.get('gate')
            out=model.semantic_ot_aligner(spans_rep,img_l,mask_t=batch['span_masks'].float(),mask_v=None,attn_t=attn_last,span_token_idxes=batch.get('span_token_idxes'),patch_pos=None)
            text_proj,visual_proj,align_text,align_visual,T,loss_ot,comps=out
            probs=model.softmax(model.span_classifier(model.ln_text(model.fusion_text(torch.cat([text_proj,align_text],dim=-1))+align_text)))
            all_centers=patch_centers(196)
            for k,ex in enumerate(examples[sample_idx:sample_idx+batch['input_ids'].shape[0]]):
                meta=side.get((cli.split,str(getattr(ex,'guid',None)))) or {}
                ents=meta.get('entities') or []
                spans=batch['span_word_idxes'][k].detach().cpu().numpy(); masks=batch['span_masks'][k].detach().cpu().numpy(); gold=batch['span_labels'][k].detach().cpu().numpy(); pred=probs[k].argmax(-1).detach().cpu().numpy()
                for ent in ents:
                    if not ent.get('matched_in_text'): continue
                    s=int(ent['token_start']); e=int(ent['token_end']); typ=ent['type']; lab_id=labels.index('B-'+typ) if 'B-'+typ in labels else labels.index(typ) if typ in labels else -1
                    span_i=None
                    for ii,(a,b) in enumerate(spans):
                        if masks[ii]>0 and int(a)==s and int(b)==e:
                            span_i=ii; break
                    if span_i is None: continue
                    trow=T[k,span_i].detach().cpu().float().numpy(); total=float(trow.sum()); dust=max(0.0,1.0-total)
                    # CLIP ViT-B/16 local output has a class token followed by 14x14 spatial patch tokens.
                    spatial=trow[1:] if len(trow)==197 else trow
                    spatial_total=float(spatial.sum())
                    centers=all_centers if len(spatial)==196 else patch_centers(len(spatial))
                    top=int(spatial.argmax()) if len(spatial) else -1
                    bbox=bbox_norm(ent.get('bbox'),meta.get('image_path','')) if ent.get('bbox') else None
                    inside_mass=''; box_area_ratio=''; mass_lift=''; pointing_hit=''; iou_top30=''
                    iou_top10=''; iou_top20=''; iou_top30=''
                    if bbox:
                        inside=[j for j,c in enumerate(centers) if point_in_box(c,bbox)]
                        inside_mass=float(spatial[inside].sum()/(spatial_total+1e-12)) if inside else 0.0
                        x1,y1,x2,y2=bbox; box_area_ratio=abs((x2-x1)*(y2-y1)); mass_lift=float(inside_mass/(box_area_ratio+1e-12))
                        pointing_hit=int(point_in_box(centers[top],bbox)) if top>=0 else 0
                        iou_top10=rect_iou(top_patch_rect(spatial,0.10,centers),bbox)
                        iou_top20=rect_iou(top_patch_rect(spatial,0.20,centers),bbox)
                        iou_top30=rect_iou(top_patch_rect(spatial,0.30,centers),bbox)
                    pred_ok=int(int(pred[span_i])==lab_id) if lab_id>=0 else 0
                    rows.append({'split':cli.split,'sample_id':getattr(ex,'guid',sample_idx+k),'image_id':ex.imgid,'mention':ent['mention'],'type':typ,'groundable':int(bool(bbox)),'bbox':json.dumps(bbox) if bbox else '', 'span_index':span_i,'span_start':s,'span_end':e,'mner_correct':pred_ok,'transport_mass':total,'dustbin_mass_est':dust,'max_transport':float(spatial.max()) if len(spatial) else 0.0,'transport_entropy':entropy(spatial),'norm_transport_entropy':entropy(spatial)/math.log(max(2,len(spatial))),'transport_mass_spatial':spatial_total,'top_patch':top,'pointing_hit':pointing_hit,'inside_mass':inside_mass,'box_area_ratio':box_area_ratio,'mass_lift':mass_lift,'iou_top10':iou_top10,'iou_top20':iou_top20,'iou_top30':iou_top30})
            sample_idx+=batch['input_ids'].shape[0]
            print(f'batch {bi+1}/{len(dl)} samples={sample_idx}',flush=True)
    write_csv(RES/f'grounding_entity_level_{cli.split}.csv',rows)
    gr=[r for r in rows if r['groundable']==1]
    rand_hits=[]
    for r in gr:
        b=json.loads(r['bbox']); rand_hits.append(int(point_in_box(rng.choice(patch_centers()),b)))
    summary={'load_mode':load_mode,'split':cli.split,'samples_processed':len(examples),'entities':len(rows),'groundable':len(gr),'ungroundable':len(rows)-len(gr),'pointing_accuracy':float(np.mean([r['pointing_hit'] for r in gr])) if gr else None,'random_pointing_baseline':float(np.mean(rand_hits)) if rand_hits else None,'mean_inside_mass':float(np.mean([float(r['inside_mass']) for r in gr])) if gr else None,'mean_box_area_ratio':float(np.mean([float(r['box_area_ratio']) for r in gr])) if gr else None,'mean_mass_lift':float(np.mean([float(r['mass_lift']) for r in gr])) if gr else None}
    write_json(RES/f'grounding_summary_{cli.split}.json',summary)
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
