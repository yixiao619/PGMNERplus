#!/usr/bin/env python3
from __future__ import annotations
import csv,json,math,sys,os,importlib.util
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader, SequentialSampler
from PIL import Image, ImageDraw
ROOT=Path('/home/jyx/yixiao/MINER-My')
EXP=ROOT/'kais_extension/experiment9_gmner_audit'; RES=EXP/'results'; VIS=EXP/'visualizations'

def read_csv(p):
    with p.open(newline='',encoding='utf-8') as fh: return list(csv.DictReader(fh))
def val(r,k):
    try: return float(r.get(k,''))
    except Exception: return None

def select_rows():
    rows=[r for r in read_csv(RES/'grounding_entity_level_test.csv') if r.get('groundable')=='1']
    det=rows[:2]
    high=sorted(rows,key=lambda r: val(r,'mass_lift') or -1,reverse=True)[:2]
    fail=sorted([r for r in rows if r.get('pointing_hit')=='0'],key=lambda r: val(r,'mass_lift') or 999)[:2]
    out=[]; seen=set()
    for tag,rs in [('deterministic',det),('high_lift',high),('failure',fail)]:
        for r in rs:
            key=(r['sample_id'],r['mention'],r['span_start'],r['span_end'])
            if key not in seen:
                seen.add(key); rr=dict(r); rr['case_type']=tag; out.append(rr)
    return out

def load_mod():
    os.chdir(ROOT); sys.path.insert(0,str(ROOT))
    sys.argv=['main-mner-robert.py','--data_dir',str(EXP/'cache/gmner_pgmner_format'),'--output_dir',str(RES/'viz_side_outputs'),'--model_name_or_path','pretrain/roberta-large','--tokenizer_name','pretrain/roberta-large','--vit_name','pretrain/clip-vit-base-patch16','--gpu_id','0','--do_train','false','--do_eval','false','--do_predict','false','--batch_size','8','--max_span_len','4','--max_seq_len','128','--seed','42','--use_pp_ib','--use_got','--got_mode','partial_sinkhorn','--ot_sinkhorn_iter','30','--partial_mass_init','0.5','--dust_cost','1.0','--entropy_eps','0.1','--ot_alpha','0.05','--auto_img_dims','true','--use_hf_clip_vision','true']
    spec=importlib.util.spec_from_file_location('pgmner_main_exp9_viz',ROOT/'main-mner-robert.py')
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); m.args.labels=str(EXP/'cache/gmner_pgmner_format/labels.txt')
    return m

def load_ckpt(model, path):
    state=torch.load(path,map_location='cpu')
    if isinstance(state,dict) and 'state_dict' in state: state=state['state_dict']
    old='image_model._hf.encoder.vision_model.'; new='image_model._hf.encoder.'; mapped={}
    for k,v in state.items(): mapped[new+k[len(old):] if k.startswith(old) else k]=v
    model.load_state_dict(mapped,strict=True)

def bbox_to_px(b,w,h):
    x1,y1,x2,y2=json.loads(b); return [x1*w,y1*h,x2*w,y2*h]
def patch_rect(j,w,h,n=196):
    g=int(round(math.sqrt(n))); x=j%g; y=j//g
    return [x*w/g,y*h/g,(x+1)*w/g,(y+1)*h/g]
def make_overlay(img_path,bbox_norm,heat,top,out,title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    img=Image.open(img_path).convert('RGB').resize((224,224))
    heat=np.asarray(heat,dtype=float)
    if heat.size==197:
        heat=heat[1:]
    heat=heat/(heat.max()+1e-12); heat=heat.reshape(14,14)
    fig,ax=plt.subplots(figsize=(4,4),dpi=180)
    ax.imshow(img); ax.imshow(heat,alpha=0.45,cmap='magma',extent=(0,224,224,0),interpolation='nearest')
    if bbox_norm:
        x1,y1,x2,y2=bbox_to_px(bbox_norm,224,224); ax.add_patch(plt.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor='lime',linewidth=2))
    tr=patch_rect(int(top),224,224); ax.add_patch(plt.Rectangle((tr[0],tr[1]),tr[2]-tr[0],tr[3]-tr[1],fill=False,edgecolor='cyan',linewidth=1.8))
    ax.set_title(title,fontsize=8); ax.axis('off'); fig.tight_layout(pad=0.05); fig.savefig(out); plt.close(fig)

def main():
    VIS.mkdir(parents=True,exist_ok=True)
    selected=select_rows(); targets={r['sample_id']:r for r in selected}
    m=load_mod(); labels=m.get_labels(m.args.labels)
    cfg=m.AutoConfig.from_pretrained(m.args.model_name_or_path,id2label={str(i):l for i,l in enumerate(labels)},label2id={l:i for i,l in enumerate(labels)})
    tok=m.RobertaTokenizer.from_pretrained(m.args.tokenizer_name)
    model=m.BertSpanNerBN(cfg,args=m.args,num_labels=len(labels)); load_ckpt(model,'/data/jyx/MINER-My/outputs/pp_ib_partial_ot_grid_dc10_m05_seed42/best_model.pth')
    model.to(m.args.device); model.eval(); model.set_partial_ot_warmup(False)
    examples=m.load_examples(m.args.data_dir,mode='test',tokenizer=tok,visual_suffix_map=m.args.visual_suffix_map,aux_knowledge_map=m.resolve_aux_knowledge_map(m.args,'test'),vllm_desc_map=m.resolve_vllm_desc_map(m.args,'test'),max_seq_len=m.args.max_seq_len,max_aux_words=int(getattr(m.args,'max_aux_words',12)),aux_dropout=0.0,desc_max_words=int(getattr(m.args,'desc_max_words',60)),vllm_desc_dropout=0.0,desc_sep_token=str(getattr(m.args,'desc_sep_token','</s>')))
    subset=[ex for ex in examples if ex.guid in targets]
    from utils.datasets import SpanNerDataset, collate_fn
    ds=SpanNerDataset(subset,args=m.args,tokenizer=tok,labels=labels,dev=True)
    dl=DataLoader(ds,sampler=SequentialSampler(ds),batch_size=4,collate_fn=collate_fn)
    outputs=[]; idx=0
    with torch.no_grad():
        for batch_cpu in dl:
            batch={k:v.to(m.args.device) for k,v in batch_cpu.items()}
            spans_rep,txt_rep_g,attn_last=model.span_encoding(**batch)
            if getattr(model,'use_pp_ib',False): model.dist_t(txt_rep_g)
            img_g,img_l=model.image_model(batch['image'])
            if getattr(model,'use_pp_ib',False): model.dist_v(img_g)
            out=model.semantic_ot_aligner(spans_rep,img_l,mask_t=batch['span_masks'].float(),mask_v=None,attn_t=attn_last,span_token_idxes=batch.get('span_token_idxes'),patch_pos=None)
            T=out[4]
            for k,ex in enumerate(subset[idx:idx+batch['input_ids'].shape[0]]):
                r=targets[ex.guid]; spans=batch['span_word_idxes'][k].detach().cpu().numpy(); masks=batch['span_masks'][k].detach().cpu().numpy()
                span_i=None
                for ii,(a,b) in enumerate(spans):
                    if masks[ii]>0 and int(a)==int(r['span_start']) and int(b)==int(r['span_end']): span_i=ii; break
                if span_i is None: continue
                heat=T[k,span_i].detach().cpu().float().numpy(); img_path=str(EXP/'cache/gmner_pgmner_format_images'/f"{ex.imgid}.jpg")
                outp=VIS/f"{r['case_type']}_{ex.guid}_{r['mention'].replace('/','_').replace(' ','_')}.png"
                make_overlay(img_path,r.get('bbox',''),heat,r.get('top_patch',0),outp,f"{r['case_type']} {r['mention']} {r['type']}")
                outputs.append({'case_type':r['case_type'],'sample_id':ex.guid,'mention':r['mention'],'type':r['type'],'mass_lift':r['mass_lift'],'pointing_hit':r['pointing_hit'],'file':str(outp)})
            idx+=batch['input_ids'].shape[0]
    (RES/'visualization_cases.json').write_text(json.dumps(outputs,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(outputs,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
