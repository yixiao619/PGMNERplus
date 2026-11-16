#!/usr/bin/env python3
"""Run one DEV-only Experiment 20 configuration with resumable artifacts."""
from __future__ import annotations
import argparse, csv, importlib.util, json, math, os, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "kais_extension/experiment20_overnight_ppib"
MATRIX = EXP / "configs/experiment_matrix.json"
CKPT = Path("/data/jyx/MINER-My/outputs/pp_ib_partial_ot_grid_dc10_m05_seed42/best_model.pth")
BASELINE = 0.89462
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

BASE_ARGS = ["main-mner-robert.py", "--data_dir", "data/twitter2017", "--output_dir", str(EXP/"cache"),
 "--model_name_or_path", "pretrain/roberta-large", "--tokenizer_name", "pretrain/roberta-large",
 "--vit_name", "pretrain/clip-vit-base-patch16", "--use_hf_clip_vision", "true", "--auto_img_dims", "true",
 "--gpu_id", "0", "--epoch", "5", "--batch_size", "16", "--max_seq_len", "128", "--max_span_len", "4",
 "--seed", "42", "--lr", "3e-5", "--bert_lr", "1e-5", "--warmup_ratio", "0.1", "--patience", "2",
 "--use_pp_ib", "--beta_polar", "0.01", "--delta_minus", "0.2", "--delta_plus", "0.8",
 "--pp_ib_lambda_balance", "1.0", "--beta_polar_warmup_epochs", "3", "--use_got", "--got_mode",
 "partial_sinkhorn", "--ot_alpha", "0.05", "--entropy_eps", "0.1", "--ot_sinkhorn_iter", "30",
 "--partial_mass_init", "0.5", "--dust_cost", "1.0", "--ot_warmup_epochs", "3",
 "--do_train", "false", "--do_eval", "false", "--do_predict", "false"]

def args_cli():
    p=argparse.ArgumentParser(); p.add_argument("--id", required=True); p.add_argument("--fixed", action="store_true")
    p.add_argument("--smoke", action="store_true"); return p.parse_args()

def load_cfg(run_id, fixed):
    matrix=json.loads(MATRIX.read_text()); base=run_id.removesuffix("_fixed")
    cfg=next(x.copy() for x in matrix["experiments"] if x["id"]==base)
    if fixed: cfg["gate_mode"]="fixed_0.5"; cfg["id"]=base+"_fixed"
    return cfg

def main_module(out):
    spec=importlib.util.spec_from_file_location("exp20_main", ROOT/"main-mner-robert.py")
    mod=importlib.util.module_from_spec(spec); old=sys.argv; sys.argv=BASE_ARGS
    try: spec.loader.exec_module(mod)
    finally: sys.argv=old
    mod.args.labels=str((ROOT/"kais_extension/configs/labels_twitter2017.txt").resolve())
    mod.args.output_dir=str(out); return mod

def checkpoint_state():
    raw=torch.load(CKPT,map_location="cpu",weights_only=True); old="image_model._hf.encoder.vision_model."; new="image_model._hf.encoder."
    return {new+k[len(old):] if k.startswith(old) else k:v for k,v in raw.items()}

def write_csv(path, rows):
    if not rows: return
    fields=[]
    for row in rows:
        for k in row:
            if k not in fields: fields.append(k)
    tmp=path.with_suffix(path.suffix+".tmp")
    with tmp.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    tmp.replace(path)

def grad_norm(loss, params):
    grads=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
    valid=[g for g in grads if g is not None]
    return float(torch.sqrt(sum((g.float()**2).sum() for g in valid))) if valid else 0.0

def scan(model, loader, device, baseline_preds=None, seed=2042):
    model.eval(); preds=[]; gates=[]; sample_means=[]; span_vars=[]; residual=[]
    text_g=[]; visual_g=[]
    with torch.no_grad():
        for bi,bc in enumerate(loader):
            batch={k:v.to(device) for k,v in bc.items()}; torch.manual_seed(seed+bi)
            prob=model(batch)[0][0]; mask=batch["span_masks"].bool(); preds.extend(prob.argmax(-1)[mask].cpu().tolist())
            text=model.dist_t.last_probe["gate"].reshape(-1).detach().cpu().numpy(); visual=model.dist_v.last_probe["gate"].reshape(-1).detach().cpu().numpy()
            text_g.extend(text.tolist()); visual_g.extend(visual.tolist()); residual.append(model.last_exp20_summary["residual_ratio"])
            if model.last_exp20_gate is None:
                combined=(text+visual)/2; gates.extend(combined.tolist()); sample_means.extend(combined.tolist()); span_vars.extend([0.0]*len(combined))
            else:
                eg=model.last_exp20_gate.detach();
                for i in range(eg.size(0)):
                    vals=eg[i][mask[i]].cpu().numpy()
                    if len(vals): gates.extend(vals.tolist()); sample_means.append(float(vals.mean())); span_vars.append(float(vals.var()))
    x=np.asarray(gates,dtype=float); ent=-(x*np.log2(np.clip(x,1e-12,1))+(1-x)*np.log2(np.clip(1-x,1e-12,1)))
    changed=None if baseline_preds is None else int(np.sum(np.asarray(preds)!=np.asarray(baseline_preds)))
    return {"predictions":preds,"gate_mean":float(x.mean()),"gate_std":float(x.std()),
      "gate_entropy":float(ent.mean()),"gate_polarization_ratio":float(np.mean((x<=.1)|(x>=.9))),
      "gate_between_sample_variance":float(np.var(sample_means)),"gate_within_span_variance":float(np.mean(span_vars)),
      "text_gate_mean":float(np.mean(text_g)),"visual_gate_mean":float(np.mean(visual_g)),
      "residual_ratio":float(np.mean(residual)),"changed_valid_span_predictions":changed}

def configure(cfg, out):
    mod=main_module(out); mod.set_seed(mod.args)
    mod.args.exp20_kind=cfg["kind"]; mod.args.exp20_gate_mode=cfg["gate_mode"]
    mod.args.exp20_alpha=cfg["alpha"]; mod.args.exp20_bottleneck=128
    from kais_extension.experiment20_overnight_ppib.models.exp20_model import BertSpanNerBNExp20
    labels=mod.get_labels(mod.args.labels); config=mod.AutoConfig.from_pretrained(mod.args.model_name_or_path)
    tokenizer=mod.AutoTokenizer.from_pretrained(mod.args.tokenizer_name)
    torch.manual_seed(2000); model=BertSpanNerBNExp20(config,args=mod.args,num_labels=len(labels))
    missing,unexpected=model.load_state_dict(checkpoint_state(),strict=False); model.to(mod.args.device); model.set_partial_ot_warmup(False)
    allowed=all(k.startswith("exp20_adapter.") for k in missing) and not unexpected
    if not allowed: raise RuntimeError(f"checkpoint mismatch missing={missing} unexpected={unexpected}")
    return mod,model,tokenizer,labels,missing

def smoke(cfg):
    out=EXP/"checkpoints"/"smoke"/cfg["id"]; out.mkdir(parents=True,exist_ok=True)
    mod,model,tok,labels,missing=configure(cfg,out)
    from utils.datasets import SpanNerDataset,collate_fn
    ex=mod.load_examples(mod.args.data_dir,"train",tok,max_seq_len=mod.args.max_seq_len)[:16]
    ds=SpanNerDataset(ex,args=mod.args,tokenizer=tok,labels=labels); batch=next(iter(DataLoader(ds,batch_size=4,collate_fn=collate_fn)))
    batch={k:v.to(mod.args.device) for k,v in batch.items()}; model.eval()
    alpha=float(model.exp20_adapter.alpha); torch.manual_seed(9); p_on=model(batch)[0][0].detach()
    model.exp20_adapter.alpha.zero_(); torch.manual_seed(9); p_off=model(batch)[0][0].detach()
    model.exp20_adapter.alpha.fill_(alpha)
    max_diff=float((p_on-p_off).abs().max()); identical=bool(torch.equal(p_on.argmax(-1),p_off.argmax(-1)))
    for p in model.parameters(): p.requires_grad=False
    for p in model.exp20_adapter.parameters(): p.requires_grad=True
    for p in list(model.dist_t.parameters())+list(model.dist_v.parameters()): p.requires_grad=True
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    gate_after=[]
    for step in range(2):
        model.train(); opt.zero_grad(set_to_none=True); _,losses=model(batch); task=losses["c"]+losses["scl"]
        gn=grad_norm(task,model.task_gate_parameters()); task.backward(); opt.step(); gate_after.append(gn)
    finite=all(torch.isfinite(v).all().item() for v in losses.values() if torch.is_tensor(v))
    result={"id":cfg["id"],"initial_probability_max_difference":max_diff,"initial_predictions_identical":identical,
      "missing_only_new_module":all(k.startswith("exp20_adapter.") for k in missing),"gate_task_grad_steps":gate_after,
      "gate_connected_after_update":gate_after[-1]>0,"finite_losses":finite,"gpu":torch.cuda.get_device_name(0),
      "peak_gpu_memory_mb":torch.cuda.max_memory_allocated()/1024**2,"pass":identical and finite and gate_after[-1]>0}
    (EXP/"results"/f"smoke_{cfg['id']}.json").write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2))
    if not result["pass"]: raise SystemExit(2)

def train(cfg):
    run_id=cfg["id"]; run_dir=EXP/"results"/"runs"/run_id; ck_dir=EXP/"checkpoints"/run_id
    run_dir.mkdir(parents=True,exist_ok=True); ck_dir.mkdir(parents=True,exist_ok=True)
    status=run_dir/"status.json"
    if status.exists() and json.loads(status.read_text()).get("status")=="complete": print("already complete"); return
    status.write_text(json.dumps({"status":"running","id":run_id,"started":time.time(),"config":cfg},indent=2)+"\n")
    start=time.time(); mod,model,tok,labels,missing=configure(cfg,ck_dir)
    for p in model.parameters(): p.requires_grad=False
    for p in model.exp20_adapter.parameters(): p.requires_grad=True
    for p in list(model.dist_t.parameters())+list(model.dist_v.parameters()): p.requires_grad=True
    head=[]
    if cfg["unfreeze_head"]:
        for module in (model.fusion_text,model.ln_text,model.span_classifier):
            for p in module.parameters(): p.requires_grad=True; head.append(p)
    from utils.datasets import SpanNerDataset,collate_fn
    tr_ex=mod.load_examples(mod.args.data_dir,"train",tok,max_seq_len=mod.args.max_seq_len)
    dv_ex=mod.load_examples(mod.args.data_dir,"dev",tok,max_seq_len=mod.args.max_seq_len)
    tr=SpanNerDataset(tr_ex,args=mod.args,tokenizer=tok,labels=labels)
    dv=SpanNerDataset(dv_ex,args=mod.args,tokenizer=tok,labels=labels,dev=True)
    dv_loader=DataLoader(dv,sampler=SequentialSampler(dv),batch_size=16,collate_fn=collate_fn)
    baseline=scan(model,dv_loader,mod.args.device)["predictions"]
    groups=[{"params":list(model.exp20_adapter.parameters()),"lr":1e-4,"weight_decay":1e-4},
      {"params":list(model.dist_t.parameters())+list(model.dist_v.parameters()),"lr":1e-5,"weight_decay":1e-4}]
    if head: groups.append({"params":head,"lr":5e-6,"weight_decay":1e-4})
    steps=math.ceil(len(tr)/16)*5; opt=torch.optim.AdamW(groups); sched=get_linear_schedule_with_warmup(opt,int(.1*steps),steps)
    rows=[]; gates=[]; best=-1; best_ep=0; bad=0; exit_reason="max_epochs"
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,6):
        gen=torch.Generator().manual_seed(42+epoch); loader=DataLoader(tr,sampler=RandomSampler(tr,generator=gen),batch_size=16,collate_fn=collate_fn)
        model.train(); model.set_polar_train_epoch(3); sums={}; n=0; grad=[]; ratios=[]
        for si,bc in enumerate(loader):
            batch={k:v.to(mod.args.device) for k,v in bc.items()}; opt.zero_grad(set_to_none=True); _,losses=model(batch); loss=losses["loss"]
            if not torch.isfinite(loss): raise RuntimeError("nonfinite loss")
            if si and si%25==0: grad.append(grad_norm(losses["c"]+losses["scl"],model.task_gate_parameters()))
            loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0); opt.step(); sched.step()
            ratios.append(model.last_exp20_summary["residual_ratio"]); n+=1
            for k,v in losses.items():
                if torch.is_tensor(v) and v.numel()==1: sums[k]=sums.get(k,0.0)+float(v.detach())
        torch.manual_seed(4242+epoch); metrics=mod.evaluate(mod.args,model,tok,labels,mode="dev",prefix=f"{run_id}_e{epoch}",only_test=False,epoch_num=epoch,split_tag="dev")[0]
        stat=scan(model,dv_loader,mod.args.device,baseline,seed=3042+epoch)
        row={"id":run_id,"epoch":epoch,"dev_precision":metrics["span_precision"],"dev_recall":metrics["span_recall"],"dev_f1":metrics["span_f1"],
          "task_gate_grad_l2":float(np.mean(grad)) if grad else 0.0,"residual_ratio_train":float(np.mean(ratios)),
          "peak_gpu_memory_mb":torch.cuda.max_memory_allocated()/1024**2,"elapsed_seconds":time.time()-start,
          **{f"train_{k}":v/n for k,v in sums.items()},**{k:v for k,v in stat.items() if k!="predictions"}}
        rows.append(row); gates.append({k:row[k] for k in ["id","epoch","gate_mean","gate_std","gate_entropy","gate_polarization_ratio","gate_between_sample_variance","gate_within_span_variance","text_gate_mean","visual_gate_mean"]})
        write_csv(run_dir/"epoch_metrics.csv",rows); write_csv(run_dir/"gate_statistics.csv",gates); print(json.dumps(row),flush=True)
        if float(metrics["span_f1"])>best: best=float(metrics["span_f1"]);best_ep=epoch;bad=0;torch.save(model.state_dict(),ck_dir/"best_model.pth")
        else: bad+=1
        if epoch>=2 and best<BASELINE-.01: exit_reason="clear_dev_degradation"; break
        if bad>=2: exit_reason="early_stopping_patience_2"; break
    best_row=next(r for r in rows if r["epoch"]==best_ep)
    collapse="NONE"
    if best_row["gate_mean"]<.05 or best_row["gate_mean"]>.95: collapse="SINGLE_END"
    elif best_row["gate_std"]<1e-4: collapse="NEAR_CONSTANT"
    summary={"status":"complete","id":run_id,"config":cfg,"best_dev_f1":best,"delta_dev_vs_original":best-BASELINE,"best_epoch":best_ep,
      "epochs_completed":len(rows),"gate_mean":best_row["gate_mean"],"gate_std":best_row["gate_std"],"gate_entropy":best_row["gate_entropy"],
      "span_gate_variance":best_row["gate_within_span_variance"],"residual_ratio":best_row["residual_ratio"],
      "changed_valid_span_predictions":best_row["changed_valid_span_predictions"],"gate_collapse":collapse,
      "task_gate_grad_l2":best_row["task_gate_grad_l2"],"added_parameters":sum(p.numel() for p in model.exp20_adapter.parameters()),
      "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),"training_seconds":time.time()-start,
      "peak_gpu_memory_mb":max(r["peak_gpu_memory_mb"] for r in rows),"exit_reason":exit_reason,"test_evaluated":False,
      "checkpoint":str(ck_dir/"best_model.pth"),"missing_keys":missing}
    status.write_text(json.dumps(summary,indent=2)+"\n"); print(json.dumps(summary,indent=2))

def main():
    a=args_cli(); cfg=load_cfg(a.id,a.fixed)
    try: smoke(cfg) if a.smoke else train(cfg)
    except Exception as exc:
        if not a.smoke:
            d=EXP/"results"/"runs"/cfg["id"]; d.mkdir(parents=True,exist_ok=True)
            (d/"status.json").write_text(json.dumps({"status":"failed","id":cfg["id"],"config":cfg,"error":repr(exc),"traceback":traceback.format_exc()},indent=2)+"\n")
        raise
if __name__=="__main__": main()
