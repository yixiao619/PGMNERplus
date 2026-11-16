#!/usr/bin/env python3
"""DEV-only short training for Experiment 24 Routes A and C."""
from __future__ import annotations

import argparse, csv, importlib.util, json, math, os, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup

ROOT = Path("/home/jyx/yixiao/MINER-My")
EXP = ROOT / "kais_extension/experiment24_literature_guided"
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def args_cli():
    p = argparse.ArgumentParser()
    p.add_argument("--route", choices=["A", "C"], required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_main(out, seed):
    argv = ["main-mner-robert.py", "--data_dir", "data/twitter2017", "--output_dir", str(out),
            "--model_name_or_path", "pretrain/roberta-large", "--tokenizer_name", "pretrain/roberta-large",
            "--vit_name", "pretrain/clip-vit-base-patch16", "--use_hf_clip_vision", "true",
            "--auto_img_dims", "true", "--gpu_id", "0", "--batch_size", "16", "--max_seq_len", "128",
            "--max_span_len", "4", "--seed", str(seed), "--use_pp_ib", "--beta_polar", "0.01",
            "--delta_minus", "0.2", "--delta_plus", "0.8", "--use_got", "--got_mode", "partial_sinkhorn",
            "--ot_alpha", "0.05", "--entropy_eps", "0.1", "--ot_sinkhorn_iter", "30",
            "--partial_mass_init", "0.5", "--dust_cost", "1.0", "--ot_warmup_epochs", "0",
            "--do_train", "false", "--do_eval", "false", "--do_predict", "false", "--eval_split_for_best", "dev"]
    spec = importlib.util.spec_from_file_location("e24_main", ROOT / "main-mner-robert.py")
    m = importlib.util.module_from_spec(spec); old = sys.argv; sys.argv = argv
    try: spec.loader.exec_module(m)
    finally: sys.argv = old
    m.args.labels = str(ROOT / "kais_extension/configs/labels_twitter2017.txt")
    m.args.output_dir = str(out)
    m.args.e24_aux_weight = 0.10
    m.args.e24_polar_weight = 0.01
    return m


def checkpoint():
    p = Path("/data/jyx/MINER-My/outputs/pp_ib_partial_ot_grid_dc10_m05_seed42/best_model.pth")
    s = torch.load(p, map_location="cpu", weights_only=True)
    old, new = "image_model._hf.encoder.vision_model.", "image_model._hf.encoder."
    return {new + k[len(old):] if k.startswith(old) else k: v for k, v in s.items()}


def setup(route, out, seed):
    m = load_main(out, seed); m.set_seed(m.args)
    labels = m.get_labels(m.args.labels)
    cfg = m.AutoConfig.from_pretrained(m.args.model_name_or_path)
    tok = m.AutoTokenizer.from_pretrained(m.args.tokenizer_name)
    torch.manual_seed(2400)
    if route == "A":
        from kais_extension.experiment24_literature_guided.models.e24_model import TaskGuidedOTPGMNER as Cls
    else:
        from models.bn_bert_ner_OT_robert import BertSpanNerBN as Cls
    model = Cls(cfg, args=m.args, num_labels=len(labels))
    missing, unexpected = model.load_state_dict(checkpoint(), strict=False)
    allowed = route == "A" and all(k.startswith("e24_visual_classifier.") for k in missing)
    if unexpected or (missing and not allowed):
        raise RuntimeError(f"checkpoint mismatch missing={missing} unexpected={unexpected}")
    model.to(m.args.device); model.set_partial_ot_warmup(False)
    return m, model, tok, labels, missing


def gnorm(loss, params):
    ps = [p for p in params if p.requires_grad]
    gs = torch.autograd.grad(loss, ps, retain_graph=True, allow_unused=True)
    gs = [g for g in gs if g is not None]
    return float(torch.sqrt(sum((g.float() ** 2).sum() for g in gs))) if gs else 0.0


def counterfactual(model, batch, clean_prob):
    if batch["image"].size(0) < 2:
        return clean_prob.sum() * 0.0, clean_prob.detach(), 0
    wrong = dict(batch); wrong["image"] = batch["image"].roll(1, 0)
    wrong_prob = model(wrong)[0][0]
    mask = batch["span_masks"].bool() & batch["span_labels"].ne(0)
    cp = clean_prob.gather(-1, batch["span_labels"].unsqueeze(-1)).squeeze(-1).clamp_min(1e-8)
    wp = wrong_prob.gather(-1, batch["span_labels"].unsqueeze(-1)).squeeze(-1).clamp_min(1e-8)
    loss = F.relu(0.05 - (cp.log() - wp.log()))[mask].mean() if mask.any() else cp.sum() * 0.0
    return loss, wrong_prob, int(mask.sum())


def scan(model, loader, device, random_images=False):
    model.eval(); pred, gold, stats = [], [], []
    with torch.no_grad():
        for bc in loader:
            b = {k: v.to(device) for k, v in bc.items()}
            if random_images and b["image"].size(0) > 1:
                b["image"] = b["image"].roll(1, 0)
            prob = model(b)[0][0]; mask = b["span_masks"].bool()
            pred.extend(prob.argmax(-1)[mask].cpu().tolist()); gold.extend(b["span_labels"][mask].cpu().tolist())
            if hasattr(model, "last_e24_stats") and model.last_e24_stats: stats.append(model.last_e24_stats.copy())
    out = {"predictions": pred, "gold": gold}
    for k in (stats[0] if stats else {}): out[k] = float(np.mean([x[k] for x in stats]))
    return out


def transitions(a, b, gold):
    idx = [i for i, (x, y, g) in enumerate(zip(a, b, gold)) if x != 0 or y != 0 or g != 0]
    return {"changed": sum(a[i] != b[i] for i in idx),
            "correct_to_wrong": sum(a[i] == gold[i] and b[i] != gold[i] for i in idx),
            "wrong_to_correct": sum(a[i] != gold[i] and b[i] == gold[i] for i in idx)}


def write_csv(path, rows):
    keys = []
    for row in rows:
        for k in row:
            if k not in keys: keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def run(a):
    out = Path(a.run_dir); out.mkdir(parents=True, exist_ok=True)
    m, model, tok, labels, missing = setup(a.route, out, a.seed)
    from utils.datasets import SpanNerDataset, collate_fn
    train_examples = m.load_examples(m.args.data_dir, "train", tok, max_seq_len=128)
    if a.smoke: train_examples = train_examples[:8]
    tr = SpanNerDataset(train_examples, args=m.args, tokenizer=tok, labels=labels)
    dv = SpanNerDataset(m.load_examples(m.args.data_dir, "dev", tok, max_seq_len=128), args=m.args, tokenizer=tok, labels=labels, dev=True)
    dlv = DataLoader(dv, sampler=SequentialSampler(dv), batch_size=16, collate_fn=collate_fn)
    bc = next(iter(DataLoader(tr, batch_size=2, collate_fn=collate_fn)))
    b = {k: v.to(m.args.device) for k, v in bc.items()}; model.train()
    prob, losses = model(b); signal = losses["e24_extra"] if a.route == "A" else counterfactual(model, b, prob[0])[0]
    target_params = list(model.semantic_ot_aligner.parameters())
    grad = gnorm(signal, target_params)
    finite = bool(torch.isfinite(losses["loss"]).all() and torch.isfinite(signal).all())
    losses["loss"].backward()
    smoke = {"route": a.route, "finite": finite, "signal_gradient_to_gota": grad,
             "checkpoint_missing_keys": missing, "peak_gpu_memory_mb": torch.cuda.max_memory_allocated()/1024**2}
    (out / "smoke.json").write_text(json.dumps(smoke, indent=2) + "\n")
    if not finite or grad <= 0: raise RuntimeError(f"unsupported smoke {smoke}")
    if a.smoke:
        (out / "status.json").write_text(json.dumps({"status":"smoke_complete", **smoke}, indent=2)+"\n")
        return

    initial_metric = m.evaluate(m.args, model, tok, labels, mode="dev", prefix=f"E24_{a.route}_initial", only_test=False, epoch_num=0, split_tag="dev")[0]
    initial = scan(model, dlv, m.args.device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-5, weight_decay=1e-4)
    epochs, patience = 5, 2
    steps = math.ceil(len(tr)/16) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1*steps), steps)
    rows=[]; best=-1.; bad=0; t0=time.time(); torch.cuda.reset_peak_memory_stats()
    for ep in range(1, epochs+1):
        gen=torch.Generator().manual_seed(a.seed+ep)
        dlt=DataLoader(tr,sampler=RandomSampler(tr,generator=gen),batch_size=16,collate_fn=collate_fn)
        model.train(); sums={}; n=0; grads=[]; cf_spans=0
        for si, bc in enumerate(dlt):
            b={k:v.to(m.args.device) for k,v in bc.items()}; opt.zero_grad(set_to_none=True)
            prob, losses=model(b); loss=losses["loss"]
            if a.route == "C":
                cf, _, ncf=counterfactual(model,b,prob[0]); loss=loss+0.10*cf; losses["e24_counterfactual"]=cf; cf_spans+=ncf
                signal=cf
            else: signal=losses["e24_extra"]
            if not torch.isfinite(loss): raise FloatingPointError("NaN/Inf")
            if si in (0,50,100): grads.append(gnorm(signal,target_params))
            loss.backward(); torch.nn.utils.clip_grad_norm_(params,1.0); opt.step(); sched.step(); n+=1
            for k,v in losses.items():
                if torch.is_tensor(v) and v.numel()==1: sums[k]=sums.get(k,0.)+float(v.detach())
        metric=m.evaluate(m.args,model,tok,labels,mode="dev",prefix=f"E24_{a.route}_e{ep}",only_test=False,epoch_num=ep,split_tag="dev")[0]
        clean=scan(model,dlv,m.args.device); rnd=scan(model,dlv,m.args.device,True)
        from_initial=transitions(initial["predictions"],clean["predictions"],clean["gold"])
        visual=transitions(clean["predictions"],rnd["predictions"],clean["gold"])
        row={"epoch":ep,"dev_precision":metric["span_precision"],"dev_recall":metric["span_recall"],"dev_f1":metric["span_f1"],
             "gradient_norm":float(np.mean(grads)) if grads else 0.,"prediction_changes_from_initial":from_initial["changed"],
             "clean_random_changed":visual["changed"],"clean_random_c_to_w":visual["correct_to_wrong"],"clean_random_w_to_c":visual["wrong_to_correct"],
             "counterfactual_entity_spans":cf_spans,"peak_gpu_memory_mb":torch.cuda.max_memory_allocated()/1024**2,
             "elapsed_seconds":time.time()-t0,**{k:v for k,v in clean.items() if k not in ("predictions","gold")},
             **{f"train_{k}":v/n for k,v in sums.items()}}
        rows.append(row); write_csv(out/"epoch_metrics.csv",rows); print(json.dumps(row),flush=True)
        if row["dev_f1"]>best:
            best=row["dev_f1"];bad=0;torch.save(model.state_dict(),out/"best_model.pth");bestrow=row.copy()
        else: bad+=1
        if bad>=patience: break
    status={"status":"complete","route":a.route,"seed":a.seed,"dataset":"Twitter-2017 TRAIN/DEV","best_dev_f1":bestrow["dev_f1"],
            "best_dev_precision":bestrow["dev_precision"],"best_dev_recall":bestrow["dev_recall"],"best_epoch":bestrow["epoch"],
            "initial_dev_f1":initial_metric["span_f1"],"prediction_changes":bestrow["prediction_changes_from_initial"],
            "clean_random_changed":bestrow["clean_random_changed"],"clean_random_c_to_w":bestrow["clean_random_c_to_w"],
            "clean_random_w_to_c":bestrow["clean_random_w_to_c"],"gradient_norm":bestrow["gradient_norm"],
            "transport_entropy":bestrow.get("transport_entropy","N/A"),"transport_max":bestrow.get("transport_max","N/A"),
            "trainable_parameters":sum(p.numel() for p in params),"added_parameters":sum(p.numel() for p in model.route_parameters()) if a.route=="A" else 0,
            "gpu_time_seconds":time.time()-t0,"peak_gpu_memory_mb":max(x["peak_gpu_memory_mb"] for x in rows),"test_evaluated":False}
    (out/"status.json").write_text(json.dumps(status,indent=2)+"\n")


if __name__ == "__main__":
    a=args_cli()
    try: run(a)
    except Exception as e:
        out=Path(a.run_dir);out.mkdir(parents=True,exist_ok=True)
        (out/"status.json").write_text(json.dumps({"status":"failed","route":a.route,"error":repr(e),"traceback":traceback.format_exc(),"test_evaluated":False},indent=2)+"\n")
        raise
