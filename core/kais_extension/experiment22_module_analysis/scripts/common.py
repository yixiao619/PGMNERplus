#!/usr/bin/env python3
from __future__ import annotations
import importlib.util, os, sys
from pathlib import Path
import torch

ROOT = Path("/home/jyx/yixiao/MINER-My")
EXP = ROOT / "kais_extension/experiment22_module_analysis"
RESULTS, FIGURES = EXP / "results", EXP / "figures"
CHECKPOINT = Path("/data/jyx/MINER-My/outputs/pp_ib_partial_ot_grid_dc10_m05_seed42/best_model.pth")

def load_main(data_dir, labels, gpu_id="0", batch_size=2):
    os.chdir(ROOT); sys.path.insert(0, str(ROOT)) if str(ROOT) not in sys.path else None
    argv=["main-mner-robert.py","--data_dir",str(data_dir),"--output_dir",str(EXP/"cache/runtime"),"--model_name_or_path","pretrain/roberta-large","--tokenizer_name","pretrain/roberta-large","--vit_name","pretrain/clip-vit-base-patch16","--use_hf_clip_vision","true","--auto_img_dims","true","--gpu_id",gpu_id,"--batch_size",str(batch_size),"--max_seq_len","128","--max_span_len","4","--seed","42","--use_pp_ib","--beta_polar","0.01","--delta_minus","0.2","--delta_plus","0.8","--pp_ib_lambda_balance","1.0","--beta_polar_warmup_epochs","3","--use_got","--got_mode","partial_sinkhorn","--ot_alpha","0.05","--entropy_eps","0.1","--ot_sinkhorn_iter","30","--partial_mass_init","0.5","--dust_cost","1.0","--do_train","false","--do_eval","false","--do_predict","false"]
    spec=importlib.util.spec_from_file_location("main_exp22",ROOT/"main-mner-robert.py"); mod=importlib.util.module_from_spec(spec); old=sys.argv; sys.argv=argv
    try: spec.loader.exec_module(mod)
    finally: sys.argv=old
    mod.args.labels=str(labels); return mod

def remap(state):
    old="image_model._hf.encoder.vision_model."; new="image_model._hf.encoder."
    return {new+k[len(old):] if k.startswith(old) else k:v for k,v in state.items()}

def build_model(mod):
    labels=mod.get_labels(mod.args.labels); cfg=mod.AutoConfig.from_pretrained(mod.args.model_name_or_path); tok=mod.AutoTokenizer.from_pretrained(mod.args.tokenizer_name)
    model=mod.BertSpanNerBN(cfg,args=mod.args,num_labels=len(labels)); state=torch.load(CHECKPOINT,map_location="cpu",weights_only=True)
    if isinstance(state,dict) and "state_dict" in state: state=state["state_dict"]
    model.load_state_dict(remap(state),strict=True); model.to(mod.args.device); model.set_partial_ot_warmup(False); model.set_polar_train_epoch(3)
    return model,tok,labels
