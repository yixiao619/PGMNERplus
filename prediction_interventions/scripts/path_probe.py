"""Entity-conditioned gradient and perturbation audit of the retained positive round."""
import os,sys,json,argparse,importlib.util
from pathlib import Path
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=64);p.add_argument('--run',required=True);p.add_argument('--output',required=True);p.add_argument('--gpu',type=int,required=True);cli=p.parse_args()
    os.environ['PGMNER_EXPERIMENT']=(Path(cli.run)/'experiment.json').read_text()
    run, dest = Path(cli.run).resolve(), Path(cli.output).resolve()
    dest.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    cmd = json.loads((run/'command.json').read_text())
    for key, value in {'--gpu_id': str(cli.gpu), '--do_train': 'false', '--do_eval': 'false',
                       '--do_predict': 'false', '--output_dir': str(dest)}.items():
        cmd[cmd.index(key)+1] = value
    sys.argv = cmd[1:]
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location('pgm_audit', ROOT/'main-mner-robert.py')
    pg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pg)
    labels = pg.get_labels(pg.args.labels)
    cfg = pg.AutoConfig.from_pretrained(pg.args.model_name_or_path)
    tok = pg.RobertaTokenizer.from_pretrained(pg.args.tokenizer_name)
    model = pg.BertSpanNerBN(cfg, args=pg.args, num_labels=len(labels))
    state = torch.load(run/'best_model.pth', map_location='cpu', weights_only=False)
    model.load_state_dict(state.get('state_dict', state), strict=True)
    del state
    model.to(pg.args.device).eval()
    model.set_partial_ot_warmup(False)
    aligner = model.semantic_ot_aligner
    examples = pg.load_examples(pg.args.data_dir, mode='dev', tokenizer=tok,
        max_seq_len=pg.args.max_seq_len, aux_dropout=0., vllm_desc_dropout=0.)
    from utils.datasets import SpanNerDataset, collate_fn
    from torch.utils.data import DataLoader
    dataset = SpanNerDataset(examples, args=pg.args, tokenizer=tok, labels=labels, dev=True)
    current=['native'];common_capture={};records=[];offset=0
    def mu_hook(m,args,out):
        if out.shape[1]==1:pass
        else:common_capture['patch_mu']=out
    aligner.ib_v.mu.register_forward_hook(mu_hook)
    def gate_hook(m,args,out):
        if current[0]!='shuffle_gate':return out
        x,mask=args[:2];g=m.current_gate.clone();gen=torch.Generator(device=x.device).manual_seed(42+offset)
        for i in range(len(g)):
            ix=mask[i].bool().nonzero().flatten();g[i,ix]=g[i,ix[torch.randperm(len(ix),device=x.device,generator=gen)]]
        m.current_gate=g;m.last_gate=g.detach();return g*m.mu(x)*mask[...,None],out[1],out[2]
    aligner.ib_v.register_forward_hook(gate_hook)
    def plan_hook(m,args,out):
        p,l=out
        if current[0]=='independent':p=p.sum(2,keepdim=True)*p.sum(1,keepdim=True)/p.sum((1,2),keepdim=True).clamp_min(1e-12)
        return p,l
    aligner.solver.register_forward_hook(plan_hook)
    for cpu in DataLoader(dataset,batch_size=2,collate_fn=collate_fn,shuffle=False):
        batch={k:v.to(pg.args.device) for k,v in cpu.items()};mask=batch['span_masks'].bool();gold=batch['span_labels'];bs=len(mask)
        with torch.no_grad():text,_,attention=model.span_encoding(**batch);_,visual=model.image_model(batch['image'])
        baseline=None
        for mode in ['native','shuffle_gate','independent','zero_local','zero_common']:
            current[0]=mode;model.zero_grad(set_to_none=True)
            out=aligner(text,visual,mask_t=mask.float(),attn_t=attention);t,local,at,_,plan=out[:5]
            if mode=='zero_local':
                a=mask.float()/mask.sum(1,keepdim=True);at=at-float(aligner.local_fusion_coefficient)*torch.bmm(plan,local)/a.clamp_min(1e-12)[...,None]
            if mode=='zero_common':at=at-float(aligner.explore.get('common_coefficient',.1))*aligner.common_features*mask[...,None]*visual.abs().sum((1,2)).gt(0)[:,None,None]
            logits=model.span_classifier(model.ln_text(model.fusion_text(torch.cat([t,at],-1))+at))
            probs=logits.softmax(-1)
            if mode=='native':
                baseline=probs.detach()
                if offset==0:
                    with torch.no_grad():ref,_=model({**batch,'span_labels':None})
                    assert torch.allclose(probs,ref[0],atol=2e-6)
                    # Recompute to retain the native graph after reference hooks.
                    out=aligner(text,visual,mask_t=mask.float(),attn_t=attention);t,local,at,_,plan=out[:5]
                    logits=model.span_classifier(model.ln_text(model.fusion_text(torch.cat([t,at],-1))+at));probs=logits.softmax(-1)
            for label,selected in [('entity',mask&(gold!=0)),('nonentity',mask&(gold==0))]:
                if not selected.any():continue
                loss=F.cross_entropy(logits[selected],gold[selected]);n=int(selected.sum())
                row={'offset':offset,'mode':mode,'group':label,'n':n,'ce_sum':float(loss.detach())*n,'correct':int((logits.argmax(-1)[selected]==gold[selected]).sum()),'changed':int((logits.argmax(-1)[selected]!=baseline.argmax(-1)[selected]).sum()),'prob_l1_sum':float((probs.detach()-baseline).abs().sum(-1)[selected].sum())}
                if mode=='native':
                    tensors={'gate_weight':aligner.ib_v.gate.weight,'gate':aligner.ib_v.current_gate,'plan':plan,'local_values':local,'aligned':at,'common':aligner.common_features}
                    grads=torch.autograd.grad(loss,list(tensors.values()),retain_graph=True,allow_unused=True)
                    row['gradient']={k:None if g is None else {'rms':float(g.square().mean().sqrt()),'input_times_grad_l1':float((g*x).abs().sum())} for (k,x),g in zip(tensors.items(),grads)}
                if mode=='native':
                    taskg=torch.autograd.grad(loss,aligner.ib_v.gate.weight,retain_graph=True)[0]
                    regg=torch.autograd.grad(out[6]['repair_polar']*float(pg.args.beta_polar),aligner.ib_v.gate.weight,retain_graph=True)[0]
                    row['gate_gradient']={'task_norm':float(taskg.norm()),'regularizer_norm':float(regg.norm()),'cosine':float(F.cosine_similarity(taskg.flatten(),regg.flatten(),dim=0))}
                    row['branch_norms']={'text':float(t.detach()[selected].norm(dim=-1).mean()),'local':float((float(aligner.local_fusion_coefficient)*aligner.local_aux_features.detach())[selected].norm(dim=-1).mean()),'common':float((float(aligner.explore.get('common_coefficient',.1))*aligner.common_features.detach()).norm(dim=-1).mean())}
                if mode=='native':
                    gates=aligner.ib_v.current_gate
                    dg=torch.autograd.grad(loss,gates,retain_graph=True)[0]
                    norms=common_capture['patch_mu'].detach().norm(dim=-1)
                    comparisons=[]
                    for j in range(bs):
                        if not selected[j].any() or not visual[j].abs().sum():continue
                        gg=gates[j,1:,0].detach().cpu().numpy();nn=norms[j,1:].cpu().numpy()
                        utility=(-dg[j,1:,0]*gates[j,1:,0]).detach().cpu().numpy()
                        def corr(a,b):
                            z=float(spearmanr(a,b).statistic)
                            return z if z==z else None
                        comparisons.append({'gate_norm_spearman':corr(gg,nn),'gate_utility_spearman':corr(gg,utility)})
                    row['region_ranking']=comparisons
                records.append(row)
        offset+=bs;(dest/'progress.json').write_text(json.dumps({'processed':offset,'total':len(dataset)}));print(offset,flush=True)
        if cli.limit and offset>=cli.limit:break
    summary={}
    for mode in ['native','shuffle_gate','independent','zero_local','zero_common']:
        summary[mode]={}
        for group in ['entity','nonentity']:
            rr=[r for r in records if r['mode']==mode and r['group']==group];n=sum(r['n'] for r in rr)
            summary[mode][group]={'n':n,'mean_ce':sum(r['ce_sum'] for r in rr)/n,'accuracy':sum(r['correct'] for r in rr)/n,'changed':sum(r['changed'] for r in rr),'mean_probability_l1':sum(r['prob_l1_sum'] for r in rr)/n}
    (dest/'records.json').write_text(json.dumps(records,indent=2));(dest/'results.json').write_text(json.dumps({'complete':True,'processed':offset,'diagnostic_subset':bool(cli.limit),'summary':summary,'note':'Gold labels used only for diagnostic stratification/gradients. Candidate accuracy is not entity F1. Gradient magnitudes are sensitivity evidence, not proof of retraining benefit.'},indent=2));print(summary)
if __name__=='__main__':main()
