from __future__ import annotations
import torch
from torch import nn
from kais_extension.experiment20_overnight_ppib.models.exp20_model import BertSpanNerBNExp20,EntityConditionedResidual

class StableEntityResidual(EntityConditionedResidual):
    def __init__(self,*a,temperature=2.0,**kw):super().__init__(*a,**kw);self.temperature=float(temperature)
    def forward(self,zt,zv,spans):
        global_z=torch.cat([zt,zv],-1);evidence=self.evidence_out(self.act(self.evidence_in(global_z))).unsqueeze(1);expanded=global_z.unsqueeze(1).expand(-1,spans.size(1),-1)
        if self.gate_mode=="stable":gamma=torch.sigmoid(self.gate_out(self.act(self.gate_in(torch.cat([spans,expanded],-1))))/self.temperature)
        elif self.gate_mode=="fixed_0.5":gamma=torch.full((*spans.shape[:2],1),.5,device=spans.device,dtype=spans.dtype)
        else:return super().forward(zt,zv,spans)
        return self.alpha*gamma*evidence,gamma.squeeze(-1)

class StablePCPPModel(BertSpanNerBNExp20):
    def __init__(self,config,args=None,num_labels=None):
        super().__init__(config,args=args,num_labels=num_labels);self.exp20_adapter=StableEntityResidual(768,int(args.hidden_dim),int(args.exp20_bottleneck),float(args.exp20_alpha),str(args.exp20_gate_mode),temperature=float(args.factory_gate_temperature));self.factory_gate_variance_target=float(args.factory_gate_variance_target);self.factory_gate_reg_weight=float(args.factory_gate_reg_weight)
    def forward(self,ori_feas,only_test=False):
        outputs,losses=super().forward(ori_feas,only_test)
        if self.last_exp20_gate is not None and losses:
            mask=ori_feas["span_masks"].float();g=self.last_exp20_gate;count=mask.sum(1).clamp_min(1);mean=(g*mask).sum(1)/count;var=(((g-mean[:,None])**2)*mask).sum(1)/count;reg=torch.relu(g.new_tensor(self.factory_gate_variance_target)-var).mean()+.1*((mean-.5)**2).mean();losses["factory_gate_reg"]=self.factory_gate_reg_weight*reg;losses["loss"]=losses["loss"]+losses["factory_gate_reg"]
        return outputs,losses
