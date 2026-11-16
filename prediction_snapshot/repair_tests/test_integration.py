"""Actual BertSpanNerBN forward/loss; tiny random encoders, no downloads."""
import sys,json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch import nn
from transformers import RobertaConfig,RobertaModel
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from models.bn_bert_ner_OT_robert import BertSpanNerBN
class ImageEncoder(nn.Module):
 def __init__(self,args):super().__init__();self.proj=nn.Linear(6,8)
 def forward(self,x):
  h=self.proj(x);return h.mean(1),h

torch.set_num_threads(1);torch.manual_seed(7)
cfg=RobertaConfig(vocab_size=32,hidden_size=16,num_hidden_layers=1,num_attention_heads=2,intermediate_size=32,max_position_embeddings=32)
args=SimpleNamespace(use_got=True,repair_modules=True,repair_cost="gate_reliability",model_name_or_path='unused',img_local_dim=8,img_global_dim=8,txt_local_dim=16,hidden_dim=8,labels='data/twitter2017/labels.txt',ot_sinkhorn_iter=50,beta_polar=.01,ib_kl_weight=.001,eval_split_for_best='dev',use_got_cac=False)
with patch('models.bn_bert_ner_OT_robert.RobertaTokenizer.from_pretrained',return_value=None),patch('models.bn_bert_ner_OT_robert.RobertaModel.from_pretrained',side_effect=lambda *a,**k:RobertaModel(cfg)),patch('models.bn_bert_ner_OT_robert.ViTImageModel',ImageEncoder):
 m=BertSpanNerBN(cfg,args=args,num_labels=5)
batch=dict(input_ids=torch.randint(3,32,(2,8)),input_mask=torch.ones(2,8),span_token_idxes=torch.tensor([[[1,1],[2,3],[4,4]]]*2),span_lens=torch.tensor([[1,2,1]]*2),morph_idxes=torch.ones(2,3,2,dtype=torch.long),span_masks=torch.ones(2,3),span_labels=torch.tensor([[1,2,0],[3,4,0]]),image=torch.randn(2,5,6))
batch['span_weights']=torch.ones(2,3)
m.train();out,loss=m(batch)
assert torch.isfinite(loss['loss'])
loss['c'].backward()
for key in ['ib_t.gate.weight','ib_v.gate.weight','ib_t.logvar.weight','ib_v.logvar.weight','solver.s_logit']:
 p=dict(m.semantic_ot_aligner.named_parameters())[key]
 assert p.grad is not None and p.grad.norm()>0,key
m.zero_grad();_,loss=m(batch);loss['loss'].backward()
assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
m.eval();batch['span_labels']=None
with torch.no_grad():
 o1,_=m(batch);o2,_=m(batch)
assert torch.equal(o1[0],o2[0])
print('PASS actual model forward, classification-only gradients, combined loss backward, deterministic inference')
