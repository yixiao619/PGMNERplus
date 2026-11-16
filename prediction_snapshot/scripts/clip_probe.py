"""Frozen native CLIP retrieval diagnostic; uses DEV text/images, no entity labels."""
import os,json,sys
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1'
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel,CLIPProcessor
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(4);torch.manual_seed(42)
rows=[];current=None;words=[]
for line in (ROOT/'data/twitter2017/dev.txt').read_text().splitlines()+['IMGID:END']:
 if line.startswith('IMGID:'):
  if current is not None:
   p=ROOT/'data/twitter2017_images'/f'{current}.jpg'
   try:
    with Image.open(p) as im:im.verify()
    rows.append((current,' '.join(words),p))
   except (OSError,ValueError):pass
  current=line.split(':',1)[1].strip();words=[]
 elif line.strip():words.append(line.split('\t')[0])
result={'n':len(rows),'scope':'DEV native CLIP retrieval; not NER or a general image-recognition benchmark','models':{}}
for variant in ['clip-vit-base-patch16','clip-vit-large-patch14']:
 path=ROOT/'pretrain'/variant
 model=CLIPModel.from_pretrained(path).cuda(0).eval();processor=CLIPProcessor.from_pretrained(path)
 ts=[];vs=[]
 with torch.inference_mode():
  for start in range(0,len(rows),16):
   batch=rows[start:start+16];ims=[]
   for _,_,p in batch:
    with Image.open(p) as im:ims.append(im.convert('RGB'))
   inputs=processor(text=[r[1] for r in batch],images=ims,return_tensors='pt',padding=True,truncation=True,max_length=77)
   out=model(**{k:v.cuda(0) for k,v in inputs.items()})
   ts.append(out.text_embeds.cpu());vs.append(out.image_embeds.cpu())
 t=F.normalize(torch.cat(ts),dim=-1);v=F.normalize(torch.cat(vs),dim=-1);sim=t@v.T
 ids=[r[0] for r in rows];correct=torch.tensor([[a==b for b in ids] for a in ids])
 top=sim.topk(5,dim=1).indices
 shifted=sim[torch.arange(len(rows)),(torch.arange(len(rows))+1)%len(rows)]
 positive=sim.diag()
 result['models'][variant]={'recall1':float(correct.gather(1,top[:,:1]).any(1).float().mean()),'recall5':float(correct.gather(1,top).any(1).float().mean()),'matched_cosine':float(positive.mean()),'shifted_cosine':float(shifted.mean()),'matched_beats_shifted_fraction':float((positive>shifted).float().mean()),'image_feature_std_mean':float(v.std(0).mean()),'unique_image_ids':len(set(ids))}
 print(variant,result['models'][variant],flush=True)
 del model;torch.cuda.empty_cache()
Path(sys.argv[1]).write_text(json.dumps(result,indent=2))
