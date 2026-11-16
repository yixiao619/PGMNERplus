import os,json,sys
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1'
import torch
from PIL import Image
from torchvision import transforms as T
from transformers import CLIPVisionModel
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(4)
paths=[]
for line in (ROOT/'data/twitter2017/dev.txt').read_text().splitlines():
 if line.startswith('IMGID:'):
  p=ROOT/'data/twitter2017_images'/(line.split(':',1)[1].strip()+'.jpg')
  try:
   with Image.open(p) as im:im.verify()
   paths.append(p)
  except (OSError,ValueError):pass
pre=CLIPVisionModel.from_pretrained(ROOT/'pretrain/clip-vit-base-patch16').cuda(0).eval()
ft=CLIPVisionModel(pre.config).cuda(0).eval()
state=torch.load('/home/jyx/PGMNER_positive_seed42_20261103/selected/sharp_full_s42/best_model.pth',map_location='cpu',weights_only=False)
prefix='image_model._hf.encoder.';subset={k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)};ft.load_state_dict(subset,strict=True);del state,subset
transforms={'historical':T.Compose([T.Resize(256),T.CenterCrop(224),T.ToTensor(),T.Normalize((.485,.456,.406),(.229,.224,.225))]),'clip':T.Compose([T.Resize(224,interpolation=T.InterpolationMode.BICUBIC),T.CenterCrop(224),T.ToTensor(),T.Normalize((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))])}
results={}
with torch.inference_mode():
 for prep,transform in transforms.items():
  for name,model in [('pretrained',pre),('finetuned',ft)]:
   rows=[]
   for start in range(0,len(paths),16):
    ims=[]
    for p in paths[start:start+16]:
     with Image.open(p) as im:ims.append(transform(im.convert('RGB')))
    h=model(pixel_values=torch.stack(ims).cuda(0)).last_hidden_state[:,1:]
    mu=h.mean(1,keepdim=True)
    ratio=(h-mu).square().sum((1,2))/h.square().sum((1,2)).clamp_min(1e-12)
    rows.extend(ratio.cpu().tolist())
   results[prep+'_'+name]={'n':len(rows),'mean_residual_energy':sum(rows)/len(rows)}
   print(prep,name,results[prep+'_'+name],flush=True)
Path(sys.argv[1]).write_text(json.dumps(results,indent=2))
