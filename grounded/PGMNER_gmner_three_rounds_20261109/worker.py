import argparse,json,time,importlib.util
import torch
from PIL import Image
from pipeline import OUT,legacy,m
from e2egmner.evaluation.geometry import smart_resize
spec=importlib.util.spec_from_file_location('qwen_loader',legacy.ROOT/'kais_extension/experiment14_gmner_integration/scripts/run_qwen_q2_cache.py');q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)

@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('round',type=int);p.add_argument('split');p.add_argument('shard',type=int);p.add_argument('--limit',type=int,default=0);a=p.parse_args()
 jobs=[x for x in legacy.read(OUT/f'r{a.round}_{a.split}_jobs.jsonl') if x['sample_index']%3==a.shard]
 path=OUT/f'r{a.round}_{a.split}_worker{a.shard}.jsonl';done={x['id'] for x in legacy.read(path)} if path.exists() else set();jobs=[x for x in jobs if x['id'] not in done]
 if a.limit:jobs=jobs[:a.limit]
 print('pending',len(jobs),flush=True)
 if not jobs:return
 torch.set_num_threads(4);model,processor=q.load_model('cuda:0');start=time.time()
 with path.open('a') as f:
  for n,j in enumerate(jobs,1):
   with Image.open(j['image_path']) as im:image=im.convert('RGB')
   images=[image];prompt=j['prompt']
   if a.round==2:
    w,h=image.size;nh,nw=smart_resize(h,w,factor=28,min_pixels=256*28*28,max_pixels=1024*28*28)
    prompt+=f'\nResized full image width={nw}, height={nh}.'
    for boxes in j['options'][1:]:
     x1,y1,x2,y2=boxes[0];x1=max(0,min(w-1,int(x1*w/nw)));x2=max(x1+1,min(w,int(x2*w/nw)));y1=max(0,min(h-1,int(y1*h/nh)));y2=max(y1+1,min(h,int(y2*h/nh)))
     crop=image.crop((x1,y1,x2,y2))
     # Square letterbox prevents extreme-aspect-ratio processor failures.
     side=max(crop.size);canvas=Image.new('RGB',(side,side),(127,127,127));canvas.paste(crop,((side-crop.width)//2,(side-crop.height)//2));images.append(canvas)
   messages=[{'role':'system','content':'You are a careful information extraction verifier. Select exactly one of the supplied option letters. Use the evidence, not the source model identity.'},{'role':'user','content':[{'type':'image'} for _ in images]+[{'type':'text','text':prompt}]}]
   text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
   batch=processor(text=[text],images=images,padding=True,return_tensors='pt');batch={k:v.to('cuda:0') if torch.is_tensor(v) else v for k,v in batch.items()}
   tokens=[processor.tokenizer.encode(c,add_special_tokens=False) for c in j['letters']];assert all(len(t)==1 for t in tokens)
   logits=model(**batch,logits_to_keep=1).logits[0,-1].float();values=logits[[x[0] for x in tokens]];probs=torch.softmax(values,0).cpu().tolist()
   row=dict(id=j['id'],sample_index=j['sample_index'],letters=j['letters'],probs=probs,choice=j['letters'][int(values.argmax())],option_logits=values.cpu().tolist(),images=len(images),note='Softmax over allowed single-token answers; not a calibrated confidence probability.')
   f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
   if n%25==0 or n==len(jobs):print(f'{n}/{len(jobs)} mean_seconds={(time.time()-start)/n:.3f}',flush=True)
if __name__=='__main__':main()
