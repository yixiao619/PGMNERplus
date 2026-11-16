import sys,json,time,argparse,importlib.util
from pathlib import Path
import torch
from PIL import Image
sys.path.insert(0,'/home/jyx/PGMNER_gmner_three_rounds_20261109')
import pipeline as prev
ROOT=Path(__file__).parent;OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location('qwen_loader',prev.legacy.ROOT/'kais_extension/experiment14_gmner_integration/scripts/run_qwen_q2_cache.py');q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
@torch.no_grad()
def main():
 a=argparse.ArgumentParser();a.add_argument('split');a.add_argument('shard',type=int);args=a.parse_args()
 jobs=[x for x in prev.legacy.read(prev.OUT/f'r1_{args.split}_jobs.jsonl') if x['sample_index']%3==args.shard]
 path=OUT/f'{args.split}_worker{args.shard}.jsonl';done={x['id'] for x in prev.legacy.read(path)} if path.exists() else set();jobs=[x for x in jobs if x['id'] not in done]
 print('pending',len(jobs),flush=True)
 if not jobs:return
 torch.set_num_threads(4);model,processor=q.load_model('cuda:0');tokens=[processor.tokenizer.encode(c,add_special_tokens=False) for c in 'AB'];assert all(len(t)==1 for t in tokens);ids=[t[0] for t in tokens];start=time.time()
 with path.open('a') as f:
  for n,j in enumerate(jobs,1):
   with Image.open(j['image_path']) as im:image=im.convert('RGB')
   original=j['prompt'];swapped=original.replace('A: Correct mention and type.\nB: Incorrect mention boundary, non-entity, or incorrect type.','A: Incorrect mention boundary, non-entity, or incorrect type.\nB: Correct mention and type.')
   assert swapped!=original
   scores={}
   for mode,prompt,visual,pos in [('visual_swapped',swapped,True,1),('text',original,False,0),('text_swapped',swapped,False,1)]:
    if not visual:prompt=prompt.replace('Do not reject a real text entity just because it is absent from the image.','Determine textual entity validity independently of visual presence.')
    messages=[{'role':'system','content':'You are a careful information extraction verifier. Select exactly one of the supplied option letters. Use the evidence, not the source model identity.'},{'role':'user','content':([{'type':'image'}] if visual else [])+[{'type':'text','text':prompt}]}]
    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    batch=processor(text=[text],**({'images':[image]} if visual else {}),padding=True,return_tensors='pt');batch={k:v.to('cuda:0') if torch.is_tensor(v) else v for k,v in batch.items()}
    logits=model(**batch,logits_to_keep=1).logits[0,-1,ids].float();scores[mode]=float(torch.softmax(logits,0)[pos])
   f.write(json.dumps(dict(id=j['id'],sample_index=j['sample_index'],scores=scores))+'\n');f.flush()
   if n%25==0 or n==len(jobs):print(f'{n}/{len(jobs)} mean_seconds={(time.time()-start)/n:.3f}',flush=True)
if __name__=='__main__':main()
