import argparse, importlib.util, json, time
from pathlib import Path
import torch
from PIL import Image
from experiment import OUT, legacy, test_indices

spec=importlib.util.spec_from_file_location('qwen_original',legacy.ROOT/'kais_extension/experiment14_gmner_integration/scripts/run_qwen_q2_cache.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
CONTEXT_PROMPT=('Given this image and its accompanying post, locate the visual region corresponding to the specified entity. '
                'Use the post to disambiguate the entity, but return a box only when the entity is visually present in the image.\n\n'
                'Post: {text}\n\nEntity: "{entity}"\nType: "{etype}"\n\nReturn only:\n[x1,y1,x2,y2]\n\n'
                'If the entity is not visually present, return:\nNone')

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['dev','test'],required=True);p.add_argument('--shard',type=int,required=True);p.add_argument('--shards',type=int,default=2);p.add_argument('--limit',type=int,default=0);a=p.parse_args()
    samples=legacy.read(legacy.R/f'pgmner_predicted_entities_{a.split}.jsonl')
    idx=range(len(samples)) if a.split=='dev' else test_indices()
    jobs={}
    for i in idx:
        if i%a.shards != a.shard: continue
        for e in samples[i]['entities']:jobs[old.entity_key(samples[i]['sample_index'],e)]=(samples[i],e)
    path=OUT/f'context_{a.split}_shard{a.shard}.jsonl'
    done={x['key'] for x in legacy.read(path)} if path.exists() else set()
    pending=[(k,s,e) for k,(s,e) in jobs.items() if k not in done]
    if a.limit:pending=pending[:a.limit]
    print(f'{a.split} shard={a.shard} pending={len(pending)} cached={len(done)}',flush=True)
    if not pending:return
    torch.set_num_threads(4)
    model,processor=old.load_model('cuda:0')
    start=time.time()
    with path.open('a') as f:
        for j,(key,s,e) in enumerate(pending,1):
            prompt=CONTEXT_PROMPT.format(text=s['text'],entity=e['mention'],etype=e['official_type'])
            with Image.open(s['image_path']) as im:image=im.convert('RGB')
            messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':prompt}]}]
            text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            inputs=processor(text=[text],images=[image],padding=True,return_tensors='pt')
            inputs={k:v.to('cuda:0') if torch.is_tensor(v) else v for k,v in inputs.items()}
            t=time.time()
            generated=model.generate(**inputs,max_new_tokens=64,do_sample=False,num_beams=1,repetition_penalty=1.,no_repeat_ngram_size=0)
            raw=processor.batch_decode([generated[0,inputs['input_ids'].shape[-1]:]],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
            parsed=old.parse_prediction(raw)
            row=dict(key=key,split=a.split,sample_index=s['sample_index'],mention=e['mention'],official_type=e['official_type'],prompt=prompt,raw_generation=raw,pred_boxes=parsed['boxes'],parse_valid=parsed['valid'],parsed_kind=parsed['kind'],seconds=time.time()-t)
            f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
            if j%25==0 or j==len(pending):print(f'generated={j}/{len(pending)} elapsed={time.time()-start:.1f}s mean={(time.time()-start)/j:.2f}s',flush=True)

if __name__=='__main__':main()
