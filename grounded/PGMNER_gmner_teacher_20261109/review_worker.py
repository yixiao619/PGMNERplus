import argparse,json,time,importlib.util
from pathlib import Path
import torch
from PIL import Image
from experiment import legacy,OUT,old
from e2egmner.data.collator import DEFAULT_SYSTEM_PROMPT
spec=importlib.util.spec_from_file_location('old_qwen',legacy.ROOT/'kais_extension/experiment14_gmner_integration/scripts/run_qwen_q2_cache.py')
qwen=importlib.util.module_from_spec(spec);spec.loader.exec_module(qwen)

def prompt(sample):
    candidates=[dict(mention=e['mention'],type=e['official_type'],confidence=round(e['entity_confidence'],4)) for e in sample['entities'] if e['entity_confidence']>=.9]
    return ('Perform grounded MNER by extracting entities from the text.\n'
            'For each entity, output `entity|type|[x1,y1,x2,y2]` if it is visible in the image; otherwise output `entity|type|None`.\n'
            'Separate multiple results with `;`, or output `None` if no entity is found.\n\n'
            'Use both textual evidence and visual evidence when available.\n\n'
            'Text: '+sample['text']+'\n\n'
            'Candidate suggestions from a separate entity extractor:\n'+json.dumps(candidates,ensure_ascii=False)+'\n'
            'These suggestions may be incorrect or incomplete. Verify their exact mention spans, entity types, and visual presence using the text and image. '
            'Correct or discard incorrect suggestions and include any named entities missed by the suggestions. '
            'Use exact mention text from the post. Entity types are PER, LOC, ORG, OTHER. '
            'Return the complete grounded MNER result, not a list of edits.')

@torch.no_grad()
def main():
    a=argparse.ArgumentParser();a.add_argument('--split',choices=['dev','test'],required=True);a.add_argument('--shard',type=int,required=True);args=a.parse_args()
    samples=legacy.read(legacy.R/f'pgmner_predicted_entities_{args.split}.jsonl')
    idx=range(len(samples)) if args.split=='dev' else old.test_indices()
    path=OUT/f'review_{args.split}_shard{args.shard}.jsonl'
    done={r['sample_index'] for r in legacy.read(path)} if path.exists() else set()
    idx=[i for i in idx if i%3==args.shard and i not in done]
    print(f'split={args.split} pending={len(idx)} cached={len(done)}',flush=True)
    if not idx:return
    torch.set_num_threads(4);model,processor=qwen.load_model('cuda:0')
    start=time.time()
    with path.open('a') as f:
        for j,i in enumerate(idx,1):
            sample=samples[i];user=prompt(sample)
            messages=[{'role':'system','content':DEFAULT_SYSTEM_PROMPT},{'role':'user','content':[{'type':'image'},{'type':'text','text':user}]}]
            with Image.open(sample['image_path']) as im:image=im.convert('RGB')
            text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            inputs=processor(text=[text],images=[image],padding=True,return_tensors='pt')
            inputs={k:v.to('cuda:0') if torch.is_tensor(v) else v for k,v in inputs.items()}
            t=time.time();generated=model.generate(**inputs,max_new_tokens=512,do_sample=False,num_beams=1,repetition_penalty=1.,no_repeat_ngram_size=0)
            output=generated[0,inputs['input_ids'].shape[-1]:]
            raw=processor.batch_decode([output],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
            row=dict(split=args.split,sample_index=i,image_id=sample['image_id'],prompt=user,raw_generation=raw,seconds=time.time()-t,generated_tokens=len(output),hit_token_limit=len(output)>=512)
            f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
            if j%10==0 or j==len(idx):print(f'{j}/{len(idx)} elapsed={time.time()-start:.1f}s mean={(time.time()-start)/j:.2f}s',flush=True)

if __name__=='__main__':main()
