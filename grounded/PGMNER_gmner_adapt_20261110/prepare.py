import sys,json,shutil,random,hashlib
from pathlib import Path
ROOT=Path(__file__).parent;SOURCE=Path('/data/hfy/VG-Think/E2E-GMNER');OUT=ROOT/'data';OUT.mkdir(exist_ok=True)
shutil.copytree(SOURCE/'src',ROOT/'src',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
for name in ['train.py','eval.py']:shutil.copy2(SOURCE/name,ROOT/name)
p=ROOT/'src/e2egmner/cli/train.py';s=p.read_text();s=s.replace('from peft import LoraConfig, get_peft_model','from peft import LoraConfig, get_peft_model, PeftModel');s=s.replace('parser.add_argument("--model_name",', 'parser.add_argument("--init_adapter", type=str, default=None)\n    parser.add_argument("--model_name",',1);s=s.replace('model = get_peft_model(model, lora_cfg)','model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True) if args.init_adapter else get_peft_model(model, lora_cfg)');p.write_text(s)
sys.path.insert(0,str(ROOT/'src'));from e2egmner.evaluation.parsing import parse_triples,extract_answer_text
sys.path.insert(0,'/home/jyx/PGMNER_gmner_teacher_20261109');import experiment as base
read=base.legacy.read;norm=base.legacy.norm
splits={k:read(SOURCE/f'data/sft/{k}_sft.jsonl') for k in ['train','dev','test']}
def user(row):return next(x for x in row['messages'] if x['role']=='user')
def txt(row):return user(row)['content'].split('Text:',1)[-1].strip()
exclude={norm(txt(r)) for k in ['dev','test'] for r in splits[k]};train=[r for r in splits['train'] if norm(txt(r)) not in exclude]
rng=random.Random(42)
def clone(r):return json.loads(json.dumps(r))
def hints(row,candidates):
 user(row)['content']+='\n\nCandidate suggestions (may be incomplete or incorrect): '+json.dumps(candidates,ensure_ascii=False)+'\nVerify and correct the candidate mention boundaries and types using the text and image. Include missed named entities. Return the complete grounded result, not a list of edits.'
def dump(name,rows):
 with (OUT/name).open('w') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
answer=[];correction=[]
for r in train:
 a=clone(r)
 for msg in a['messages']:
  if msg['role']=='assistant':msg['content']='<think></think><answer>'+extract_answer_text(msg['content'])+'</answer>'
 answer.append(a)
 c=clone(r)
 if rng.random()<.5:
  target=next(x['content'] for x in r['messages'] if x['role']=='assistant');es=parse_triples(extract_answer_text(target),strict=False,where='train_only');cs=[]
  for e in es:
   if rng.random()<.25:continue
   mention=e.text;typ=e.etype
   if rng.random()<.25:typ=rng.choice([x for x in ['PER','LOC','ORG','OTHER'] if x!=typ])
   if rng.random()<.2 and len(mention.split())>1:mention=' '.join(mention.split()[:-1])
   cs.append(dict(mention=mention,type=typ))
  if rng.random()<.3:
   words=txt(r).split()
   if words:cs.append(dict(mention=rng.choice(words),type=rng.choice(['PER','LOC','ORG','OTHER'])))
  hints(c,cs)
 correction.append(c)
for name,rows in [('control_train.jsonl',train),('answer_train.jsonl',answer),('correction_train.jsonl',correction),('dev.jsonl',splits['dev'])]:dump(name,rows)
for split in ['dev','test']:
 pg=read(base.legacy.R/f'pgmner_predicted_entities_{split}.jsonl');rows=[]
 for r,s in zip(splits[split],pg):
  assert norm(txt(r))==norm(s['text'])
  c=clone(r);hints(c,[dict(mention=e['mention'],type=e['official_type']) for e in s['entities'] if e['entity_confidence']>=.9]);rows.append(c)
 ids=list(range(len(rows))) if split=='dev' else base.old.test_indices()
 dump(f'correction_{split}.jsonl',[rows[i] for i in ids])
 if split=='test':dump('test835.jsonl',[splits['test'][i] for i in ids])
(OUT/'audit.json').write_text(json.dumps(dict(original_train=len(splits['train']),filtered_train=len(train),excluded_overlap=len(splits['train'])-len(train),dev=len(splits['dev']),test=835,seed=42,correction='Synthetic corruptions of TRAIN labels only; DEV/TEST hints from real frozen PGMNER predictions.',hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.jsonl')}),indent=2))
print((OUT/'audit.json').read_text())
