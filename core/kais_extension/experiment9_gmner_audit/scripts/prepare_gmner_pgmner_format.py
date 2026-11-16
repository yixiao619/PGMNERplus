#!/usr/bin/env python3
import json, re, os
from pathlib import Path

ROOT = Path('/home/jyx/yixiao/MINER-My')
EXP = ROOT/'kais_extension/experiment9_gmner_audit'
CACHE = EXP/'cache/gmner_pgmner_format'
GMNER_ROOT = Path('/data/hfy/VG-Think/E2E-GMNER/data')
SFT = {'train': GMNER_ROOT/'sft/train_sft.jsonl','dev': GMNER_ROOT/'sft/dev_sft.jsonl','test': GMNER_ROOT/'sft/test_sft.jsonl'}
TYPE_MAP={'OTHER':'MISC','MISC':'MISC','PER':'PER','PERSON':'PER','LOC':'LOC','LOCATION':'LOC','ORG':'ORG','ORGANIZATION':'ORG'}
TEXT_RE=re.compile(r'Text:\s*(.*)$',re.S); ANSWER_RE=re.compile(r'<answer>(.*?)</answer>',re.S)
BOX_RE=re.compile(r'^\s*\[\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\]\s*$')

def norm(s): return re.sub(r'\s+',' ',(s or '').strip())
def parse_text(c):
    m=TEXT_RE.search(c or '')
    return norm(m.group(1) if m else c)
def parse_answer(c):
    m=ANSWER_RE.search(c or '')
    ans=(m.group(1) if m else c or '').strip()
    if not ans or ans=='None': return []
    out=[]
    for p in ans.split(';'):
        p=p.strip()
        if not p or p=='None': continue
        bits=[b.strip() for b in p.split('|')]
        if len(bits)<3: continue
        mention=bits[0]; typ=TYPE_MAP.get(bits[1].upper(), bits[1].upper())
        bs='|'.join(bits[2:]).strip(); bbox=None
        if bs!='None':
            bm=BOX_RE.match(bs)
            if bm: bbox=[float(x) for x in bm.groups()]
        out.append({'mention':mention,'type':typ,'bbox':bbox,'groundable':bbox is not None})
    return out

def label_tokens(tokens, ents):
    labs=['O']*len(tokens); used=[False]*len(tokens)
    low=[t.lower() for t in tokens]
    side=[]
    for ei,e in enumerate(ents):
        mt=e['mention'].split(); ml=[x.lower() for x in mt]
        found=None
        if ml:
            for i in range(0, len(tokens)-len(ml)+1):
                if any(used[i:i+len(ml)]): continue
                if low[i:i+len(ml)]==ml:
                    found=(i,i+len(ml)-1); break
        if found is None:
            side.append({**e,'entity_index':ei,'token_start':None,'token_end':None,'matched_in_text':False})
            continue
        s,t=found
        labs[s]='B-'+e['type']
        for k in range(s+1,t+1): labs[k]='I-'+e['type']
        for k in range(s,t+1): used[k]=True
        side.append({**e,'entity_index':ei,'token_start':s,'token_end':t,'matched_in_text':True})
    return labs, side

def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    imgdir = Path(str(CACHE)+'_images'); imgdir.mkdir(parents=True, exist_ok=True)
    (CACHE/'labels.txt').write_text('O\nPER\nLOC\nORG\nMISC\n', encoding='utf-8')
    metadata=[]; unmatched=0; symlinked=0
    for split,path in SFT.items():
        with path.open(encoding='utf-8') as fin, (CACHE/f'{split}.txt').open('w',encoding='utf-8') as fout:
            for line_idx,line in enumerate(fin):
                obj=json.loads(line); msgs=obj.get('messages') or []
                user=next((m for m in msgs if m.get('role')=='user'),{})
                asst=next((m for m in msgs if m.get('role')=='assistant'),{})
                text=parse_text(user.get('content','')); tokens=text.split()
                ents=parse_answer(asst.get('content',''))
                labs, side=label_tokens(tokens, ents)
                unmatched += sum(1 for s in side if not s['matched_in_text'])
                img_rel=(obj.get('images') or [''])[0]
                src=GMNER_ROOT/img_rel if img_rel else None
                imgid=Path(img_rel).stem if img_rel else f'{split}_{line_idx}'
                dst=imgdir/f'{imgid}.jpg'
                if src and src.exists() and not dst.exists():
                    try:
                        os.symlink(src, dst); symlinked += 1
                    except FileExistsError:
                        pass
                fout.write(f'IMGID:{imgid}\n')
                for tok,lab in zip(tokens,labs): fout.write(f'{tok}\t{lab}\n')
                fout.write('\n')
                metadata.append({'split':split,'sample_id':obj.get('id'),'image_id':imgid,'image_rel':img_rel,'image_path':str(src) if src else '', 'text':text, 'entities':side})
    (CACHE/'gmner_entity_sidecar.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in metadata)+'\n',encoding='utf-8')
    summary={'cache_data_dir':str(CACHE),'cache_image_dir':str(imgdir),'samples':len(metadata),'unmatched_entities':unmatched,'new_symlinks':symlinked}
    (EXP/'results/gmner_pgmner_cache_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
