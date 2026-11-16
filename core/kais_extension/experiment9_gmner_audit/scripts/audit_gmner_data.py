#!/usr/bin/env python3
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('/home/jyx/yixiao/MINER-My')
OUT = ROOT / 'kais_extension/experiment9_gmner_audit/results'
GMNER_ROOT = Path('/data/hfy/VG-Think/E2E-GMNER/data')
SFT = {
    'train': GMNER_ROOT / 'sft/train_sft.jsonl',
    'dev': GMNER_ROOT / 'sft/dev_sft.jsonl',
    'test': GMNER_ROOT / 'sft/test_sft.jsonl',
}
PGMNER = {
    'twitter2015': {
        'train': ROOT / 'data/twitter2015/train.txt',
        'dev': ROOT / 'data/twitter2015/dev.txt',
        'test': ROOT / 'data/twitter2015/test.txt',
    },
    'twitter2017': {
        'train': ROOT / 'data/twitter2017/train.txt',
        'dev': ROOT / 'data/twitter2017/dev.txt',
        'test': ROOT / 'data/twitter2017/test.txt',
    },
}
TYPE_MAP = {'OTHER': 'MISC', 'MISC': 'MISC', 'PER': 'PER', 'PERSON': 'PER', 'LOC': 'LOC', 'LOCATION': 'LOC', 'ORG': 'ORG', 'ORGANIZATION': 'ORG'}

TEXT_RE = re.compile(r'Text:\s*(.*)$', re.S)
ANSWER_RE = re.compile(r'<answer>(.*?)</answer>', re.S)
BOX_RE = re.compile(r'^\s*\[\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\]\s*$')

def norm_text(s):
    return re.sub(r'\s+', ' ', (s or '').strip())

def parse_user_text(content):
    m = TEXT_RE.search(content or '')
    return norm_text(m.group(1) if m else content)

def parse_answer(content):
    m = ANSWER_RE.search(content or '')
    ans = m.group(1) if m else content
    ans = ans.strip()
    if not ans or ans == 'None':
        return []
    ents = []
    for part in ans.split(';'):
        part = part.strip()
        if not part or part == 'None':
            continue
        bits = [b.strip() for b in part.split('|')]
        if len(bits) < 3:
            ents.append({'raw': part, 'parse_error': 'lt3_fields'})
            continue
        mention = bits[0]
        typ = TYPE_MAP.get(bits[1].upper(), bits[1].upper())
        box_s = '|'.join(bits[2:]).strip()
        if box_s == 'None':
            bbox = None
        else:
            bm = BOX_RE.match(box_s)
            bbox = [float(x) for x in bm.groups()] if bm else None
        ents.append({'mention': mention, 'type': typ, 'bbox': bbox, 'raw_bbox': box_s, 'groundable': bbox is not None})
    return ents

def image_id_from_path(p):
    stem = Path(p).stem
    return stem

def read_sft(split, path):
    rows = []
    if not path.exists():
        return rows
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            msgs = obj.get('messages') or []
            user = next((m for m in msgs if m.get('role') == 'user'), {})
            asst = next((m for m in msgs if m.get('role') == 'assistant'), {})
            text = parse_user_text(user.get('content', ''))
            ents = parse_answer(asst.get('content', ''))
            images = obj.get('images') or []
            image_rel = images[0] if images else ''
            image_abs = GMNER_ROOT / image_rel if image_rel else None
            rows.append({'split': split, 'id': obj.get('id'), 'text': text, 'image_rel': image_rel, 'image_abs': str(image_abs) if image_abs else '', 'image_id': image_id_from_path(image_rel) if image_rel else '', 'entities': ents})
    return rows

def read_pgmner(path):
    samples = []
    if not path.exists():
        return samples
    img = None; toks=[]; labs=[]
    def flush():
        if img is not None:
            samples.append({'image_id': img, 'text': norm_text(' '.join(toks)), 'tokens': toks[:], 'labels': labs[:]})
    with path.open('r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line=line.rstrip('\n')
            if not line:
                flush(); img=None; toks=[]; labs=[]; continue
            if line.startswith('IMGID:'):
                flush(); img=line.split(':',1)[1].strip(); toks=[]; labs=[]; continue
            parts=line.split('\t')
            if len(parts) >= 2:
                toks.append(parts[0]); labs.append(parts[-1])
    flush()
    return samples

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gm_rows = []
    for split, path in SFT.items():
        gm_rows.extend(read_sft(split, path))

    pg_by_img = defaultdict(list)
    pg_by_text = defaultdict(list)
    pg_counts = {}
    for ds, splits in PGMNER.items():
        for split, path in splits.items():
            samples = read_pgmner(path)
            pg_counts[f'{ds}/{split}'] = len(samples)
            for s in samples:
                rec = {'dataset': ds, 'split': split, **s}
                pg_by_img[s['image_id']].append(rec)
                pg_by_text[s['text']].append(rec)

    split_stats = {}
    entity_rows = []
    image_exists = 0
    bbox_multi_count = 0
    parse_errors = 0
    compat = Counter()
    for r in gm_rows:
        split = r['split']
        st = split_stats.setdefault(split, Counter())
        st['samples'] += 1
        st['entities'] += len(r['entities'])
        if r['image_abs'] and Path(r['image_abs']).exists():
            image_exists += 1; st['image_exists'] += 1
        img_matches = pg_by_img.get(r['image_id'], [])
        text_matches = pg_by_text.get(r['text'], [])
        if img_matches:
            compat['image_id_match'] += 1; st['pgmner_image_match'] += 1
        if text_matches:
            compat['text_match'] += 1; st['pgmner_text_match'] += 1
        if any(m['dataset']=='twitter2015' for m in img_matches + text_matches):
            compat['twitter2015_match'] += 1; st['twitter2015_match'] += 1
        if any(m['dataset']=='twitter2017' for m in img_matches + text_matches):
            compat['twitter2017_match'] += 1; st['twitter2017_match'] += 1
        for idx,e in enumerate(r['entities']):
            if e.get('parse_error'):
                parse_errors += 1; st['parse_errors'] += 1
                continue
            typ=e['type']
            groundable=bool(e['groundable'])
            st[f'type_{typ}'] += 1
            st['groundable' if groundable else 'ungroundable'] += 1
            st[f'{typ}_groundable' if groundable else f'{typ}_ungroundable'] += 1
            entity_rows.append({
                'split': split, 'sample_id': r['id'], 'image_id': r['image_id'], 'image_rel': r['image_rel'],
                'text': r['text'], 'entity_index': idx, 'mention': e['mention'], 'type': typ,
                'groundable': int(groundable), 'bbox': json.dumps(e['bbox']) if e['bbox'] is not None else '',
                'pgmner_image_match': int(bool(img_matches)), 'pgmner_text_match': int(bool(text_matches)),
                'pgmner_match_datasets': ';'.join(sorted(set(m['dataset'] for m in img_matches + text_matches))),
            })

    with (OUT/'entity_statistics.csv').open('w', newline='', encoding='utf-8') as f:
        fields = ['split','sample_id','image_id','image_rel','text','entity_index','mention','type','groundable','bbox','pgmner_image_match','pgmner_text_match','pgmner_match_datasets']
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(entity_rows)

    by_split = {k: dict(v) for k,v in split_stats.items()}
    total_entities=len(entity_rows)
    audit = {
        'twitter_gmner_local_exists': all(p.exists() for p in SFT.values()),
        'dataset_root': str(GMNER_ROOT),
        'annotation_files': {k: str(v) for k,v in SFT.items()},
        'image_root': str(GMNER_ROOT),
        'splits': {k: {'path': str(v), 'line_count': sum(1 for _ in v.open()) if v.exists() else None} for k,v in SFT.items()},
        'format': 'jsonl with messages; user content contains Text; assistant content contains entity|type|bbox or None; images is a relative path list',
        'bbox_representation': '[x1,y1,x2,y2] in assistant answer, or None for ungroundable; parsed as floats',
        'ungroundable_representation': 'bbox string None',
        'multiple_boxes_per_entity': 'not observed by parser; schema appears one bbox or None per entity',
        'image_dimensions_available': 'available by opening image files; not embedded in JSONL',
        'sample_count_total': len(gm_rows),
        'entity_count_total': total_entities,
        'groundable_total': sum(int(r['groundable']) for r in entity_rows),
        'ungroundable_total': sum(1-int(r['groundable']) for r in entity_rows),
        'image_files_existing_samples': image_exists,
        'parse_errors': parse_errors,
        'split_stats': by_split,
        'pgmner_sample_counts': pg_counts,
        'compatibility': dict(compat),
        'compatibility_summary': 'GMNER SFT paths indicate Twitter-2015 images; matching against local PGMNER Twitter-2015/2017 is recorded by image id and normalized text.',
    }
    with (OUT/'dataset_audit.json').open('w', encoding='utf-8') as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)
    print(json.dumps(audit, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
