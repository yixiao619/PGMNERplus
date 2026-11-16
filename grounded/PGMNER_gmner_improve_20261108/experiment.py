"""DEV-only policy selection; official scorer and immutable legacy caches."""
import sys, json, hashlib, itertools, argparse
from pathlib import Path
from collections import Counter
import numpy as np
sys.path.insert(0, '/home/jyx/PGMNER_routing_baselines')
import evaluate as legacy

OUT = Path(__file__).parent / 'results'
OUT.mkdir(exist_ok=True)
TYPES = ['PER', 'LOC', 'ORG', 'OTHER']
FILTERS = [0., .9, .95, .98, .99, .995]
ROUTES = [('direct', 0.), ('none', 0.), ('head', .2), ('head', .5), ('head', .8), ('head', .95), ('entity', .98), ('entity', .995)]
OPTIONS = [(f, r, t) for f in FILTERS for r, t in ROUTES]

def save(name, value):
    p = OUT / name
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temp.replace(p)

def test_indices():
    side = legacy.read(legacy.ROOT / 'kais_extension/experiment9_gmner_audit/cache/gmner_pgmner_format/gmner_entity_sidecar.jsonl')
    seen = {legacy.norm(x['text']) for x in side if x['split'] in ('train', 'dev')}
    rows = [x for x in side if x['split'] == 'test']
    idx = [i for i, x in enumerate(rows) if 'twitter2015' in x['image_rel'].lower() and legacy.norm(x['text']) not in seen]
    assert len(idx) == 835
    return idx

def context_cache(split, data):
    paths = list(OUT.glob(f'context_{split}_shard*.jsonl'))
    cache = {x['key']: x for p in paths for x in legacy.read(p)}
    indices = range(len(data[0])) if split == 'dev' else test_indices()
    expected = {legacy.m.cache_key(data[0][i]['sample_index'], e) for i in indices for e in data[0][i]['entities']}
    assert expected <= cache.keys(), f'Missing {len(expected-cache.keys())} context generations'
    return cache

def predict(sample, q, policy, only_type=None):
    pred = []
    for e in sample['entities']:
        typ = e['official_type']
        if only_type and typ != only_type:
            continue
        f, route, threshold = policy[typ]
        if e['entity_confidence'] < f:
            continue
        qr = q[legacy.m.cache_key(sample['sample_index'], e)]
        reject = route == 'none' or (route == 'head' and e['groundability_probability'] < threshold) or (route == 'entity' and e['entity_confidence'] < threshold)
        pred.append(legacy.m.EntityTriple(e['mention'], typ, None if reject else qr['pred_boxes'], bool(qr['parse_valid']) or reject))
    return legacy.m.parse_triples(legacy.m.triples_to_canon_text(pred), strict=False, where='uniform_canonical')

def counts(data, policy, idx=None):
    pg, q, gold = data
    return np.array([[legacy.m.count_correct_gmner(p := predict(pg[i], q, policy), gold[i], .5), len(p), len(gold[i])] for i in (range(len(pg)) if idx is None else idx)])

def score(c):
    return legacy.f1(c)

def table(data):
    pg, q, gold = data
    arr = np.zeros((len(TYPES), len(OPTIONS), len(pg), 3), dtype=np.int64)
    for ti, typ in enumerate(TYPES):
        for oi, option in enumerate(OPTIONS):
            policy = {t: option for t in TYPES}
            for i, s in enumerate(pg):
                p = predict(s, q, policy, typ)
                g = [g for g in gold[i] if g.etype == typ]
                arr[ti, oi, i] = [legacy.m.count_correct_gmner(p, g, .5), len(p), len(g)]
    # A partition by type must give exactly the same official score.
    k = OPTIONS.index((.98, 'head', .5))
    assert np.array_equal(arr[:, k].sum(0), counts(data, {t: OPTIONS[k] for t in TYPES}))
    return arr

def fit(arr, idx, family):
    if family == 'global':
        oi = max(range(len(OPTIONS)), key=lambda j: score(arr[:, j, idx].sum(0)))
        return [oi] * 4
    # Fixed three coordinate passes; each maximizes joint micro-F1, not per-type F1.
    chosen = [OPTIONS.index((.98, 'head', .5))] * 4
    for _ in range(3):
        for ti in range(4):
            def objective(j):
                opt = chosen.copy(); opt[ti] = j
                return score(sum(arr[k, opt[k]][idx] for k in range(4)))
            chosen[ti] = max(range(len(OPTIONS)), key=objective)
    return chosen

def policy(indices):
    return {typ: OPTIONS[indices[i]] for i, typ in enumerate(TYPES)}

def diagnose(data, pol, idx):
    pg, q, gold = data
    # Official matching gaps avoid ambiguous greedy assignment for duplicated mentions.
    totals = Counter(); by_type = {t: Counter() for t in TYPES}
    for i in idx:
        pred = predict(pg[i], q, pol)
        for typ in TYPES:
            p = [p for p in pred if p.etype == typ]; g = [g for g in gold[i] if g.etype == typ]
            ner = legacy.m.count_correct_mner(p, g)
            # Remove localization only, preserving presence/None and region_valid.
            pp = [legacy.m.EntityTriple(x.text, x.etype, [(0,0,1,1)] if x.regions is not None else None, x.region_valid) for x in p]
            gg = [legacy.m.EntityTriple(x.text, x.etype, [(0,0,1,1)] if x.regions is not None else None, True) for x in g]
            routing = legacy.m.count_correct_gmner(pp, gg, .5)
            joint = legacy.m.count_correct_gmner(p, g, .5)
            row = dict(gold=len(g), gold_box=sum(x.regions is not None for x in g), predicted=len(p), entity_matched=ner, joint_correct=joint, missed_entity=len(g)-ner, spurious_entity=len(p)-ner, presence_or_format_gap=ner-routing, localization_gap=routing-joint)
            assert ner >= routing >= joint
            by_type[typ].update(row); totals.update(row)
    return {'total': dict(totals), 'by_type': {t: dict(v) for t,v in by_type.items()}, 'definition': 'Gold errors partition into missed_entity + presence_or_format_gap + localization_gap + joint_correct. Presence gap includes parse validity. Official maximum matching; no single-box sidecar labels.'}

def baseline():
    dev = legacy.load('dev'); test = legacy.load('test'); idx = test_indices()
    records = []
    locked = json.loads(Path('/home/jyx/PGMNER_routing_baselines/dev_selection.json').read_text())['locked']
    for cal in (False, True):
        for mode in ('direct','none','type','entity_conf','head'):
            c = legacy.counts(test, mode, locked.get(mode), cal, idx)
            records.append(dict(method=mode, filter_098=cal, gmner_f1=score(c), counts=c.sum(0).tolist()))
    historical = next(x for x in records if x['method']=='head' and x['filter_098'])
    assert abs(historical['gmner_f1']-.46337)<1e-5
    pol = {t: (.98,'head',.5) for t in TYPES}
    save('baseline_and_errors.json', {'results':records, 'dev_errors':diagnose(dev,pol,range(len(dev[0]))), 'test835_errors':diagnose(test,pol,idx), 'test_status':'Previously examined test; exploratory comparison, not fresh confirmation.'})
    print('BASELINE_REPRODUCED', historical, flush=True)

def select(include_context):
    dev = legacy.load('dev')
    # Hash-grouped by normalized text to keep repeated posts in one partition.
    fitidx = [i for i,s in enumerate(dev[0]) if int(hashlib.sha256(legacy.norm(s['text']).encode()).hexdigest()[:8],16)%10 < 7]
    validx = sorted(set(range(len(dev[0])))-set(fitidx))
    sources = {'original': dev}
    if include_context:
        sources['context'] = (dev[0], context_cache('dev', dev), dev[2])
    trials = []; tables = {}
    fixed = {t:(.98,'head',.5) for t in TYPES}
    # Unmodified baseline is first and wins ties.
    trials.append(dict(source='original',family='fixed_baseline',validation_f1=score(counts(dev,fixed,validx)),policy=fixed))
    for src, data in sources.items():
        a = table(data); tables[src] = a
        for fam in ('global','type_specific'):
            chosen = fit(a,fitidx,fam)
            trials.append(dict(source=src,family=fam,validation_f1=score(sum(a[k,chosen[k]][validx] for k in range(4))),policy=policy(chosen)))
    winner = max(trials,key=lambda x:x['validation_f1'])
    finalpol = fixed if winner['family']=='fixed_baseline' else policy(fit(tables[winner['source']],list(range(len(dev[0]))),winner['family']))
    result = dict(winner=winner, final_policy=finalpol, trials=trials, fit_posts=len(fitidx), validation_posts=len(validx), dev_f1_after_refit=score(counts(sources[winner['source']],finalpol)), selection='Family/source selected on grouped DEV holdout; threshold fit on DEV fit partition; chosen family refit on full DEV before TEST. All rules canonicalized identically.', seed=42)
    save('locked_policy.json' if include_context else 'cached_dev_pilot.json',result)
    print(json.dumps(result),flush=True)

def final():
    lock=json.loads((OUT/'locked_policy.json').read_text())
    data=legacy.load('test'); idx=test_indices()
    if lock['winner']['source']=='context':
        data=(data[0],context_cache('test',data),data[2])
    c=counts(data,lock['final_policy'],idx)
    baseline_data=legacy.load('test')
    b=counts(baseline_data,{t:(.98,'head',.5) for t in TYPES},idx)
    rng=np.random.default_rng(42); boot=[]
    for _ in range(2000):
        ii=rng.integers(0,len(idx),len(idx));boot.append(score(c[ii])-score(b[ii]))
    np.savez(OUT/'final_per_post_counts.npz',candidate=c,baseline=b)
    save('final_results.json',dict(gmner_f1=score(c),baseline_f1=score(b),delta_pp=100*(score(c)-score(b)),paired_post_bootstrap_95ci_pp=(100*np.quantile(boot,[.025,.975])).tolist(),counts=c.sum(0).tolist(),lock=lock,errors=diagnose(data,lock['final_policy'],idx),scope='Same 835-post protocol; test previously examined; single-seed exploratory calibration/grounding extension, not PPIB/GOTA retraining or SOTA claim.'))
    print((OUT/'final_results.json').read_text(),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['baseline','pilot','select','final']);a=p.parse_args()
    if a.stage=='baseline': baseline()
    elif a.stage=='pilot': select(False)
    elif a.stage=='select': select(True)
    else: final()
