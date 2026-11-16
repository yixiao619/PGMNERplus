import sys,json,time,traceback
from pathlib import Path
sys.path.insert(0,'/home/jyx/PGMNER_gmner_recombine_20261111');import task as old
OUT=Path(__file__).parent;B=old.base
def save(n,x):(OUT/n).write_text(json.dumps(x,indent=2))
def pred(b,threshold):
 data,idx,pg,t,new=b;out={}
 for i in idx:
  td={old.key(x):x for x in t[i]};pk={(e['mention'],e['official_type']) for e in data[0][i]['entities'] if e['entity_confidence']>=threshold}
  out[i]=[td.get(old.key(x),x) for x in new[i] if old.key(x) in td or old.key(x) in pk]
 return out
try:
 save('status.json',dict(stage='dev_selection'))
 b=old.load('dev');trials=[]
 for threshold in [.98,.9,.95,.99,.995]:trials.append(dict(threshold=threshold,**B.metrics(b[0],b[1],pred(b,threshold))))
 win=max(trials,key=lambda x:x['gmner_f1']);save('locked.json',dict(winner=win,trials=trials,scope='Finite DEV-only refinement of PG confidence support; .98 incumbent wins ties.'))
 b=old.load('test');save('results.json',dict(winner=win,selected=B.metrics(b[0],b[1],pred(b,win['threshold'])),incumbent=B.metrics(b[0],b[1],pred(b,.98)),scope='Exploratory previously examined835 TEST; no further tuning.'))
 save('status.json',dict(stage='complete',time=time.strftime('%Y-%m-%d %H:%M:%S')))
except Exception:save('status.json',dict(stage='failed',error=traceback.format_exc()));raise
