import json,csv,argparse
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--audit',required=True);a=p.parse_args();root=Path(a.audit)
d=json.loads((root/'samples.json').read_text());z=np.load(root/'per_post_counts.npz');names=z.files
texts={};image_key=None
for line in Path('/data/jyx/twitter2017/dev.txt').read_text().splitlines():
 if line.startswith('IMGID:'):
  image_key=line.split(':',1)[1].strip();texts[image_key]=[]
 elif line.strip() and image_key is not None:texts[image_key].append(line.split('\t')[0])
rows=[]
for i,post in enumerate(d['posts']):
 row={'index':i,'image_id':post['image_id'],'donor_image_id':post['donor_image_id'],'readable':post['readable'],'donor_readable':post['donor_readable'],'changed_candidates':len(post['changes']),'text':' '.join(post.get('words',texts.get(post['image_id'].removesuffix('.jpg'),[])))}
 for k in names:
  c=np.asarray(z[k][i]).reshape(-1);row[k+'_errors']=float(c[1]+c[2]-2*c[0])
 for k in names[1:]:row[k+'_error_delta']=row[k+'_errors']-row['native_errors']
 rows.append(row)
with (root/'sample_error_attribution.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary={k:{'improved_posts':sum(r[k+'_error_delta']<0 for r in rows),'worsened_posts':sum(r[k+'_error_delta']>0 for r in rows),'total_error_delta':sum(r[k+'_error_delta'] for r in rows)} for k in names[1:]}
summary['note']='Errors=FP+FN from exact pruned entities; per-post error changes do not equal global F1 deltas. All posts retained.'
(root/'sample_attribution_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
