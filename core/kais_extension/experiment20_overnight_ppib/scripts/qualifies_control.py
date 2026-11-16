#!/usr/bin/env python3
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); d=json.loads(p.read_text())
ok=d.get("status")=="complete" and (d.get("best_dev_f1",0)>0.89462 or (d.get("best_dev_f1",0)>=0.89462 and d.get("changed_valid_span_predictions",0)>0))
raise SystemExit(0 if ok else 1)
