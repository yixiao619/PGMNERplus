#!/usr/bin/env bash
E=/home/jyx/yixiao/MINER-My/kais_extension/experiment23_prediction_coupling
python - "$E/run_state.json" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
if p.exists():
 x=json.loads(p.read_text())
 for s in x.get('submissions',[]):Path(s,'stop.request').touch()
print('Stop requested; active jobs will finish and no new jobs will start.')
PY
