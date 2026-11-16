"""Plot every prespecified contrast, irrespective of its sign."""
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(sys.argv[1])
r = json.loads((root/'results.json').read_text())
if not r['complete']:
    raise SystemExit('Do not plot a smoke subset as a full experiment')
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, title, names, labels in [
    (axes[0], 'E1: Gate assignment', ['E1_gate_assignment'], ['Native - shuffled gates']),
    (axes[1], 'E2: Visual matching', ['E2_transport_assignment', 'E2_correct_image'],
     ['Native - independent plan', 'Correct - wrong image'])]:
    estimates = [r['paired'][n] for n in names]
    y = np.array([e['delta_f1']*100 for e in estimates])
    lower = np.array([e['ci95'][0]*100 for e in estimates])
    upper = np.array([e['ci95'][1]*100 for e in estimates])
    ax.scatter(np.arange(len(y)), y, color='#2166ac')
    ax.vlines(np.arange(len(y)), lower, upper, color='#2166ac')
    ax.axhline(0, color='black', linewidth=.8)
    ax.set_xticks(np.arange(len(y)), labels, rotation=12, ha='center', fontsize=8)
    ax.set_ylabel('DEV F1 difference (percentage points)')
    ax.set_title(title)
noise = [0., .25, .5, .75]
for gate, plan, label in [
    ('native', 'native', 'Native gates + learned assignment'),
    ('shuffle_both', 'native', 'Shuffled gates + learned assignment'),
    ('native', 'independent', 'Native gates + independent assignment'),
    ('shuffle_both', 'independent', 'Shuffled gates + independent assignment')]:
    axes[2].plot(np.array(noise)*100,
                 [100*r['f1'][f'n{n:g}_{gate}_{plan}'] for n in noise], marker='o', label=label)
axes[2].set_title('E3: Controlled patch corruption')
axes[2].set_xlabel('Replaced non-CLS patches (%)')
axes[2].set_ylabel('DEV F1 (%)')
axes[2].legend(fontsize=6)
fig.suptitle(root.name + ' | seed 42, exploratory DEV; paired-post 95% intervals', fontsize=11)
fig.tight_layout()
for ext in ['png', 'pdf']:
    fig.savefig(root/f'three_experiments.{ext}', dpi=180)
plt.close(fig)
