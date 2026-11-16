import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from mechanism_audit import change_gate, decorrelate_plan, paired_estimates
import numpy as np

torch.manual_seed(42)
p = torch.rand(2, 4, 5)
p[0, 3] = 0; p[0, :, 4] = 0
q = decorrelate_plan(p)
assert torch.allclose(p.sum(1), q.sum(1), atol=1e-6)
assert torch.allclose(p.sum(2), q.sum(2), atol=1e-6)
assert q[0, 3].count_nonzero() == q[0, :, 4].count_nonzero() == 0
assert not torch.allclose(p, q)
g = torch.rand(2, 6, 1)
mask = torch.tensor([[1,1,1,0,0,0], [1,1,1,1,1,1]])
for mode in ['mean', 'shuffle']:
    x = change_gate(g, mask, mode, 42)
    for i in range(2):
        ix=mask[i].bool()
        assert torch.allclose(x[i, ix].mean(),g[i, ix].mean())
        if mode == 'shuffle':
            assert torch.equal(x[i, ix].sort(dim=0).values,g[i, ix].sort(dim=0).values)
names=['n0_native_native','n0_shuffle_both_native','n0_native_independent','wrong_image',
       'n0.5_native_native','n0.5_shuffle_both_native','n0.5_native_independent','n0.5_shuffle_both_independent']
result=paired_estimates({k: np.array([[1.,2.,2.], [2.,3.,4.]]) for k in names})
assert all(v['delta_f1']==0 and v['ci95']==[0.,0.] for v in result.values())
print('PASS gate distribution preservation, plan marginals/padding, paired bootstrap null')
