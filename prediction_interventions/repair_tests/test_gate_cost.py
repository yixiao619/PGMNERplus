"""Verify plan-level gate intervention and gradients, independent of fusion."""
import copy
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.repaired_alignment import RepairedAligner

torch.set_num_threads(1)
torch.manual_seed(71)
device = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
model = RepairedAligner(8, 6, 8, n_iters=100,
                        cost_mode='gate_reliability').to(device).eval()
x = torch.randn(2, 4, 8, device=device)
v = torch.randn(2, 5, 6, device=device)
mt = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], device=device)
mv = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]], device=device)
old = copy.deepcopy(model)
old.cost_mode = 'cosine'

def plan(m):
    return m(x, v, mt, mv)[4]

base = plan(model)
oldbase = plan(old)
assert torch.equal(base, plan(model))
# Nonuniform gate changes, without changing feature directions or values' mu.
with torch.no_grad():
    model.ib_t.gate.weight.mul_(40)
    old.ib_t.gate.weight.copy_(model.ib_t.gate.weight)
new = plan(model)
legacy_drift = (plan(old) - oldbase).abs().max().item()
new_drift = (new - base).abs().max().item()
assert legacy_drift < 2e-6, legacy_drift
assert new_drift > 1e-4, new_drift
assert new[0, 3].count_nonzero() == 0
assert new[0, :, 4].count_nonzero() == 0
assert model.solver.last_residual.max() < 1e-4
# A task depending ONLY on the plan must train both gates at inference-time
# means, demonstrating a route distinct from gated value/fusion amplitudes.
(new * torch.randn_like(new)).sum().backward()
norms = {}
for name in ['ib_t.gate.weight', 'ib_v.gate.weight']:
    grad = dict(model.named_parameters())[name].grad
    assert grad is not None and torch.isfinite(grad).all() and grad.norm() > 1e-7
    norms[name] = grad.norm().item()
# When the IB is removed or unpolarized, matching reduces exactly to cosine.
for mode in ['off', 'vib']:
    model.ib = old.ib = mode
    assert torch.equal(plan(model), plan(old))
print(json.dumps({'device': device, 'legacy_plan_drift': legacy_drift,
                  'new_plan_drift': new_drift, 'plan_only_gradients': norms,
                  'status': 'PASS'}, indent=2))
