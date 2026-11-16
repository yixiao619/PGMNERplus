import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.repaired_alignment import LocalPolarizedIB

torch.set_num_threads(1)
torch.manual_seed(42)
x = torch.randn(2, 5, 8)
mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1.]])
ib = LocalPolarizedIB(8).eval()
ib.balance_mode = 'bernoulli'
for bias in [-50., -12., 12., 50.]:
    with torch.no_grad():
        ib.gate.weight.zero_()
        ib.gate.bias.fill_(bias)
    ib.zero_grad()
    _, _, loss = ib(x, mask)
    loss.backward()
    grad = ib.gate.bias.grad.item()
    assert torch.isfinite(loss) and abs(grad) > .1
    assert (grad > 0) == (bias > 0), (bias, grad)
    print('saturated bias, restoring gradient:', bias, grad)
with torch.no_grad():
    ib.gate.bias.fill_(.2)
    ib.gate.weight.normal_(std=.2)
_, _, original = ib(x, mask)
_, _, padded = ib(torch.cat([x, torch.randn(2, 7, 8)], 1),
                  torch.cat([mask, torch.zeros(2, 7)], 1))
assert torch.allclose(original, padded, atol=1e-6)
# Legacy objective pushes an already uniform, high gate further upward.
ib.balance_mode = 'legacy'
with torch.no_grad():
    ib.gate.weight.zero_(); ib.gate.bias.fill_(3.)
ib.zero_grad(); ib(x, mask)[2].backward()
assert ib.gate.bias.grad.item() < 0
print('PASS saturated recovery, padding invariance, legacy collapse reproduction')
