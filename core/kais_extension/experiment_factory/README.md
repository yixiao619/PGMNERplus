# PGMNER Experiment Factory V2

This local-only factory runs DEV-selected PP-IB/GOT-A screening jobs without modifying the original PGMNER source or historical checkpoints. It never evaluates TEST.

## Route audit

- **P1 KL regularization:** original weight, half weight, and two-epoch KL warmup. The Full Stage-1 job is the same-protocol baseline.
- **P2 prediction-coupled IB:** implemented through the Exp20 wrapper, but disabled by default because Exp20 `B2` and `B2_fixed` already provide the exact learned/fixed matched experiment and tied on DEV. Their artifacts are referenced in `routes.json`.
- **P3 stable selective gate:** entity-conditioned gate with temperature 2, a small target-variance hinge, and matched fixed-gate control. This differs from Repair-A and Exp20's unconstrained gate; variance alone is not treated as success.
- **G1 Partial vs Full OT:** replaces only the Sinkhorn backend after loading the same checkpoint. Both variants train under the same DEV-only protocol.
- **G2 adaptive mass:** predicts bounded per-sample mass from cost-matrix mean, standard deviation, and minimum. Its final layer is initialized to the checkpoint mass, preserving initial behavior. The control uses the same fixed actual mass.
- **G3 reliability-weighted OT:** learns token/patch reliabilities and rescales pair costs. Zero initialization yields reliability factor 1 and exact original-cost behavior. The control sets weighting strength to zero.
- **G4 OT loss optimization:** constant weight versus a two-epoch OT-loss warmup. Exp22 already verified the GOT-A loss gradient path.

## Shared-GPU policy

`configs/gpu_policy.json` contains the explicit UUID whitelist. A job starts when:

`free_memory >= expected_peak + max(15% total_memory, 12 GiB)`.

High utilization is logged but does not block low-memory jobs. At most one Factory process is placed on a GPU and global concurrency defaults to two. Existing processes are never killed, reprioritized, reset, or modified. CUDA memory-fraction APIs are not used. OOM failures are recorded and are not retried.

Every assignment records physical UUID, PCI bus ID, free/used memory, utilization, existing compute processes, expected peak, and reserve. Peak memory measured by the job is included in its final status.

## Immutable workflow

```bash
cd /home/jyx/yixiao/MINER-My/kais_extension/experiment_factory

# Inspect without writing.
python submit.py --stage 1 --dry-run

# Snapshot Stage 0 real-batch audits, or Stage 1 short screening.
python submit.py --stage 0
python submit.py --stage 1

# Start the latest immutable submission in a detached session.
bash scripts/start_background.sh

# Status and safe stop (stop schedules no new jobs; active jobs finish).
python status.py
python status.py --request-stop

# Refresh summaries manually if needed.
python summarize.py --submission "$(cat runs/latest_submission.txt)"
```

Editing source configs never changes an existing submission because `submit.py` copies every selected job and policy into a timestamped directory.

Stage 3 is rejected unless `stage3.enable_full_training` is explicitly set to `true`. Candidate promotion remains a human decision after matched-control review.

## Expected cost

Historical H200 measurements give the following planning ranges under contention:

| Route | Stage-1 epochs | Expected peak | Approx. wall time |
|---|---:|---:|---:|
| P1 / Full baseline | 4 | 5.2 GiB | 12-25 min/job |
| P3 | 4 | 4.3 GiB | 15-30 min/job |
| G1 / G4 | 4 | 5.2 GiB | 12-25 min/job |
| G2 / G3 | 4 | 4.3-4.4 GiB | 12-30 min/job |

Shared-GPU contention can increase these times. The scheduler timeout and actual peak usage remain part of the immutable submission record.

## Screening integrity

Stage-1 one-seed gains are signals only. A route is not attributed to a learned gate/mass/reliability mechanism unless its matched control is weaker. Constant mechanisms, zero prediction changes with no DEV gain, collapse, non-finite loss, or absent task gradients are elimination evidence. TEST is intentionally absent from all runners and summaries.
