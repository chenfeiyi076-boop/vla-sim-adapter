# Phase 6B: real 20-step DDP smoke

Phase 5B validated one step. This smoke validates consecutive steps on the
Phase-6A source-sharded real RLDS stream, including strict gradient contracts,
DDP synchronization, AdamW state progression and CUDA allocated-memory stability.

From the repository root on the existing four-A100 server environment:

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=4 vla-scripts/hybrid_multistep_smoke.py --vlm-path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b --hf-config pretrained_models/configs --data-root /data/x2227/datasets/libero --steps 20
```

Exactly four CUDA/NCCL workers are required; LOCAL_RANK sets each device.
TensorFlow GPUs are disabled. The unchanged Phase-5B `load_assets` function is
reused, including its native-to-HF mapping and two-camera configuration.
The reachable-vision/legacy exclusion policy and model architecture are unchanged.

One global optimizer step consumes one local batch per rank: four samples
globally, with no accumulation. Twenty steps consume 20 samples/rank, 80 total.
There are no epochs and no use of `len(dataset)` to set the step count.
Precision is bf16 autocast around forward, backward outside autocast; AdamW
uses the exact Hybrid groups at LR 1e-6. There is no scheduler, LR warmup,
clipping, GradScaler, checkpoint, evaluation or optimizer offload.

The outer `HybridFlowTrainingModule` is wrapped once in DDP with
`find_unused_parameters=False`, `gradient_as_bucket_view=True`. Every training
forward enters `training_model(batch, normalizer)` in `hybrid_training_step`.
Access to `.module` is inspection only. Rank-specific seed `1000 + rank` is set
after DDP parameter broadcast; model construction starts with seed 7.

Each rank explicitly supplies `rank` and `world_size` to HybridRLDSDataset:
`libero_spatial_no_noops`, 224px, shuffle buffer 100, train=True, image_aug=False.
Global dataset statistics feed the unchanged HybridZScoreNormalizer. Source
splits are gathered at startup and must be four distinct instructions. This
confirms use of Phase-6A source sharding; it does not repeat or replace that
phase's empirical fingerprint-overlap test. One DataLoader (batch_size=1,
num_workers=0) and one iterator are constructed per rank and reused continuously.

## Per-step utility and diagnostics

`hybrid_training_step` accepts an outer Hybrid module or DDP wrapper, batch,
normalizer, optimizer and autocast settings. It verifies optimizer identities
exactly match requires_grad=True parameters, clears gradients, runs the outer
forward, checks a finite scalar local loss, backpropagates, and requires every
selected parameter to have a gradient. Excluded parameters and Action Queries
must have no gradient. Encoder/flow gradient norms must each be finite and
positive. Norms are aggregated on-device, without per-parameter host syncs.
Any failed check aborts before optimizer.step. It returns Python diagnostics
only; no graph-bearing tensors. Gradients remain available for inspection.

After the step and before clearing gradients, the runner all-gathers the first
64 fp32 gradient values of encoder.projector.fc1.weight and
flow_head.action_decoder.weight. It also compares their current parameter
slices. Maximum differences from rank zero must be <=1e-7. DDP alone reduces
gradients; the only manual SUM reduction is the detached local loss for global
mean logging. Loss is required to be finite, not decreasing.

After step 20 both representative slices must differ from snapshots taken
before step 1. Per-step slice changes are not required. Every optimizer
parameter must have AdamW state with step exactly 20; min/max are reported.

## Memory and acceptance

After each step's diagnostics, gradients are cleared with set_to_none=True,
batch/temporary device references are dropped, and CUDA is synchronized.
Only Python numeric records are retained. The report includes each step's
allocated-after-cleanup GiB and peak allocated GiB. The first three iterations
are allocation warmup (not learning-rate warmup). For each rank:

`steady_memory_span_gib = max(allocated_after_step_gib[3:]) - min(allocated_after_step_gib[3:])`

This must be <=0.5 GiB. Reserved memory is not used; empty_cache is never
called. A failed memory check prints the full series and exits nonzero without
adjusting the threshold. Other failures raise clear errors and exit nonzero.

Rank zero prints structured JSON with all per-rank losses, t means, velocity
RMS, group norms, gradient/exclusion checks, synchronized-slice differences,
parameter changes, memory series, source splits and optimizer state min/max.
Only a successful 20-step run prints:

`PHASE 6B REAL MULTI-STEP DDP TRAINING SMOKE PASSED`

`--steps 1..19` is permitted only for debugging and never prints that line or
sets phase6b_passed=True. Runs of <=3 steps do not assess steady-state memory.

CPU synthetic tests exercise repeated real optimizer steps and fake outer
wrappers/collectives. They do not establish real NCCL or CUDA-memory acceptance.
Even a real 20-step pass does not establish convergence, useful policy quality,
a correct long-run LR schedule, optimal hyperparameters, checkpoint/resume
correctness or LIBERO task success. Those are outside Phase 6B.
