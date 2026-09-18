# Phase 6D: formal Spatial baseline and trained-checkpoint evaluation

This is an initial controlled baseline, not a claim of optimal hyperparameters.
It composes validated source sharding (6A), DDP steps (6B), and full training-state
checkpoint/resume (6C). The original VLA-Adapter path and all sealed Hybrid model,
flow, Euler, parameter-selection and action-convention components are unchanged.

## Run the short server gate first

Training dataset identity is selected by `--dataset-key`; the default remains
`libero_spatial_no_noops`. Omitting it and explicitly selecting Spatial produce
identical metadata, including `experiment=hybrid_spatial_formal_v1`, preserving
existing Spatial checkpoint/resume compatibility. The script name is unchanged.

The controlled registry in `prismatic/training/hybrid_formal.py` supports the four
single-dataset keys confirmed in the OXE configs: `libero_spatial_no_noops`,
`libero_object_no_noops`, `libero_goal_no_noops`, and `libero_10_no_noops`.
Non-Spatial identities are respectively `hybrid_object_formal_v1`,
`hybrid_goal_formal_v1`, and `hybrid_10_formal_v1`. Unknown keys and mixtures are
rejected; missing statistics never fall back to Spatial. Dataset changes are
incompatible with strict resume. The evaluator remains Spatial-only in this phase.

The server currently has only `/data/x2227/datasets/libero/libero_spatial_no_noops`.
Object/Goal/LIBERO-10 have CPU configuration tests only, not real training validation.
After the corresponding RLDS data is installed, select its registered key without
editing the trainer. A short identity smoke on the existing Spatial data is:

```sh
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  vla-scripts/train_hybrid_spatial.py \
  --vlm-path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
  --hf-config pretrained_models/configs --data-root /data/x2227/datasets/libero \
  --dataset-key libero_spatial_no_noops --per-device-batch-size 1 \
  --run-dir /path/to/runs/hybrid-spatial-identity-smoke \
  --max-steps 3 --log-every 1
```

Use a fresh run directory. Check `run_config.json` for the Spatial identity,
`train.jsonl` for finite losses/steps 1..3, and `checkpoints/step-00000003.pt`.
This command is for the CUDA server, not the Windows CPU development environment.

From the repository root in the validated four-A100 environment:

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=4 vla-scripts/train_hybrid_spatial.py --vlm-path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b --hf-config pretrained_models/configs --data-root /data/x2227/datasets/libero --run-dir /path/to/runs/hybrid-spatial-gate --max-steps 100 --lr-decay-step 75 --save-every 50 --log-every 10 --shuffle-buffer-size 100
```

Check train.jsonl: updates 1..75 use 1e-6; updates 76..100 use 1e-7. Checkpoints
must exist at steps 50 and 100. Before a long run, also exercise trained loading
and a single task/episode using the real original RLDS statistics JSON:

```sh
python -m experiments.robot.libero.run_hybrid_eval --checkpoint /path/to/runs/hybrid-spatial-gate/checkpoints/step-00000100.pt --vlm-path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b --hf-config pretrained_models/configs --statistics /path/to/dataset_statistics.json --expected-step 100 --task-id 0 --trials-per-task 1 --output /path/to/gate-eval.json
```

Runtime correctness is required; task success is measured, not required by a
programmed threshold. No long training run is launched during Windows development.

## Initial full recipe (after the gate)

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=4 vla-scripts/train_hybrid_spatial.py --vlm-path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b --hf-config pretrained_models/configs --data-root /data/x2227/datasets/libero --run-dir /path/to/runs/hybrid-spatial-baseline --max-steps 40000 --learning-rate 1e-6 --lr-decay-step 30000 --lr-decay-factor 0.1 --save-every 10000 --log-every 20 --shuffle-buffer-size 10000 --seed 7
```

The baseline command uses four NCCL workers, one sample/rank, effective batch 4, no accumulation.
`--per-device-batch-size` controls the physical DataLoader batch (default 1).
One or four torchrun workers are supported; accumulation remains fixed at 1.
Startup prints world_size, per_device_batch_size, gradient_accumulation_steps,
and effective_global_batch_size. The existing metadata field local_batch_size
records this same CLI value; effective_global_batch_size is derived, not another control.
AdamW uses the validated reachable encoder and full flow-head parameter groups
at the same LR. No freezing stage, clipping, warmup, GradScaler or scheduler
object/state. Forward uses bf16 autocast through the outer DDP wrapper with
find_unused_parameters=False and gradient_as_bucket_view=True. Every update
uses the strict hybrid_training_step gradient checks. Loss need not decrease
monotonically. Global step advances only after a successful optimizer update.

LR is stateless and depends on completed steps: global_step 0..29999 selects
1e-6 for the next update; global_step 30000 selects 1e-7 for update 30001.
Resume reconstructs the next LR from global_step, overriding the LR restored
in optimizer groups as appropriate at the boundary. There are no epoch semantics
and no step count derived from len(dataset).

Formal training enables existing image augmentation by default; use
`--no-image-aug` only for an explicitly different compatible recipe. Dataset is
only libero_spatial_no_noops, with two cameras at 224px, observation-only Qwen
prompt, H10/A7/P8 and obs[i]->actions[i:i+10]. Canonical standardized VLA values
are z-scored using GLOBAL statistics. Explicit rank/world_size source splits
precede repeat/window/shuffle; one distinct split per rank is required. One DataLoader
and one iterator are reused, num_workers=0. No DistributedSampler is added.

## Short physical-batch throughput comparison (single CUDA GPU)

Run from the repository root on the validated CUDA server, using fresh run directories:

```sh
for B in 1 2 4 6; do
  CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes=1 --nproc_per_node=1 \
    vla-scripts/train_hybrid_spatial.py \
    --vlm-path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
    --hf-config pretrained_models/configs --data-root /data/x2227/datasets/libero \
    --run-dir /path/to/runs/hybrid-physical-batch-${B} \
    --per-device-batch-size "$B" --max-steps 8 --benchmark-warmup-steps 3 \
    --learning-rate 1e-6 --lr-decay-step 30000 --lr-decay-factor 0.1 \
    --save-every 10000 --log-every 1 --shuffle-buffer-size 10000 --seed 7
done
```

This runs eight physical batches/optimizer updates per process, with the usual
augmentation, AdamW and bf16 autocast. No accumulation, LR scaling or scheduler is
added. The normal final checkpoint is still saved at step 8. Batch size/world size
are part of exact resume metadata: use a fresh run when changing them.

The opt-in benchmark prints per-rank `seconds_per_optimizer_step`, `samples_per_sec`,
`mean_data_wait_sec`, `mean_compute_sec`, `data_wait_fraction`, `compute_fraction`,
and peak allocated/reserved CUDA GiB. CUDA is synchronized before each batch fetch.
Data wait measures only `next(iterator)`; compute measures the existing training
step including CPU-to-GPU transfers, forward, backward, optimizer update and a
final CUDA synchronization. Both means exclude the same warmup steps as total time.
Fractions divide each mean by total seconds per update; their sum can be slightly
below one due to shape/LR checks and post-update cleanup. These wall-clock phases
diagnose input stalls, not GPU utilization: compute can also contain CPU work.
Timing includes data retrieval and one whole
training update with CUDA synchronization; excludes logging and checkpoint I/O.
The first three updates are excluded from timing averages; memory peaks include
warmup and are sampled before the final checkpoint. Compare **1 / seconds_per_step**
against **2 / seconds_per_step**, not optimizer steps per second. For multi-rank
benchmarks these are local rates, not an aggregated global throughput measurement.
Existing strict finite-loss/gradient checks run each step, and final AdamW state
steps must equal global_step. There is no scheduler.step(): LR is computed once
per batch from the completed update count.

These commands require real CUDA, native weights and RLDS assets. CPU tests do not
establish whether batch 2 fits in 40 GiB or improves throughput. If it OOMs, report
that result without substituting accumulation or changing precision/augmentation.

## Outputs and resume

Rank zero atomically creates run_config.json with the formal metadata: experiment,
dataset, world/batch dimensions, H/A/P, optimizer, base LR and decay policy,
augmentation/shuffle settings, source splits, the four float32 normalization
vectors, max/save/log steps and seed. Existing incompatible manifests are fatal;
fresh runs with existing training outputs are rejected. Files are never deleted.

Independent of normal --log-every cadence, train.jsonl always records both sides
of the LR boundary when those updates run: step 75 at 1e-6 and step 76 at 1e-7
for the gate; step 30000 at 1e-6 and step 30001 at 1e-7 for the formal recipe.
At log intervals, these boundary steps and the final step, train.jsonl receives one JSON object:
global_step, learning_rate, loss_mean, t_mean, velocity_rms_mean,
encoder_grad_norm_mean, flow_head_grad_norm_mean, cuda_allocated_gib,
cuda_peak_allocated_gib and timestamp. Scalars are reduced across ranks; memory
values are the maximum across ranks. No graph-bearing history is retained.

Rank zero saves complete v1 checkpoints to checkpoints/step-XXXXXXXX.pt at every
save_every boundary, and once at the final step even if off-boundary. A final
boundary is not saved twice. Before each save, all ranks check AdamW step equals
global_step, and representative encoder/head parameters and AdamW moments agree
within 1e-7. Atomic checkpoint save defaults to no overwrite; disk errors are
broadcast so peers abort. No retention deletion or best-model selection.

Use `--resume /path/to/step-XXXXXXXX.pt` with identical recipe arguments. Every
rank loads full training state and validates all formal metadata, including
source splits and normalization, before training. The restored step must be less
than max_steps. Existing manifests must match. An interrupted run directory or
a new empty directory can be used; existing checkpoint names are never replaced.
To test gate resume from step 50 after a completed gate, use a NEW run directory
with the same max_steps=100, lr_decay_step=75 and save_every=50 arguments.

RLDS iterator/TF shuffle state and exact RNG are not restored. Each resumed
segment starts a new stream in the same source shard and seeds torch with
seed+1000+rank+global_step. This is not bit-exact uninterrupted continuation.
The completion line requires max_steps, matching AdamW states and a final file:
`HYBRID SPATIAL FORMAL TRAINING COMPLETED`. It is not a policy-quality claim.

## Full trained evaluation

```sh
python -m experiments.robot.libero.run_hybrid_eval --checkpoint /path/to/runs/hybrid-spatial-baseline/checkpoints/step-00040000.pt --vlm-path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b --hf-config pretrained_models/configs --statistics /path/to/dataset_statistics.json --expected-step 40000 --num-steps 10 --trials-per-task 50 --output /path/to/spatial-eval.json
```

Only trusted local project checkpoints should be loaded. The evaluation-only
loader strictly restores full trained encoder/head weights without constructing
or restoring an optimizer. It checks format, step and expected formal metadata.
The supplied statistics JSON must contain libero_spatial_no_noops; action and
proprio mean/std are independently canonicalized to float32 and must be numerically
compatible with checkpoint normalization_statistics: evaluation uses torch.allclose
with atol=1e-5 and rtol=1e-5. This only tolerates insignificant floating-point
aggregation/serialization differences; it is not a fallback statistics mechanism.
Training resume metadata compatibility remains exact.

Omit task-id for all Spatial tasks, or specify one task. Exactly 50 official
initial states/task are used by default; requesting more than available fails.
Torch policy noise is seeded deterministically per task/trial. HybridLiberoPolicy
uses 10 Euler steps, denormalizes once to canonical H10, and existing
run_single_episode executes the first 8 actions/query. Official process_action
is the sole canonical-to-environment conversion. Runtime/model errors propagate;
there is no video or parallel simulator execution.

The result reports checkpoint/global_step, Euler/open-loop steps, per-task
successes/trials/rates and policy-call/action-step totals, plus overall successes,
trials and success rate. Optional JSON output is atomic and refuses an existing
path. No minimum success threshold is encoded. This phase does not establish
optimal hyperparameters, convergence or useful policy quality until measured.
