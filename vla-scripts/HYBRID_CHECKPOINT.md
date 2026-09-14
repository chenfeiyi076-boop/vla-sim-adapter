# Phase 6C: checkpoint and true process-restart resume

This phase builds on validated Phase-6A rank source sharding and Phase-6B DDP
training. It proves model, optimizer and global-step continuity across two
separate torchrun jobs. Both jobs must pass; loading inside the save process
does not satisfy the process-restart requirement.

From the repository root in the existing four-A100 environment, run A:

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=4 vla-scripts/hybrid_checkpoint_smoke.py --mode save --checkpoint /path/to/shared/hybrid-step5.pt --vlm-path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b --hf-config pretrained_models/configs --data-root /data/x2227/datasets/libero --steps 5
```

Wait for normal exit of all four workers, then start run B as a NEW command:

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=4 vla-scripts/hybrid_checkpoint_smoke.py --mode resume --checkpoint /path/to/shared/hybrid-step5.pt --vlm-path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b --hf-config pretrained_models/configs --data-root /data/x2227/datasets/libero --steps 5
```

Use a new, writable shared checkpoint path visible to every rank. The script
requires exactly 5 steps in each invocation, world size 4/NCCL, batch size 1 per
rank, bf16 autocast and AdamW LR 1e-6. TensorFlow sees no GPUs. Existing native
loading, model/parameter policy, source sharding and DDP settings are reused.
Each process creates one rank-aware loader and one iterator for its segment.
Every differentiable forward still enters the outer DDP via hybrid_training_step.

## Version 1 payload

The checkpoint contains `format_version`, `global_step`, complete `encoder`
and `flow_head` state_dicts (including excluded parameters/buffers), full AdamW
`optimizer` state_dict (moments, step values and param-group settings), ordered
`optimizer_parameter_names` by named group, and `metadata`.

Canonical names come from the unwrapped module's named_parameters by parameter
identity, so tied parameters appear once in the optimizer layout. Groups must
contain exactly the selected parameters, with no exclusions, missing parameters
or duplicates. Saved and current group order, group names and parameter order
must match exactly before restore. Numeric optimizer IDs alone are insufficient.

Metadata records dataset key, world/local/effective-global batch sizes, H/A/P,
AdamW and LR, accumulation=1, scheduler=None, and four ordered source-split
strings. Resume supplies all fields as expected_metadata and rejects any
mismatch. The schema has no scheduler, scaler, epoch or RNG state.

All saved tensors are recursively detached, copied to CPU and cloned, including
nested optimizer tensors. Snapshot preparation does not mutate live state.
Rank zero alone writes a same-directory temporary file using torch.save, flushes
it, then atomically publishes it with os.replace. On failure only that exact
temporary file is removed. overwrite=False rejects existing destinations before
writing; no retention/deletion policy is introduced. Overwriting is an explicit
utility option, not enabled by this smoke. Save exceptions are broadcast to all
ranks so a failed writer does not leave peers silently waiting at a barrier.

Only load trusted local checkpoints produced by this project. Do not use
torch.load on untrusted checkpoint files. Loading uses map_location="cpu" and
weights_only=True. Schema/version, step, metadata, named layout and AdamW step
compatibility are checked before model mutation. Both model components load
strict=True; optimizer.load_state_dict then restores complete AdamW state.
Selected parameter/layout and step consistency are checked again afterward.
A strict key/shape failure may leave a partial model restore: it is fatal and
training must not continue. Load returns only version, step and metadata.

## Real acceptance

Run A increments global_step only after each successful optimizer step: 0->5.
Before rank-zero save, all optimizer state steps must equal 5, both representative
parameters must have changed, and model parameters and representative AdamW
exp_avg/exp_avg_sq slices must agree across ranks within 1e-7. Full state, not
just these slices, is saved. JSON includes path, size, splits and sync diagnostics.

Run B reconstructs normal assets and a fresh optimizer/DDP group. Every rank
loads the same checkpoint. Before any forward, global_step and every AdamW
state step must equal 5, source splits must match exactly, parameter/moment
slices must synchronize, and exclusions/Action Queries must remain unchanged.
Only then does it seed the resumed segment with `2000 + rank` and train 5->10.
Every resumed step enforces finite loss, all selected gradients present,
excluded/query no-grad, finite/nonzero encoder/flow gradient norms, and gradient
and parameter slice synchronization. Final AdamW steps must all equal 10 and
both representatives must have changed from the restored step-5 snapshot.
There is no loss-decrease requirement and no second step-10 checkpoint.

Formal success requires both lines from the separate jobs:

```text
PHASE 6C CHECKPOINT SAVE PASSED
PHASE 6C CHECKPOINT RESUME PASSED
```

## Explicit limits

No RLDS position, TensorFlow iterator/shuffle buffer, DataLoader iterator or RNG
state is serialized. Resume starts a new stochastic stream in the same rank
source shard; it does not skip N samples to pretend continuity. This is NOT
bit-exact uninterrupted training or identical next-sample/noise replay. CPU
continuation tests manually align future RNG and batches solely to demonstrate
that restored AdamW moments reproduce the same next update.

No scheduler, LR warmup, GradScaler, epoch, accumulation, clipping, evaluation,
simulator, cloud upload or optimizer offload is added. CPU tests do not replace
the two real four-GPU process-restart runs.
