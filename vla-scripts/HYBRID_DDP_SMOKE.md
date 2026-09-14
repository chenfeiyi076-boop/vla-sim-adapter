# Phase 5B: one-step DDP mechanics smoke

Run from the repository root in the validated server environment with 4 A100s:

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=4 vla-scripts/hybrid_ddp_smoke.py --vlm-path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b --hf-config pretrained_models/configs --data-root /data/x2227/datasets/libero
```

The script uses NCCL, sets the device from LOCAL_RANK, and requires world size
4. It loads native VLM weights with the validated in-memory HF renames; no HF
weights checkpoint is created. TensorFlow GPUs are disabled before RLDS imports
so decoding does not reserve training GPU memory. Backbone assets/caches must
already be available as in the validated Phase-5A server run.

`HybridFlowTrainingModule` registers `encoder` and `flow_head`. Its forward
delegates directly to the unchanged `hybrid_flow_loss`. The Phase-5A parameter
policy is applied before wrapping the outer module in DDP with
`find_unused_parameters=False` and `gradient_as_bucket_view=True`. Each forward
is `ddp_model(batch, normalizer)`; `.module` access only selects diagnostic
parameters, never invokes encoder/loss outside DDP.

Trainable parameters are exactly the Phase-5A observation encoder modules
(reachable vision backbone, projector, input embeddings and decoder) and all flow-head
parameters. Action Queries, legacy proprio/action modules and the untied LM
vocabulary head remain excluded. Tied embeddings are deduplicated. AdamW uses
these exact two groups with LR 1e-6 and default remaining settings.

Each rank independently constructs `HybridRLDSDataset`,
`HybridRLDSBatchTransform`, and `PaddedCollatorForHybridFlow`, and obtains one
real batch of size 1 from `libero_spatial_no_noops`. Its own dataset's exact
statistics construct `HybridZScoreNormalizer`. H=10/A=7/P=8, 224px inputs,
canonical gripper and flow equations remain unchanged. A small 100-frame
shuffle buffer limits smoke startup cost; no image augmentation is enabled.
This is an IterableDataset: there is no DistributedSampler or guessed sharding.
Ranks may see overlapping data. **This validates DDP mechanics, not final
rank-aware long-run RLDS sharding.**

DDP performs its normal rank-zero parameter broadcast. After construction and
batch loading, PyTorch CPU/CUDA RNGs are seeded with `1000 + rank` so local
flow noise/timesteps differ. One bf16-autocast forward is followed by backward
outside autocast and exactly one AdamW step. Model weights stay float32.

Before the step, all ranks must report finite local loss, finite/nonzero
encoder and flow gradients, and no excluded/Action Query gradients. Missing
gradients on any trainable parameter cause an explicit failure. First-64
slices of `encoder.projector.fc1.weight.grad` and
`flow_head.action_decoder.weight.grad` are all-gathered; maximum difference
from rank zero must be <=1e-7. This tests gradients already synchronized by DDP;
it does not manually average them. After step, the corresponding parameter
slices must change and agree across ranks at the same tolerance. Rank zero
prints all per-rank checks, losses, sampled t means and peak CUDA allocated GiB.

## Validated vision parameter policy

The current HF vision implementation extracts the second-to-last TIMM block's
features. The real single-batch diagnostic found exactly 43 missing-gradient
tensors, all belonging to the final blocks, final norms and SigLIP attention
pool. These modules are structurally outside the selected intermediate output.

`_hybrid_vision_parameters` now excludes `blocks[-1]`, `norm` and `attn_pool`
(when present) from each actual `featurizer` and `fused_featurizer`, using module
structure and parameter identity rather than fixed block indices or arbitrary
name matching. All earlier vision parameters remain eligible. Tiny backbones
without this TIMM structure retain all vision parameters. The exclusions are
applied before DDP reducer construction, with `find_unused_parameters=False`
unchanged. No forward, flow math, normalization or legacy action path changes.
The smoke still rejects any other missing trainable gradient; real four-GPU
synchronization remains a separate server check.

CPU tests use the real wrapper/loss with tiny backbones and fake collectives.
They are not proof of NCCL synchronization or A100 memory use; the command
above must pass on the real server. No formal training loop, scheduler,
checkpoint, evaluation, logging service or rank sharding is added.
