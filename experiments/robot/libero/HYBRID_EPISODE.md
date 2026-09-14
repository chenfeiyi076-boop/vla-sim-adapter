# Phase 4: one Hybrid LIBERO-Spatial episode

From the repository root, in an environment already able to run the official
LIBERO evaluator, supply the native Prismatic DINOv2/SigLIP + Qwen2.5-0.5B
checkpoint, the existing HF config/tokenizer/processor directory and the original
RLDS statistics JSON exported for the dataset used in Phase 2:

```sh
python -m experiments.robot.libero.run_hybrid_episode --policy hybrid --vlm-path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b --hf-config pretrained_models/configs --statistics /path/to/dataset_statistics.json --task-id 0 --initial-state-index 0 --num-steps 10 --device cuda:0
```

The statistics file must contain the exact outer key `libero_spatial_no_noops`
and its action/proprio mean and standard deviation. The loader does not fall
back to another dataset, recompute statistics, or use a bounds/quantile rule.
The key alone cannot establish file provenance: supply the actual RLDS export,
not invented values or statistics from the official SimVLA dataset pipeline.
No converted HF checkpoint is saved or required. The native loader retains its
existing backbone asset/cache resolution; statistics must be supplied locally.

This runner loads only the Prismatic backbone checkpoint and initializes the
flow head randomly. It does not load/save a flow checkpoint or train. Local
HF classes bypass the legacy loader's checkpoint config/source rewrites. The
runner calls `prismatic.models.load(..., load_for_training=True)` to match the
validated native loading path (this only loads weights; no training occurs),
renames its state dict in memory, then calls `load_state_dict(strict=False)`.
Unexpected keys and missing weights other than `action_queries.weight` are
rejected. The inherited Action Query module is never called. Models run in
float32/eval mode and inference is gradient-free.

## Reused evaluator behavior

- `run_libero_eval.prepare_observation`: primary/wrist extraction using
  `get_libero_image` and `get_libero_wrist_image` (both rotate 180 degrees),
  existing resize, and position + quaternion-to-axis-angle + gripper qpos.
- `openvla_utils.prepare_images_for_vla`: original RGB validation and center
  crop; existing Prismatic processor packs primary before wrist at 224 pixels.
- `get_libero_env`, `load_initial_states`, `get_libero_dummy_action`,
  `get_image_resize_size`, `set_seed_everywhere`, `TASK_MAX_STEPS`: original
  environment, initialization, settling and limits (Spatial: 220 policy steps).
- `run_libero_eval.process_action`: the sole canonical-to-environment
  conversion, immediately before `env.step(action.tolist())`.

The dedicated loop mirrors `run_episode`'s reset, 10 settling steps, action
queue, time limit and `done` success detection. It propagates exceptions rather
than swallowing them. It never calls the benchmark's task/trial loops. Exactly
one selected task and one initial state are run; the environment closes even
on an error. There is no success-rate aggregation or video/log output file.

## Tensor and action contract

`HybridLiberoPolicy` creates one human turn with `QwenPromptBuilder`; no action
tokens, labels or Action Queries enter `encode_observation`. Encoder inputs
are IDs/mask `[1,L]` and pixels `[1,12,224,224]`, with six channels per view.
The encoder returns hidden features `[1,T,896]` and boolean mask `[1,T]`.
Raw proprio `[1,8]` is z-score normalized with `HybridZScoreNormalizer`.
`sample_actions_euler` produces normalized `[1,10,7]` in 10 steps by default.
The bridge checks shapes and finite values, including after denormalization.

**Denormalization occurs only in `HybridLiberoPolicy.__call__`, immediately
after Euler**: `canonical = normalizer.denormalize_action(normalized)`, using
`x * (std + 1e-6) + mean` for all seven dimensions. The returned `[10,7]` chunk
has canonical VLA action units, including canonical gripper. There is no clip,
binarization, inversion or second normalization in the bridge.

**Gripper conversion occurs only in the existing `process_action`**:
`normalize_gripper_action(..., binarize=True)` followed by
`invert_gripper_action` for `model_family="openvla"`. Thus canonical 0 becomes
env +1 (close) and canonical 1 becomes env -1 (open). Other action dimensions
pass through unchanged. The evaluator's existing threshold behavior, including
exactly 0.5, is retained.

The flow horizon is 10. The execution window retains the evaluator default 8:
execute the first eight predicted actions, then re-observe/re-plan. This does
not change `NUM_ACTIONS_CHUNK` or any official VLA path.

## Validation

```sh
python -m pytest -q -p no:cacheprovider tests/test_flow_action_head.py tests/test_hybrid_data.py tests/test_flow_euler_sampler.py tests/test_hybrid_policy.py
```

Bridge tests exercise the real tiny flow head, Euler, z-score normalizer and
Qwen builder with a synthetic encoder/processor. Rollout tests extract the
actual evaluator helpers without importing unavailable simulator dependencies,
and check camera orientation, proprio, queue refill, one-time gripper
conversion, timeout, `done`, and error propagation using a fake environment.
These tests do not establish actual simulator/model integration acceptance.

For the real run, acceptance requires at least one successful Hybrid inference
and `env.step`, with finite valid tensors and no runtime errors through the
episode. `success=False` at the time limit is acceptable for the random head;
the result prints action step and policy call counts. Real-run validation in
the current Windows environment is pending: LIBERO, robosuite, TensorFlow and
timm are absent, CUDA is unavailable, and no local checkpoint/statistics were
found in the main repository. No dependencies were installed.
