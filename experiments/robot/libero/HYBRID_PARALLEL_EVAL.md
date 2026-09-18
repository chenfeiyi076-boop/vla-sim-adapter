# Hybrid evaluation: serial reference, partial results and independent workers

## Standard suite selection

All three Hybrid entry points accept `--task-suite`, default `libero_spatial`.
`hybrid_suite_config.py` maps the four supported suites to their RLDS keys.
The official `TaskSuite` and `TASK_MAX_STEPS` definitions were moved unchanged
from `run_libero_eval.py` to the lightweight `libero_suite_config.py`; the official
evaluator re-exports them. There is one step-limit table, not a Hybrid copy.

| Suite | Statistics/checkpoint dataset key | Max action steps |
| --- | --- | --- |
| libero_spatial | libero_spatial_no_noops | 220 |
| libero_object | libero_object_no_noops | 280 |
| libero_goal | libero_goal_no_noops | 300 |
| libero_10 | libero_10_no_noops | 520 |

Statistics JSON must contain the selected dataset key. Trained evaluation checks
the matching experiment identity from the training registry, dataset key, H/A/P,
normalization vectors and optional expected step. No cross-suite override exists.
`run_hybrid_episode` remains the separate random-head single-episode smoke;
use `run_hybrid_eval` or its parallel launcher for trained checkpoints.

Final results add `task_suite`, `dataset_key`, and `max_episode_steps`; manifests
add the first two and retain `max_action_steps` with the selected limit. Merge
still requires identical worker manifests. Resume keeps exact manifest checks;
do not mix old and new journals within a resumed run. For old-vs-new diagnostic
comparison only, the comparator derives missing identity fields from old Spatial
manifests. No episode/action/hash comparisons are relaxed.

Task ownership remains sorted-selected-task ordinal modulo worker count. Every
task retains one environment and sequential trials in one worker. GPU isolation,
seed formulas, preprocessing, first-observation traces and action execution are
unchanged. An empty worker assignment remains valid.

### CUDA0 trained-checkpoint smoke (not 500 episodes)

Only the Spatial checkpoint is currently available. Use the actual statistics
artifact that matches it, and fresh output/run paths:

```sh
python -m experiments.robot.libero.run_hybrid_eval_parallel \
  --num-workers 1 --devices 0 --task-suite libero_spatial \
  --checkpoint /data/x2227/vla_adapter/VLA-Adapter/runs/hybrid_spatial_phase6d_40k/checkpoints/step-00040000.pt \
  --vlm-path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
  --hf-config pretrained_models/configs --statistics /path/to/dataset_statistics.json \
  --expected-step 40000 --task-id 0 --trials-per-task 2 --debug-trace \
  --run-dir /path/to/eval/spatial-suite-smoke --output /path/to/eval/spatial-suite-smoke.json
```

The launcher sets CUDA_VISIBLE_DEVICES=0, local cuda:0, EGL=0 and disables TF GPU.
First compare a retained old Spatial run with the new run using identical input
assets, task/trial selection, seed, threads and debug/profile flags:

```sh
python -m experiments.robot.libero.compare_hybrid_eval \
  --baseline /path/to/old-spatial/worker-0.partial.json \
  --candidate /path/to/eval/spatial-suite-smoke/worker-0.partial.json
```

Check first raw/processed observation hashes, noise/action hashes, full traces,
zero max action difference and success. Then compare default vs explicit Spatial
and serial vs four-worker task-sharded smokes. CPU tests do not establish real GPU
bit identity; these server checks have not been rerun by local development.

After matching checkpoints/statistics exist, repeat the same command in order
with `--task-suite libero_object`, then `libero_goal`, then `libero_10`, replacing
checkpoint, statistics, expected step, run directory and output for each suite.
Do not reuse Spatial weights or disable validation for these smokes. No automatic
downloads, non-Spatial training, or full benchmark run is part of this change.

This extends evaluation only. Training, checkpoint loaders, normalization, H10,
Euler integration, execute-first-8, settling, original process_action, max steps,
success=done, cameras and trial-to-official-state mapping are unchanged. No new
autocast is enabled. Policy calls remain in torch.inference_mode().

The normal serial path now prepares observations only when the queue is empty.
`--reference` is a debug reference: original eager preprocessing and the default
Torch RNG. Both modes retain the original per-episode torch.manual_seed call.
Normal mode additionally creates one device-local Generator per episode, seeded
with `base_seed + task_id*100000 + trial_id`; all replans advance that same object.
There is no per-replan reseeding. The flow sampler itself is unchanged.

## Files and progress

Use `--output result.json`; the serial journal defaults to
`result.json.partial.json`. Alternatively pass `--partial PATH`. With neither
option, records exist only in memory (no disk resume). Progress goes to stderr,
one flushed line per completed episode, with task/trial, success, observed task
and worker/serial success rates, calls, steps, time and an explicitly estimated
ETA. Final stdout and final JSON retain the old aggregate schema.

Each completed episode atomically publishes the journal with flush/fsync/replace.
It contains the manifest, complete episode identity table, worker assignment and
records. Records contain IDs, official initial-state hash, seed, task description,
success, policy_calls, action_steps and elapsed_sec. Debug/profiling fields are
optional. The manifest binds checkpoint SHA256/path/step/metadata, supplied
normalization vectors, suite/tasks/trials, seed formula, Euler/execution settings,
image settings, model dtype and local HF processor/tokenizer asset hashes.

Resume requires the exact same identity and worker count/assignment. It rejects
duplicates, unknown IDs, modified state hashes and inconsistent counters. It
skips published episodes only, and never merges incompatible experiments. A final
result requires the complete expected ID set. The final file is not overwritten.
Files must be treated as immutable inputs while evaluation is running.

Use `--resume` with the same command. `--stop-after N` is a smoke-only controlled
stop after N **new** episodes per process, after their publication. It leaves no
final JSON. An incomplete parallel run exits with a merge error; this is expected
for the interruption smoke. Rerun without stop-after and with resume.

Exclusive launcher/journal lock files prevent concurrent writers. Normal exits,
exceptions and handled Linux SIGTERM clean them up. Hard kill, power loss or
Windows TerminateProcess may leave locks/temp files. Inspect the recorded PID and
all worker processes first; only after confirming no writer is alive, manually
remove the specific stale lock. Never use broad clean/delete commands. Missing
worker journals on resume start empty; published journals remain authoritative.

## Resource isolation

The parallel launcher starts fresh `python -m ...` processes, never forks a live
CUDA model and never uses DDP. Each worker owns one model/normalizer and at most
one active environment, with its own journal. Parent alone merges/writes final.
Worker failure is detected; peers are terminated/reaped and partials retained.

`--devices` are logical Torch indices under the inherited CUDA_VISIBLE_DEVICES.
`--egl-devices` are REQUIRED explicit EGL enumeration indices, one per worker.
They are not inferred from Torch device numbers. The launcher sets MUJOCO_GL=egl,
PYOPENGL_PLATFORM=egl and each worker's MUJOCO_EGL_DEVICE_ID before imports.
The installed server robosuite/EGL patch and physical-device mapping MUST be
verified; there is no static guarantee these indices map to the desired GPUs.

Worker startup disables TensorFlow GPU visibility before importing the evaluator,
then sets Torch's CUDA device. TF image operations consequently run on CPU; this
placement change needs equivalence smoke against the original serial path.
OMP/MKL/Torch/TF intra-op threads default to 1 per worker, TF inter-op to 1;
`--threads` configures the former. Measure before increasing. Training distributed
rank variables are removed from child environments.

## Required server gate: run from repo root, not from Windows development

Replace the five asset/output paths. Commands use bash arrays so paths stay quoted.
No real CUDA/LIBERO tests or speedup measurements were run on Windows.

```bash
CKPT=/path/to/step-00040000.pt
VLM=pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b
HF=pretrained_models/configs
STATS=/path/to/dataset_statistics.json
RUN=/path/to/new-eval-smoke
mkdir -p "$RUN"
COMMON=(--checkpoint "$CKPT" --vlm-path "$VLM" --hf-config "$HF" --statistics "$STATS" --expected-step 40000 --task-ids 0 4 --trials-per-task 4 --seed 7 --num-steps 10 --debug-trace --profile)
# Explicitly verify these EGL enumeration indices on the server; not an assumed mapping.
EGL=(0 1 2 3)

# A: old eager/default-RNG protocol, instrumented; one GPU, 8 episodes.
python -m experiments.robot.libero.run_hybrid_eval "${COMMON[@]}" --device cuda:0 --reference --output "$RUN/a.json" 2> "$RUN/a.log"
# B: lazy preprocessing and episode Generator, still serial.
python -m experiments.robot.libero.run_hybrid_eval "${COMMON[@]}" --device cuda:0 --output "$RUN/b.json" 2> "$RUN/b.log"
python -m experiments.robot.libero.compare_hybrid_eval --baseline "$RUN/a.json.partial.json" --candidate "$RUN/b.json.partial.json"
# C: fresh-process worker path, one worker (also exercises TF CPU placement).
python -m experiments.robot.libero.run_hybrid_eval_parallel "${COMMON[@]}" --num-workers 1 --devices 0 --egl-devices "${EGL[0]}" --run-dir "$RUN/c" --output "$RUN/c.json" 2> "$RUN/c.log"
python -m experiments.robot.libero.compare_hybrid_eval --baseline "$RUN/b.json.partial.json" --candidate "$RUN/c/worker-0.partial.json"
# D: same eight original task/trial identities, four workers.
python -m experiments.robot.libero.run_hybrid_eval_parallel "${COMMON[@]}" --num-workers 4 --devices 0 1 2 3 --egl-devices "${EGL[@]}" --run-dir "$RUN/d" --output "$RUN/d.json" 2> "$RUN/d.log"
python -m experiments.robot.libero.compare_hybrid_eval --baseline "$RUN/b.json.partial.json" --candidate "$RUN/d/worker-0.partial.json" "$RUN/d/worker-1.partial.json" "$RUN/d/worker-2.partial.json" "$RUN/d/worker-3.partial.json"
# E: controlled interruption after one new episode per worker. Nonzero merge exit expected.
python -m experiments.robot.libero.run_hybrid_eval_parallel "${COMMON[@]}" --num-workers 4 --devices 0 1 2 3 --egl-devices "${EGL[@]}" --run-dir "$RUN/e" --output "$RUN/e.json" --stop-after 1 2> "$RUN/e-stop.log"
python -m experiments.robot.libero.run_hybrid_eval_parallel "${COMMON[@]}" --num-workers 4 --devices 0 1 2 3 --egl-devices "${EGL[@]}" --run-dir "$RUN/e" --output "$RUN/e.json" --resume 2> "$RUN/e-resume.log"
python -m experiments.robot.libero.compare_hybrid_eval --baseline "$RUN/d/worker-0.partial.json" "$RUN/d/worker-1.partial.json" "$RUN/d/worker-2.partial.json" "$RUN/d/worker-3.partial.json" --candidate "$RUN/e/worker-0.partial.json" "$RUN/e/worker-1.partial.json" "$RUN/e/worker-2.partial.json" "$RUN/e/worker-3.partial.json"
```

The comparator checks IDs, states, seeds, task text, successes and counters. Debug
mode records noise hashes and the small executed post-process_action action trace
(no images). Hash differences are reported with max/mean absolute action difference
when shapes match, not automatically classified as a failure. Different outcome
fields produce a nonzero comparator exit. Inspect noise mismatches, not merely
the exit code. Debug is not recommended for full evaluation and does not change
normal precision. Explicit debug noise is drawn with the same shape/device/dtype
and passed as initial_noise to the unchanged sampler.

Environment seed/reset protocol is deliberately UNCHANGED: env.seed(0) at task
creation, official set_init_state at each trial. An environment's reset history
differs under sharding or skipped episodes. This may matter in the installed
LIBERO version. Do not claim equivalent trajectories, safe scientific continuation
or cross-GPU bit identity until A-E verify this. If it fails, stop and investigate;
do not silently reseed/recreate environments to obtain an apparent match.

Only after A-E pass may a full 500-episode run be started manually. Remove task-ids,
use trials-per-task=50, omit debug-trace, and choose a fresh run directory/output.
No command auto-launches that full benchmark after the smoke.

## Profiling and performance evidence

`--profile` prints model-load seconds, environment creation time and segment
totals for reset, prepare_observation, policy inference, flow sampling and env.step.
Flow is a subset of policy time; percentages are NOT additive. CUDA is synchronized
around flow timing only in profiling mode. Episode records include timings; end
summary reports episode/sec, calls/sec and category percentages. Resumed summaries
use newly executed records only. Parallel parent reports wall time; resumed total
episode counts include historical work, so do not divide those totals by resumed
wall time to claim throughput. Compare fresh A/B/D run wall times for speedup.

Externally monitor nvidia-smi utilization/VRAM and CPU usage/load/thread counts.
No GPU utilization, peak VRAM, CPU utilization or speedup is inferred from source.
The first optimization is four independent serial workers; no vectorized or
batched-environment inference is implemented.
