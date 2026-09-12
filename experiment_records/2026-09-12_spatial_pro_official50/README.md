# VLA-Adapter LIBERO-Spatial-Pro Evaluation

## Date

2026-09-12

## Goal

Evaluate the released VLA-Adapter `LIBERO-Spatial-Pro` checkpoint
on the LIBERO Spatial benchmark.

## Base Repository

Repository:

OpenHelix-Team/VLA-Adapter

Base commit:

23fa0c9c159e2aa04341cdd3e924f44061311060

Local experiment commit:

060ce11

Branch:

repro/spatial-pro-eval

## Server / Environment

- 4 × NVIDIA A100-PCIE-40GB
- Python 3.10.16
- PyTorch 2.2.0+cu118
- CUDA runtime 11.8
- robosuite 1.4.1
- mujoco 2.3.7
- numpy 1.26.4

Detailed environment files:

- `system_info.txt`
- `pip_freeze.txt`
- `conda_list.txt`

## Checkpoint

Local checkpoint:

`pretrained_models/LIBERO-Spatial-Pro`

Model:

VLA-Adapter LIBERO-Spatial-Pro

Backbone:

Prismatic VLM with:

- DINOv2
- SigLIP
- Qwen2.5-0.5B

Policy configuration:

- Pro version enabled
- L1 regression action head
- proprio input enabled
- 2 input images
- action chunk length = 8
- action dimension = 7

## Evaluation Protocol

Benchmark:

`libero_spatial`

Number of tasks:

10

Trials per task:

50

Total episodes:

500

Main evaluation options:

- `use_l1_regression = True`
- `use_minivlm = True`
- `num_images_in_input = 2`
- `use_proprio = True`
- `num_open_loop_steps = 8`
- `use_pro_version = True`

## Multi-GPU Evaluation

Evaluation was parallelized across four independent processes.

This was not DDP.

Task split:

- GPU0: task 0, 1, 2
- GPU1: task 3, 4, 5
- GPU2: task 6, 7
- GPU3: task 8, 9

`run_libero_eval.py` was locally modified to support:

- `task_start`
- `task_end`

The exact patch is saved in:

`run_libero_eval.patch`

## EGL Multi-GPU Fix

robosuite 1.4.1 initially failed for GPU1-GPU3 because EGL interpreted
`CUDA_VISIBLE_DEVICES=1/2/3` as the local EGL device index.

Each worker exposes only one GPU, so the local EGL device index should be 0.

The local robosuite EGL implementation was patched so that each isolated
single-GPU worker uses local EGL device 0.

This patch was applied outside the Git repository, inside the Conda environment.

## Results

| Task ID | Successes | Success Rate |
|---:|---:|---:|
| 0 | 50 / 50 | 100% |
| 1 | 50 / 50 | 100% |
| 2 | 50 / 50 | 100% |
| 3 | 50 / 50 | 100% |
| 4 | 48 / 50 | 96% |
| 5 | 49 / 50 | 98% |
| 6 | 48 / 50 | 96% |
| 7 | 50 / 50 | 100% |
| 8 | 48 / 50 | 96% |
| 9 | 50 / 50 | 100% |

Overall:

493 / 500

Success rate:

98.6%

Released reported Spatial-Pro result:

99.6%

Difference:

- 5 fewer successful rollouts
- 1.0 percentage point lower

## Worker Results

- GPU0: 150 / 150 = 100.0%
- GPU1: 147 / 150 = 98.0%
- GPU2: 98 / 100 = 98.0%
- GPU3: 98 / 100 = 98.0%

## Smoke Test

Before the full evaluation, a 10-task × 1-trial smoke test was run.

Result:

10 / 10 successful

This verified the full pipeline:

LIBERO environment
→ third-person RGB
→ wrist RGB
→ proprio
→ language instruction
→ VLA
→ action head
→ 8×7 action chunk
→ simulator
→ success detection
→ rollout video

## Notes

The evaluation used the released `LIBERO-Spatial-Pro` checkpoint rather than
a model trained locally.

TensorFlow GPU warnings were present during startup but did not affect PyTorch
GPU inference.

LIBERO dataset-path warnings were also non-fatal.

The current result should be treated as the local reproduction baseline for
future self-trained checkpoints.
