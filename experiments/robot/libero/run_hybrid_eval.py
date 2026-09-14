"""Trained-checkpoint Hybrid LIBERO-Spatial evaluation; official action path unchanged."""

import argparse
import json
from pathlib import Path

import torch


def load_trained_policy(args, *, center_crop=True):
    from experiments.robot.libero.run_hybrid_episode import load_policy
    from prismatic.training.hybrid_checkpoint import load_hybrid_model_checkpoint
    from prismatic.training.hybrid_formal import normalization_metadata, validate_normalization_metadata

    with args.statistics.open(encoding="utf-8") as stream:
        statistics = json.load(stream)
    normalization = normalization_metadata(statistics["libero_spatial_no_noops"])
    # Existing load_policy uses HybridLiberoPolicy and load_hybrid_normalizer.
    policy = load_policy(args.vlm_path, args.hf_config, args.statistics, args.device, args.num_steps, center_crop)
    info = load_hybrid_model_checkpoint(args.checkpoint, policy.encoder, policy.flow_head,
        expected_metadata=dict(experiment="hybrid_spatial_formal_v1", dataset_key="libero_spatial_no_noops",
            action_horizon=10, action_dim=7, proprio_dim=8))
    if "normalization_statistics" not in info["metadata"]:
        raise ValueError("Checkpoint metadata missing normalization_statistics")
    validate_normalization_metadata(info["metadata"]["normalization_statistics"], normalization)
    if args.expected_step is not None and info["global_step"] != args.expected_step:
        raise ValueError(f"Checkpoint global_step {info['global_step']} != expected {args.expected_step}")
    policy.encoder.eval()
    policy.flow_head.eval()
    return policy, info


def evaluate_tasks(cfg, suite, policy, *, trials_per_task, task_id=None, seed=7):
    from experiments.robot.libero.run_libero_eval import get_libero_env, load_initial_states
    from experiments.robot.libero.run_hybrid_episode import run_single_episode

    if type(trials_per_task) is not int or trials_per_task < 1:
        raise ValueError("trials_per_task must be positive")
    if cfg.num_open_loop_steps != 8:
        raise ValueError("Hybrid evaluation requires the validated 8-step execution window")
    if task_id is not None and (type(task_id) is not int or not 0 <= task_id < suite.n_tasks):
        raise ValueError("task_id is outside LIBERO-Spatial")
    ids = list(range(suite.n_tasks)) if task_id is None else [task_id]
    # Check every selected task's official states before beginning any rollout.
    states = {}
    for index in ids:
        states[index], _ = load_initial_states(cfg, suite, index)
        if trials_per_task > len(states[index]):
            raise ValueError(f"Task {index} has only {len(states[index])} official initial states")
    results = []
    for index in ids:
        env, description = get_libero_env(suite.get_task(index), cfg.model_family, resolution=cfg.env_img_res)
        successes = calls = action_steps = 0
        try:
            for trial in range(trials_per_task):
                # Seed only torch policy randomness; do not replace official states.
                torch.manual_seed(seed + index * 100000 + trial)
                episode = run_single_episode(cfg, env, description, policy, states[index][trial])
                successes += int(episode["success"])
                calls += episode["policy_calls"]
                action_steps += episode["action_steps"]
        finally:
            env.close()
        results.append(dict(task_id=index, task_description=description, successes=successes,
                            trials=trials_per_task, success_rate=successes / trials_per_task,
                            total_policy_calls=calls, total_action_steps=action_steps))
    total_successes = sum(r["successes"] for r in results)
    total_trials = sum(r["trials"] for r in results)
    return dict(task_results=results, total_successes=total_successes, total_trials=total_trials,
                overall_success_rate=total_successes / total_trials)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "vlm-path", "hf-config", "statistics"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--trials-per-task", type=int, default=50)
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-step", type=int)
    args = parser.parse_args()
    if args.num_steps < 1 or args.trials_per_task < 1 or (args.expected_step is not None and args.expected_step < 0):
        parser.error("Euler steps/trials must be positive; expected-step must be nonnegative")
    if args.output is not None and args.output.exists():
        parser.error("Output file already exists; choose a new path")
    from experiments.robot.libero.run_libero_eval import GenerateConfig, TaskSuite, benchmark, set_seed_everywhere
    from prismatic.training.hybrid_formal import atomic_write_json

    cfg = GenerateConfig(task_suite_name=TaskSuite.LIBERO_SPATIAL, num_open_loop_steps=8, seed=args.seed)
    set_seed_everywhere(args.seed)
    policy, info = load_trained_policy(args, center_crop=cfg.center_crop)
    suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
    result = dict(checkpoint=str(args.checkpoint), global_step=info["global_step"],
                  num_euler_steps=args.num_steps, num_open_loop_steps=cfg.num_open_loop_steps,
                  **evaluate_tasks(cfg, suite, policy, trials_per_task=args.trials_per_task, task_id=args.task_id, seed=args.seed))
    if args.output is not None:
        atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
