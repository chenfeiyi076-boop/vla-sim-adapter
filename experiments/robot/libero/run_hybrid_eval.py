"""Trained-checkpoint Hybrid LIBERO-Spatial evaluation; official action path unchanged."""

import argparse
import json
import sys
import time
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


def evaluate_tasks(cfg, suite, policy, **kwargs):
    from experiments.robot.libero.hybrid_eval_results import journal_lock
    with journal_lock(kwargs.get("partial")):
        return _evaluate_tasks(cfg, suite, policy, **kwargs)


def _evaluate_tasks(cfg, suite, policy, *, trials_per_task, task_id=None, seed=7,
                   task_ids=None, partial=None, manifest=None, resume=False,
                   worker_id=0, num_workers=1, reference=False, profile=False,
                   debug=False, stop_after=None):
    from experiments.robot.libero.run_libero_eval import get_libero_env, load_initial_states
    from experiments.robot.libero.run_hybrid_episode import run_single_episode
    from experiments.robot.libero.hybrid_eval_results import (
        EpisodeJournal, PlannedStop, aggregate, array_hash,
    )

    if type(trials_per_task) is not int or trials_per_task < 1:
        raise ValueError("trials_per_task must be positive")
    if cfg.num_open_loop_steps != 8:
        raise ValueError("Hybrid evaluation requires the validated 8-step execution window")
    if task_id is not None and (type(task_id) is not int or not 0 <= task_id < suite.n_tasks):
        raise ValueError("task_id is outside LIBERO-Spatial")
    ids = task_ids if task_ids is not None else (list(range(suite.n_tasks)) if task_id is None else [task_id])
    if not ids or len(set(ids)) != len(ids) or any(type(i) is not int or not 0 <= i < suite.n_tasks for i in ids):
        raise ValueError("Invalid selected task IDs")
    ids = sorted(ids)
    # Check every selected task's official states before beginning any rollout.
    states = {}
    for index in ids:
        states[index], _ = load_initial_states(cfg, suite, index)
        if trials_per_task > len(states[index]):
            raise ValueError(f"Task {index} has only {len(states[index])} official initial states")
    specs = []
    for index in ids:
        for trial in range(trials_per_task):
            specs.append(dict(global_episode_id=index * trials_per_task + trial, task_id=index,
                trial_id=trial, episode_seed=seed + index * 100000 + trial,
                initial_state_hash=array_hash(states[index][trial])))
            task = suite.get_task(index)
            if hasattr(task, "language"):
                specs[-1]["task_description"] = task.language
    manifest = dict(manifest or {}, selected_task_ids=ids, trials_per_task=trials_per_task,
                    base_seed=seed, rng_mode="global-reference" if reference else "episode-generator")
    journal = EpisodeJournal(partial, manifest, specs, resume=resume, worker_id=worker_id, num_workers=num_workers)
    start = time.monotonic()
    executed = 0
    env_creation = 0.
    for index in ids:
        pending = [s for s in specs if s["task_id"] == index and s["global_episode_id"] in journal.allowed
                   and s["global_episode_id"] not in journal.completed]
        if not pending:
            continue
        env_start = time.monotonic()
        env, description = get_libero_env(suite.get_task(index), cfg.model_family, resolution=cfg.env_img_res)
        env_creation += time.monotonic() - env_start
        try:
            for spec in pending:
                trial = spec["trial_id"]
                # Seed only torch policy randomness; do not replace official states.
                torch.manual_seed(spec["episode_seed"])
                if hasattr(policy, "begin_episode"):
                    policy.begin_episode(spec["episode_seed"], reference=reference, profile=profile, debug=debug)
                episode_start = time.monotonic()
                episode = run_single_episode(cfg, env, description, policy, states[index][trial])
                record = dict(spec, **episode, elapsed_sec=time.monotonic() - episode_start)
                record["task_description"] = description
                if profile and getattr(policy, "profile", None) is not None:
                    record["profile"] = dict(policy.profile)
                if debug and getattr(policy, "debug_trace", None) is not None:
                    record["debug"] = dict(policy.debug_trace)
                    record["debug"]["action_hash"] = array_hash(policy.debug_trace["actions"])
                journal.append(record)
                executed += 1
                elapsed = time.monotonic() - start
                task_rows = [r for r in journal.records if r["task_id"] == index]
                print(f"worker={worker_id} [{len(journal.completed)}/{len(journal.allowed)}] "
                      f"task_id={index} trial_id={trial} success={episode['success']} "
                      f"task_success_rate={sum(r['success'] for r in task_rows) / len(task_rows):.4f} "
                      f"overall_success_rate={sum(r['success'] for r in journal.records) / len(journal.records):.4f} "
                      f"policy_calls={episode['policy_calls']} action_steps={episode['action_steps']} "
                      f"episode_time={record['elapsed_sec']:.2f}s elapsed={elapsed:.2f}s "
                      f"ETA_estimate={elapsed / executed * (len(journal.allowed) - len(journal.completed)):.2f}s",
                      file=sys.stderr, flush=True)
                if stop_after is not None and executed >= stop_after:
                    raise PlannedStop("Smoke stopped after publishing completed episodes; no final JSON")
        finally:
            env.close()
    journal.require_complete()
    if profile:
        elapsed = time.monotonic() - start
        # Only this process segment: resumed historical rows are excluded.
        fresh = journal.records[-executed:] if executed else []
        totals = {key: sum(r.get("profile", {}).get(key, 0.) for r in fresh)
                  for key in ("reset", "prepare_observation", "policy", "flow_sampling", "env_step")}
        print(json.dumps(dict(profile=totals, env_creation_sec=env_creation, total_seconds=elapsed,
            episodes_per_sec=executed / max(elapsed, 1e-9),
            policy_calls_per_sec=sum(r["policy_calls"] for r in fresh) / max(elapsed, 1e-9),
            percent={key: 100 * value / max(elapsed, 1e-9) for key, value in totals.items()})),
            file=sys.stderr, flush=True)
    return aggregate(journal.records)


def build_manifest(args, cfg, policy, info):
    from experiments.robot.libero.hybrid_eval_results import file_hash
    from experiments.robot.robot_utils import get_image_resize_size
    from prismatic.training.hybrid_formal import normalization_metadata
    statistics = json.loads(args.statistics.read_text(encoding="utf-8"))["libero_spatial_no_noops"]
    return dict(schema_version=1, checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=file_hash(args.checkpoint), global_step=info["global_step"],
        task_suite_name=str(cfg.task_suite_name), checkpoint_metadata=info["metadata"],
        normalization_statistics=normalization_metadata(statistics),
        episode_seed_formula="base_seed + task_id * 100000 + trial_id",
        num_euler_steps=args.num_steps, num_open_loop_steps=cfg.num_open_loop_steps,
        resize_size=get_image_resize_size(cfg), env_img_res=cfg.env_img_res,
        num_steps_wait=cfg.num_steps_wait, center_crop=cfg.center_crop,
        model_family=cfg.model_family, max_action_steps=220,
        encoder_dtype=str(next(policy.encoder.parameters()).dtype),
        head_dtype=str(next(policy.flow_head.parameters()).dtype),
        vlm_path=str(args.vlm_path.resolve()), hf_config=str(args.hf_config.resolve()),
        hf_assets={str(p.relative_to(args.hf_config)): file_hash(p)
                   for p in sorted(args.hf_config.rglob("*")) if p.is_file()},
        debug=args.debug_trace, profiling=args.profile)


def main(argv=None, *, worker_id=0, num_workers=1):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("checkpoint", "vlm-path", "hf-config", "statistics"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--trials-per-task", type=int, default=50)
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--task-ids", type=int, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-step", type=int)
    parser.add_argument("--partial", type=Path, help="Atomic per-episode journal; defaults to OUTPUT.partial.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reference", action="store_true", help="Debug old global RNG + eager preprocessing")
    parser.add_argument("--debug-trace", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--stop-after", type=int, help="Smoke: publish N new episodes then exit without final JSON")
    args = parser.parse_args(argv)
    if args.task_id is not None and args.task_ids is not None:
        parser.error("Choose task-id or task-ids")
    if args.partial is None and args.output is not None:
        args.partial = args.output.with_name(args.output.name + ".partial.json")
    if (args.resume or args.stop_after is not None) and args.partial is None:
        parser.error("resume/stop-after requires a partial path or output")
    if args.stop_after is not None and args.stop_after < 1:
        parser.error("stop-after must be positive")
    if args.partial is not None and args.partial.exists() and not args.resume:
        parser.error("Partial already exists; use resume or a new path")
    if args.partial is not None and args.output is not None and args.partial.resolve() == args.output.resolve():
        parser.error("Partial and final paths must differ")
    if args.num_steps < 1 or args.trials_per_task < 1 or (args.expected_step is not None and args.expected_step < 0):
        parser.error("Euler steps/trials must be positive; expected-step must be nonnegative")
    if args.output is not None and args.output.exists():
        parser.error("Output file already exists; choose a new path")
    from experiments.robot.libero.run_libero_eval import GenerateConfig, TaskSuite, benchmark, set_seed_everywhere
    from prismatic.training.hybrid_formal import atomic_write_json

    cfg = GenerateConfig(task_suite_name=TaskSuite.LIBERO_SPATIAL, num_open_loop_steps=8, seed=args.seed)
    set_seed_everywhere(args.seed)
    load_start = time.monotonic()
    policy, info = load_trained_policy(args, center_crop=cfg.center_crop)
    if args.profile:
        print(f"model_load_sec={time.monotonic() - load_start:.3f}", file=sys.stderr, flush=True)
    suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
    manifest = build_manifest(args, cfg, policy, info) if args.partial is not None else None
    from experiments.robot.libero.hybrid_eval_results import PlannedStop
    try:
        result = dict(checkpoint=str(args.checkpoint), global_step=info["global_step"],
            num_euler_steps=args.num_steps, num_open_loop_steps=cfg.num_open_loop_steps,
            **evaluate_tasks(cfg, suite, policy, trials_per_task=args.trials_per_task, task_id=args.task_id,
                task_ids=args.task_ids, seed=args.seed, partial=args.partial, manifest=manifest,
                resume=args.resume, worker_id=worker_id, num_workers=num_workers, reference=args.reference,
                profile=args.profile, debug=args.debug_trace, stop_after=args.stop_after))
    except PlannedStop as error:
        print(str(error), file=sys.stderr, flush=True)
        return
    if args.output is not None:
        atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
