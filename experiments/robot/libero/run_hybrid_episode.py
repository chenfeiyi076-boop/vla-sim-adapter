"""Explicit Hybrid-only entry point: one Spatial task, one episode, random flow head.

Run from repository root with python -m experiments.robot.libero.run_hybrid_episode.
Uses a native VLM, local HF configuration assets and original RLDS statistics.
The official VLA evaluator and its policy dispatch are not modified.
"""

import argparse
from collections import deque
from pathlib import Path

import numpy as np
import time


def run_single_episode(cfg, env, description, policy, initial_state):
    """Same settling/queue/done semantics as run_episode; runtime errors propagate.

    Keep this small opt-in loop separate because the official run_episode catches
    exceptions and dispatches exclusively to the legacy VLA policy. No retries,
    task/trial loop, or success-rate aggregation is performed.
    """
    from experiments.robot.libero.run_libero_eval import (
        TASK_MAX_STEPS, get_libero_dummy_action, get_image_resize_size,
        prepare_observation, process_action,
    )

    profile = getattr(policy, "profile", None)
    def timed(name, operation):
        if profile is None:
            return operation()
        start = time.monotonic()
        result = operation()
        profile[name] = profile.get(name, 0.) + time.monotonic() - start
        return result

    start = time.monotonic()
    env.reset()
    obs = env.set_init_state(initial_state) if initial_state is not None else env.get_observation()
    if profile is not None:
        profile["reset"] = profile.get("reset", 0.) + time.monotonic() - start
    actions = deque()
    resize_size = get_image_resize_size(cfg)
    policy_calls = action_steps = 0
    success = False
    for t in range(TASK_MAX_STEPS[cfg.task_suite_name] + cfg.num_steps_wait):
        if t < cfg.num_steps_wait:
            obs, _, _, _ = timed("env_step", lambda: env.step(get_libero_dummy_action(cfg.model_family)))
            continue
        # Debug reference reproduces the original eager preprocessing order.
        if getattr(policy, "reference_rng", False):
            observation, _ = timed("prepare_observation", lambda: prepare_observation(obs, resize_size))
        if not actions:
            if not getattr(policy, "reference_rng", False):
                observation, _ = timed("prepare_observation", lambda: prepare_observation(obs, resize_size))
            chunk = timed("policy", lambda: policy(observation, description))
            if not isinstance(chunk, np.ndarray) or chunk.shape != (10, 7) or not np.isfinite(chunk).all():
                raise ValueError("Hybrid policy must return finite canonical actions [10,7]")
            # Preserve evaluator's execution window (default 8), independently of H=10.
            actions.extend(chunk[:cfg.num_open_loop_steps])
            policy_calls += 1
        action = process_action(actions.popleft(), cfg.model_family)
        if getattr(policy, "debug_trace", None) is not None:
            policy.debug_trace["actions"].append(np.asarray(action).tolist())
        obs, _, done, _ = timed("env_step", lambda: env.step(action.tolist()))
        action_steps += 1
        if done:
            success = True
            break
    if action_steps == 0:
        raise RuntimeError("Episode did not execute any Hybrid actions")
    return dict(success=success, policy_calls=policy_calls, action_steps=action_steps)


def load_native_weights(encoder, native):
    """Same ordered native-to-HF renames as the validated finetune loading path."""
    replacements = (
        ("vision_backbone.dino_featurizer", "vision_backbone.featurizer"),
        ("vision_backbone.siglip_featurizer", "vision_backbone.fused_featurizer"),
        ("llm_backbone.llm", "language_model"),
        ("projector.projector.0", "projector.fc1"),
        ("projector.projector.2", "projector.fc2"),
        ("projector.projector.4", "projector.fc3"),
        ("gamma", "scale_factor"),
    )
    converted = {}
    for key, value in native.state_dict().items():
        for old, new in replacements:
            key = key.replace(old, new)
        converted[key] = value
    missing, unexpected = encoder.load_state_dict(converted, strict=False)
    critical_missing = [key for key in missing if key != "action_queries.weight"]
    if critical_missing or unexpected:
        raise ValueError(f"Invalid native VLM weights: missing={critical_missing}, unexpected={unexpected}")


def load_policy(vlm_path, hf_config, statistics, device, num_steps, center_crop):
    from transformers import AutoTokenizer
    from prismatic.models import load
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import PrismaticForConditionalGeneration
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    from prismatic.models.flow_action_head import SimVLAFlowActionHead
    from experiments.robot.libero.hybrid_policy import HybridLiberoPolicy, load_hybrid_normalizer

    normalizer = load_hybrid_normalizer(statistics)
    config = OpenVLAConfig.from_pretrained(hf_config, local_files_only=True)
    if config.text_config.hidden_size != 896 or config.text_config.model_type != "qwen2":
        raise ValueError("Expected the Prismatic Qwen2.5-0.5B backbone (hidden_size=896)")
    if (config.image_sizes != [224, 224] or not config.use_fused_vision_backbone
            or config.vision_backbone_id != "dinosiglip-vit-so-224px"):
        raise ValueError("Expected fused DINOv2/SigLIP at 224x224")
    # Explicit local classes avoid get_vla's checkpoint config/source-file rewrites.
    native = load(str(vlm_path), hf_token="", load_for_training=True)
    encoder = PrismaticForConditionalGeneration(config)
    load_native_weights(encoder, native)
    del native
    encoder.vision_backbone.set_num_images_in_input(2)
    encoder = encoder.to(device).eval()
    processor = PrismaticProcessor(
        PrismaticImageProcessor.from_pretrained(hf_config, local_files_only=True),
        AutoTokenizer.from_pretrained(hf_config, local_files_only=True, trust_remote_code=False),
    )
    head = SimVLAFlowActionHead().to(device).eval()
    return HybridLiberoPolicy(encoder, processor, head, normalizer,
                             center_crop=center_crop, num_steps=num_steps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, choices=["hybrid"])
    parser.add_argument("--vlm-path", required=True, type=Path)
    parser.add_argument("--hf-config", required=True, type=Path)
    parser.add_argument("--statistics", required=True, type=Path)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--initial-state-index", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if not args.vlm_path.is_dir() or not args.hf_config.is_dir() or not args.statistics.is_file():
        parser.error("Provide existing native VLM and HF config directories and exact RLDS statistics JSON")
    if args.num_steps < 1:
        parser.error("--num-steps must be positive")

    from experiments.robot.libero.run_libero_eval import (
        GenerateConfig, TaskSuite, benchmark, get_libero_env, load_initial_states, set_seed_everywhere,
    )

    cfg = GenerateConfig(task_suite_name=TaskSuite.LIBERO_SPATIAL, seed=args.seed, num_trials_per_task=1)
    set_seed_everywhere(cfg.seed)
    suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
    if not 0 <= args.task_id < suite.n_tasks:
        parser.error("--task-id is outside LIBERO-Spatial")
    states, _ = load_initial_states(cfg, suite, args.task_id)
    if not 0 <= args.initial_state_index < len(states):
        parser.error("--initial-state-index is outside the task's initial states")
    policy = load_policy(args.vlm_path, args.hf_config, args.statistics, args.device, args.num_steps, cfg.center_crop)
    env, description = get_libero_env(suite.get_task(args.task_id), cfg.model_family, resolution=cfg.env_img_res)
    try:
        result = run_single_episode(cfg, env, description, policy, states[args.initial_state_index])
        print({"policy": "hybrid (random/untrained)", "task_id": args.task_id, "episodes": 1, **result})
    finally:
        env.close()


if __name__ == "__main__":
    main()
