"""Observation-only Hybrid policy; outputs canonical RLDS actions, not env actions."""

import json
from pathlib import Path

import torch
import time


from experiments.robot.libero.hybrid_suite_config import DEFAULT_TASK_SUITE, read_suite_statistics


def load_hybrid_normalizer(path, task_suite=DEFAULT_TASK_SUITE):
    """Require the original RLDS dataset_statistics.json, with the exact dataset key."""
    from prismatic.vla.hybrid_normalization import HybridZScoreNormalizer

    return HybridZScoreNormalizer(read_suite_statistics(path, task_suite))


def check_tensor(name, value, shape):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {shape}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")


class HybridLiberoPolicy:
    """Single-observation bridge using existing camera preprocessing and Qwen prompt.

    Encoder and flow head must be on the same device. The encoder may be bf16;
    the random flow head and Euler state use float32. No action/gripper clipping
    or conversion occurs here: process_action() owns the env conversion.
    """

    def __init__(self, encoder, processor, flow_head, normalizer, *, center_crop=True, num_steps=10):
        if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 1:
            raise ValueError("num_steps must be a positive integer")
        if (flow_head.action_horizon, flow_head.action_dim, flow_head.proprio_dim) != (10, 7, 8):
            raise ValueError("Hybrid requires H=10, action_dim=7, proprio_dim=8")
        self.encoder = encoder.eval()
        self.processor = processor
        self.flow_head = flow_head.eval()
        self.normalizer = normalizer
        self.center_crop = center_crop
        self.num_steps = num_steps
        self.generator = None
        self.reference_rng = False
        self.profile = None
        self.debug_trace = None

    def begin_episode(self, seed, *, reference=False, profile=False, debug=False):
        self.reference_rng = reference
        device = next(self.encoder.parameters()).device
        self.generator = None if reference else torch.Generator(device=device).manual_seed(seed)
        self.profile = {} if profile else None
        self.debug_trace = {"noise_hashes": [], "actions": []} if debug else None

    @torch.inference_mode()
    def __call__(self, observation, task_description):
        from experiments.robot.openvla_utils import prepare_images_for_vla
        from prismatic.models.backbones.llm.prompting import QwenPromptBuilder
        from prismatic.models.flow_sampling import sample_actions_euler

        parameter = next(self.encoder.parameters())
        device, dtype = parameter.device, parameter.dtype
        images = prepare_images_for_vla(
            [observation["full_image"], observation["wrist_image"]], self
        )
        builder = QwenPromptBuilder("openvla")
        builder.add_turn("human", f"What action should the robot take to {task_description.lower()}?")
        prompt = builder.get_prompt()
        primary, wrist = [self.processor(prompt, image) for image in images]
        ids = primary["input_ids"].to(device)
        mask = primary["attention_mask"].to(device)
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0 or ids.dtype != torch.long:
            raise ValueError("encoder input_ids must be nonempty int64 [1,L]")
        check_tensor("encoder attention_mask", mask, ids.shape)
        if not ((mask == 0) | (mask == 1)).all() or not mask.any():
            raise ValueError("encoder attention_mask must be binary with valid tokens")
        for view in (primary, wrist):
            check_tensor("encoder per-view pixels", view["pixel_values"], (1, 6, 224, 224))
        pixels = torch.cat([primary["pixel_values"], wrist["pixel_values"]], dim=1).to(device, dtype)
        check_tensor("encoder pixel_values", pixels, (1, 12, 224, 224))
        state = torch.as_tensor(observation["state"], device=device, dtype=torch.float32)
        check_tensor("observation state", state, (8,))
        proprio = state.unsqueeze(0)
        check_tensor("proprio", proprio, (1, 8))
        encoded = self.encoder.encode_observation(
            input_ids=ids, attention_mask=mask.bool(), pixel_values=pixels
        )
        features = encoded["features"]
        feature_mask = encoded["feature_mask"]
        if features.ndim != 3 or features.shape[1] == 0:
            raise ValueError("encoder features must be nonempty [1,T,896]")
        check_tensor("encoder features", features, (1, features.shape[1], 896))
        check_tensor("encoder feature_mask", feature_mask, features.shape[:2])
        if feature_mask.dtype != torch.bool or not feature_mask.any():
            raise ValueError("encoder feature_mask must be boolean with valid tokens")
        proprio = self.normalizer.normalize_proprio(proprio)
        check_tensor("normalized proprio", proprio, (1, 8))
        sampling = dict(num_steps=self.num_steps, generator=self.generator)
        if self.debug_trace is not None:
            from experiments.robot.libero.hybrid_eval_results import array_hash
            noise = torch.randn((1, 10, 7), device=proprio.device, dtype=proprio.dtype, generator=self.generator)
            self.debug_trace["noise_hashes"].append(array_hash(noise.cpu().numpy()))
            sampling["initial_noise"] = noise
        if self.profile is not None:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.monotonic()
        normalized = sample_actions_euler(self.flow_head, features.float(), proprio, feature_mask, **sampling)
        if self.profile is not None:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            self.profile["flow_sampling"] = self.profile.get("flow_sampling", 0.) + time.monotonic() - start
        check_tensor("normalized actions", normalized, (1, 10, 7))
        canonical = self.normalizer.denormalize_action(normalized)
        check_tensor("denormalized actions", canonical, (1, 10, 7))
        return canonical[0].float().cpu().numpy()
