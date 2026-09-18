"""Stateless formal-experiment helpers; no models, datasets or distributed setup."""

import json
import math
from numbers import Real
import os
from pathlib import Path
import tempfile

import torch


DEFAULT_HYBRID_DATASET_KEY = "libero_spatial_no_noops"
HYBRID_DATASET_CONFIGS = {
    "libero_spatial_no_noops": {"experiment": "hybrid_spatial_formal_v1"},
    "libero_object_no_noops": {"experiment": "hybrid_object_formal_v1"},
    "libero_goal_no_noops": {"experiment": "hybrid_goal_formal_v1"},
    "libero_10_no_noops": {"experiment": "hybrid_10_formal_v1"},
}


def formal_experiment_name(dataset_key):
    if not isinstance(dataset_key, str) or dataset_key not in HYBRID_DATASET_CONFIGS:
        raise ValueError(f"Unsupported Hybrid formal dataset key: {dataset_key}")
    return HYBRID_DATASET_CONFIGS[dataset_key]["experiment"]


def _positive_int(name, value):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _positive_real(name, value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def hybrid_learning_rate(global_step, *, base_lr, decay_step=None, decay_factor=0.1):
    """LR for the NEXT update, given the number of already completed updates."""
    if type(global_step) is not int or global_step < 0:
        raise ValueError("global_step must be a nonnegative integer")
    _positive_real("base_lr", base_lr)
    _positive_real("decay_factor", decay_factor)
    if decay_factor > 1:
        raise ValueError("decay_factor must be <= 1")
    if decay_step is not None:
        _positive_int("decay_step", decay_step)
    return float(base_lr * (decay_factor if decay_step is not None and global_step >= decay_step else 1))


def set_optimizer_learning_rate(optimizer, lr):
    _positive_real("lr", lr)
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def effective_flow_head_learning_rate(base_lr, flow_head_lr=None):
    _positive_real("learning_rate", base_lr)
    result = base_lr if flow_head_lr is None else flow_head_lr
    _positive_real("flow_head_learning_rate", result)
    return float(result)


def set_group_learning_rates(groups, *, vlm_lr, flow_head_lr):
    _positive_real("vlm_lr", vlm_lr)
    _positive_real("flow_head_lr", flow_head_lr)
    rates = {"vlm": float(vlm_lr), "flow_head": float(flow_head_lr)}
    for group in groups:
        if not isinstance(group.get("lr_role"), str) or group["lr_role"] not in rates:
            raise ValueError(f"Missing or unknown optimizer lr_role: {group.get('lr_role')}")
    for group in groups:
        group["lr"] = rates[group["lr_role"]]


def set_optimizer_learning_rates(optimizer, *, vlm_lr, flow_head_lr):
    set_group_learning_rates(optimizer.param_groups, vlm_lr=vlm_lr, flow_head_lr=flow_head_lr)


def normalization_metadata(statistics):
    """The exact four float32 vectors used by HybridZScoreNormalizer."""
    result = {}
    for kind, dim in (("action", 7), ("proprio", 8)):
        result[kind] = {}
        for name in ("mean", "std"):
            value = torch.as_tensor(statistics[kind][name], dtype=torch.float32).detach().cpu()
            if value.shape != (dim,) or not torch.isfinite(value).all() or (name == "std" and (value < 0).any()):
                raise ValueError(f"Invalid {kind}.{name}: expected finite [{dim}], std >= 0")
            result[kind][name] = value.tolist()
    return result


def validate_normalization_metadata(expected, actual, *, atol=1e-5, rtol=1e-5):
    """Evaluation-only numeric compatibility; training resume still uses exact metadata."""
    vectors = []
    for label, metadata in (("expected", expected), ("actual", actual)):
        if not isinstance(metadata, dict) or set(metadata) != {"action", "proprio"}:
            raise ValueError(f"{label} normalization metadata requires exactly action and proprio")
        converted = {}
        for kind, dim in (("action", 7), ("proprio", 8)):
            fields = metadata[kind]
            if not isinstance(fields, dict) or set(fields) != {"mean", "std"}:
                raise ValueError(f"{label} {kind} requires exactly mean and std")
            for field in ("mean", "std"):
                try:
                    tensor = torch.as_tensor(fields[field], dtype=torch.float32, device="cpu").detach()
                except (TypeError, ValueError, RuntimeError) as error:
                    raise ValueError(f"Invalid {label} {kind}.{field}") from error
                if tensor.shape != (dim,) or not torch.isfinite(tensor).all():
                    raise ValueError(f"Invalid {label} {kind}.{field}: expected finite [{dim}]")
                converted[kind, field] = tensor
        vectors.append(converted)
    for (kind, field), expected_tensor in vectors[0].items():
        actual_tensor = vectors[1][kind, field]
        if not torch.allclose(expected_tensor, actual_tensor, atol=atol, rtol=rtol):
            difference = (expected_tensor - actual_tensor).abs().max().item()
            raise ValueError(f"Normalization metadata mismatch: {kind}.{field}, max_abs_diff={difference}")


def build_formal_metadata(*, source_splits, statistics, max_steps, world_size=4, local_batch_size=1,
                          dataset_key=DEFAULT_HYBRID_DATASET_KEY,
                          base_learning_rate=1e-6, lr_decay_step=30000, lr_decay_factor=.1,
                          flow_head_learning_rate=None,
                          image_aug=True, shuffle_buffer_size=10000, save_every=10000, log_every=20, seed=7):
    hybrid_learning_rate(0, base_lr=base_learning_rate, decay_step=lr_decay_step, decay_factor=lr_decay_factor)
    for name, value in (("max_steps", max_steps), ("shuffle_buffer_size", shuffle_buffer_size),
                        ("save_every", save_every), ("log_every", log_every)):
        _positive_int(name, value)
    _positive_int("local_batch_size", local_batch_size)
    if type(world_size) is not int or world_size not in (1, 4):
        raise ValueError("Training requires world_size=1 or 4")
    if (len(source_splits) != world_size or any(not isinstance(s, str) or not s for s in source_splits)
            or len(set(source_splits)) != world_size):
        raise ValueError("Require one distinct ordered source-split string per rank")
    if type(image_aug) is not bool or type(seed) is not int or seed < 0:
        raise ValueError("image_aug must be bool; seed must be a nonnegative integer")
    head_lr = effective_flow_head_learning_rate(base_learning_rate, flow_head_learning_rate)
    extra = {"flow_head_learning_rate": head_lr} if head_lr != base_learning_rate else {}
    return dict(experiment=formal_experiment_name(dataset_key), dataset_key=dataset_key, **extra,
                world_size=world_size, local_batch_size=local_batch_size,
                effective_global_batch_size=world_size * local_batch_size,
                action_horizon=10, action_dim=7, proprio_dim=8, optimizer="AdamW",
                base_learning_rate=float(base_learning_rate), lr_policy="single_step_decay",
                lr_decay_step=lr_decay_step, lr_decay_factor=float(lr_decay_factor),
                gradient_accumulation=1, scheduler=None, gradient_clipping=None, warmup_steps=0,
                image_aug=image_aug, shuffle_buffer_size=shuffle_buffer_size, source_splits=list(source_splits),
                normalization_statistics=normalization_metadata(statistics),
                max_steps=max_steps, save_every=save_every, log_every=log_every, seed=seed)


def checkpoint_name(global_step):
    _positive_int("global_step", global_step)
    return f"step-{global_step:08d}.pt"


def checkpoint_due(global_step, *, max_steps, save_every):
    for name, value in (("global_step", global_step), ("max_steps", max_steps), ("save_every", save_every)):
        _positive_int(name, value)
    return global_step <= max_steps and (global_step % save_every == 0 or global_step == max_steps)


def atomic_write_json(path, value, *, overwrite=False):
    path = Path(path)
    if not overwrite and os.path.lexists(path):
        raise FileExistsError(path)
    encoded = json.dumps(value, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if not overwrite and os.path.lexists(path):
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def prepare_run_manifest(run_dir, metadata, *, resume=False):
    run_dir = Path(run_dir)
    path = run_dir / "run_config.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != metadata:
            raise ValueError("Existing run_config.json is incompatible")
    if not resume and ((run_dir / "train.jsonl").exists()
                       or ((run_dir / "checkpoints").is_dir() and any((run_dir / "checkpoints").iterdir()))):
        raise FileExistsError("Fresh run directory already contains training outputs; use resume or a new directory")
    if not path.exists():
        atomic_write_json(path, metadata)
