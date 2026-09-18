"""Versioned Hybrid training-state checkpoints for trusted local files only.

Never use torch.load on untrusted checkpoints. No RNG, iterator, scheduler,
scaler or epoch state is part of this contract. Distributed coordination is
the caller's responsibility; only rank zero should call save.
"""

import copy
import os
from pathlib import Path
import tempfile

import torch


def _valid_step(value):
    if type(value) is not int or value < 0:
        raise ValueError("global_step must be a nonnegative int, not bool")


def _module_and_layout(training_model, optimizer):
    module = getattr(training_model, "module", training_model)
    if not all(isinstance(getattr(module, name, None), torch.nn.Module) for name in ("encoder", "flow_head")):
        raise ValueError("Training model must contain encoder and flow_head modules")
    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError("Hybrid checkpoint version 1 requires AdamW")
    names = {id(p): name for name, p in module.named_parameters()}
    selected = {id(p) for p in module.parameters() if p.requires_grad}
    actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
    if not selected or len(actual) != len(selected) or set(actual) != selected:
        raise ValueError("Optimizer must contain exactly selected trainable parameters, without duplicates/exclusions")
    layout = []
    for group in optimizer.param_groups:
        name = group.get("name")
        if not isinstance(name, str) or not name or name in [g["name"] for g in layout]:
            raise ValueError("Optimizer groups require unique nonempty names")
        layout.append({"name": name, "parameters": [names[id(p)] for p in group["params"]]})
        if "lr_role" in group:
            prefix = {"vlm": "encoder.", "flow_head": "flow_head."}.get(group["lr_role"])
            if prefix is None or not all(names[id(p)].startswith(prefix) for p in group["params"]):
                raise ValueError("Optimizer lr_role does not match parameter ownership")
    return module, layout


def _cpu_snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        result = type(value)((key, _cpu_snapshot(item)) for key, item in value.items())
        if hasattr(value, "_metadata"):
            result._metadata = _cpu_snapshot(value._metadata)
        return result
    if isinstance(value, list):
        return [_cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_snapshot(item) for item in value)
    return copy.deepcopy(value)


def _validate_optimizer_state(state, layout, global_step):
    if not isinstance(state, dict) or not isinstance(state.get("state"), dict) or not isinstance(state.get("param_groups"), list):
        raise ValueError("Malformed optimizer state")
    groups = state["param_groups"]
    if len(groups) != len(layout):
        raise ValueError("Optimizer group structure mismatch")
    ids = []
    for group, expected in zip(groups, layout):
        if (not isinstance(group, dict) or group.get("name") != expected["name"]
                or not isinstance(group.get("params"), list) or len(group["params"]) != len(expected["parameters"])):
            raise ValueError("Optimizer group structure mismatch")
        ids.extend(group["params"])
    if any(type(key) is not int for key in ids) or len(ids) != len(set(ids)) or set(state["state"]) - set(ids):
        raise ValueError("Malformed optimizer parameter IDs")
    for key in ids:
        entry = state["state"].get(key)
        if entry is None and global_step == 0:
            continue
        if not isinstance(entry, dict) or not {"step", "exp_avg", "exp_avg_sq"} <= entry.keys():
            raise ValueError("Missing AdamW state/step/moments for selected parameter")
        step = torch.as_tensor(entry["step"])
        if step.numel() != 1 or not torch.isfinite(step).all() or step.item() != global_step:
            raise ValueError("AdamW state step must equal checkpoint global_step")


def save_hybrid_checkpoint(path, training_model, optimizer, *, global_step, metadata, overwrite=False):
    """Write full detached CPU snapshots via one same-directory temporary file."""
    path = Path(path)
    if not overwrite and os.path.lexists(path):
        raise FileExistsError(f"Checkpoint already exists: {path}")
    _valid_step(global_step)
    if type(metadata) is not dict:
        raise ValueError("metadata must be a plain dictionary")
    module, layout = _module_and_layout(training_model, optimizer)
    optimizer_state = optimizer.state_dict()
    _validate_optimizer_state(optimizer_state, layout, global_step)
    payload = _cpu_snapshot(dict(format_version=1, global_step=global_step,
        encoder=module.encoder.state_dict(), flow_head=module.flow_head.state_dict(),
        optimizer=optimizer_state, optimizer_parameter_names=layout, metadata=metadata))
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if not overwrite and os.path.lexists(path):
            raise FileExistsError(f"Checkpoint appeared during save: {path}")
        os.replace(temporary, path)
    finally:
        # Only our exact temporary file; no glob or directory cleanup.
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_hybrid_checkpoint(path, training_model, optimizer, *, expected_metadata=None):
    """Restore a trusted project checkpoint; return only small version/step/metadata."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {"format_version", "global_step", "encoder", "flow_head", "optimizer",
                "optimizer_parameter_names", "metadata"}
    if type(payload) is not dict or not required <= payload.keys():
        raise ValueError("Malformed Hybrid checkpoint: required schema keys missing")
    if type(payload["format_version"]) is not int or payload["format_version"] != 1:
        raise ValueError("Unsupported Hybrid checkpoint format_version")
    _valid_step(payload["global_step"])
    if type(payload["metadata"]) is not dict:
        raise ValueError("Checkpoint metadata must be a plain dictionary")
    module, layout = _module_and_layout(training_model, optimizer)
    if payload["optimizer_parameter_names"] != layout:
        raise ValueError("Optimizer parameter-name layout mismatch (group/name/order)")
    if expected_metadata is not None:
        if type(expected_metadata) is not dict:
            raise ValueError("expected_metadata must be a plain dictionary")
        for key, value in expected_metadata.items():
            if key not in payload["metadata"] or payload["metadata"][key] != value:
                raise ValueError(f"Checkpoint metadata mismatch: {key}")
        if "base_learning_rate" in expected_metadata:
            expected_head = expected_metadata.get("flow_head_learning_rate", expected_metadata["base_learning_rate"])
            saved_head = payload["metadata"].get("flow_head_learning_rate", payload["metadata"].get("base_learning_rate"))
            if saved_head != expected_head:
                raise ValueError("Checkpoint metadata mismatch: flow_head_learning_rate")
    _validate_optimizer_state(payload["optimizer"], layout, payload["global_step"])
    # Layout/name/order were validated above. Old optimizer states lack lr_role;
    # restore it from the matching current group, never infer roles from group index.
    for saved, current in zip(payload["optimizer"]["param_groups"], optimizer.param_groups):
        if "lr_role" in current:
            if "lr_role" in saved and saved["lr_role"] != current["lr_role"]:
                raise ValueError("Checkpoint optimizer lr_role mismatch")
            saved["lr_role"] = current["lr_role"]
    # All obvious compatibility failures above precede any parameter mutation.
    module.encoder.load_state_dict(payload["encoder"], strict=True)
    module.flow_head.load_state_dict(payload["flow_head"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    _, restored_layout = _module_and_layout(training_model, optimizer)
    if restored_layout != layout:
        raise ValueError("Optimizer layout changed during restore")
    _validate_optimizer_state(optimizer.state_dict(), layout, payload["global_step"])
    return dict(format_version=1, global_step=payload["global_step"], metadata=copy.deepcopy(payload["metadata"]))


def load_hybrid_model_checkpoint(path, encoder, flow_head, *, expected_metadata=None):
    """Evaluation-only strict model load from a trusted local project checkpoint.

    No optimizer is constructed or restored. Never load untrusted checkpoint files.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {"format_version", "global_step", "encoder", "flow_head", "metadata"}
    if type(payload) is not dict or not required <= payload.keys():
        raise ValueError("Malformed Hybrid model checkpoint: required keys missing")
    if type(payload["format_version"]) is not int or payload["format_version"] != 1:
        raise ValueError("Unsupported Hybrid checkpoint format_version")
    _valid_step(payload["global_step"])
    if type(payload["metadata"]) is not dict:
        raise ValueError("Checkpoint metadata must be a plain dictionary")
    if expected_metadata is not None:
        if type(expected_metadata) is not dict:
            raise ValueError("expected_metadata must be a plain dictionary")
        for key, value in expected_metadata.items():
            if key not in payload["metadata"] or payload["metadata"][key] != value:
                raise ValueError(f"Checkpoint metadata mismatch: {key}")
    encoder.load_state_dict(payload["encoder"], strict=True)
    flow_head.load_state_dict(payload["flow_head"], strict=True)
    return dict(format_version=1, global_step=payload["global_step"], metadata=copy.deepcopy(payload["metadata"]))
