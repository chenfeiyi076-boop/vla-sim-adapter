"""Phase 5A: one Hybrid loss/optimizer step, independent of rollout and training loops.

Supply a batch from PaddedCollatorForHybridFlow and its dataset's
HybridZScoreNormalizer. Models must already be on the same single device.
The caller controls autocast; this module never casts model weights globally.

Single-GPU use with an already loaded encoder/head and one real collated RLDS
batch (H=10, A=7, P=8; raw canonical values, not legacy-normalized values):

    encoder.to("cuda")
    flow_head.to("cuda")
    optimizer = make_hybrid_smoke_optimizer(encoder, flow_head)
    loss, diagnostics = hybrid_optimizer_step(encoder, flow_head, batch, normalizer, optimizer)

Use HybridZScoreNormalizer(dataset.dataset_statistics["libero_spatial_no_noops"])
from that same RLDS dataset. This utility does not load models/data or invent
statistics. For custom autocast/backward handling use hybrid_flow_loss directly.
"""

import torch

from prismatic.models.flow_action_head import compute_flow_matching_loss, sample_flow_matching_inputs


def _hybrid_vision_parameters(vision_backbone):
    """Exclude only the structural tail outside Prismatic's intermediate output.

    Each TIMM featurizer returns second-to-last-block features, without its
    final norm or attention pooling. Other modules stay eligible; synthetic
    backbones without these featurizers retain all their parameters.
    """
    excluded = set()
    for name in ("featurizer", "fused_featurizer"):
        featurizer = getattr(vision_backbone, name, None)
        if featurizer is None:
            continue
        blocks = getattr(featurizer, "blocks", None)
        if blocks is None or len(blocks) == 0:
            raise ValueError(f"{name} must have transformer blocks for intermediate-layer parameter selection")
        tail = (blocks[-1], getattr(featurizer, "norm", None), getattr(featurizer, "attn_pool", None))
        excluded.update(id(p) for module in tail if module is not None for p in module.parameters())
    return [p for p in vision_backbone.parameters() if id(p) not in excluded]


def _hybrid_parameters(encoder, flow_head):
    # Select by parameter identity so tied Qwen input/output embeddings remain
    # trainable exactly once, while an untied, unused vocabulary head is excluded.
    used = {id(p) for p in _hybrid_vision_parameters(encoder.vision_backbone)}
    used.update(id(p) for module in (encoder.projector,
                                encoder.get_input_embeddings(), encoder.get_decoder())
                for p in module.parameters())
    encoder_params = [p for p in encoder.parameters() if id(p) in used]
    flow_params = list(flow_head.parameters())
    if not encoder_params or not flow_params:
        raise ValueError("Hybrid requires encoder and flow-head parameters")
    if used.intersection(id(p) for p in flow_params):
        raise ValueError("Encoder and flow head must not share parameters")
    return encoder_params, flow_params


def hybrid_parameter_groups(encoder, flow_head):
    """Enable all observation/flow parameters; disable unrelated encoder modules.

    Excluded parameters are retained, with requires_grad=False and stale grads
    cleared. No legacy forward implementation or module is changed/deleted.
    Both groups use the same AdamW defaults for this smoke, without a LR policy.
    """
    encoder_params, flow_params = _hybrid_parameters(encoder, flow_head)
    used = {id(p) for p in encoder_params}
    for parameter in encoder.parameters():
        parameter.requires_grad_(id(parameter) in used)
        if id(parameter) not in used:
            parameter.grad = None
    for parameter in flow_params:
        parameter.requires_grad_(True)
    return [{"name": "encoder", "params": encoder_params}, {"name": "flow_head", "params": flow_params}]


def make_hybrid_smoke_optimizer(encoder, flow_head, *, lr=1e-4):
    """One AdamW over the exact Hybrid parameter set; no scheduler/state loading."""
    return torch.optim.AdamW(hybrid_parameter_groups(encoder, flow_head), lr=lr)


def _check(name, tensor, shape):
    if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {shape}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must be finite")


def hybrid_flow_loss(encoder, flow_head, batch, normalizer):
    """Return differentiable scalar loss and detached scalar diagnostics.

    Does not set train/eval, clear gradients, backward, or step an optimizer.
    Input tensors are never modified. Normalization retains the existing fp32
    statistics arithmetic; encoder/features retain their dtype/autocast path.
    """
    if (flow_head.action_horizon, flow_head.action_dim, flow_head.proprio_dim) != (10, 7, 8):
        raise ValueError("Hybrid requires H=10, A=7, P=8")
    parameter = next(encoder.vision_backbone.parameters())
    device = parameter.device
    if any(p.device != device for model in (encoder, flow_head) for p in model.parameters()):
        raise ValueError("Encoder and flow head must be on the same single device")
    ids = batch["input_ids"].to(device=device)
    mask = batch["attention_mask"].to(device=device)
    pixels = batch["pixel_values"].to(device=device, dtype=parameter.dtype)
    actions = batch["actions"].to(device=device)
    proprio = batch["proprio"].to(device=device)
    if ids.ndim != 2 or min(ids.shape) < 1 or ids.dtype != torch.long:
        raise ValueError("input_ids must be nonempty int64 [B,L]")
    B = ids.shape[0]
    _check("attention_mask", mask, ids.shape)
    if not ((mask == 0) | (mask == 1)).all() or not mask.bool().any(dim=1).all():
        raise ValueError("attention_mask must be binary with valid tokens in each sample")
    if pixels.ndim != 4 or pixels.shape[0] != B or any(d == 0 for d in pixels.shape):
        raise ValueError("pixel_values must be nonempty [B,C,H,W]")
    _check("pixel_values", pixels, pixels.shape)
    _check("actions", actions, (B, 10, 7))
    _check("proprio", proprio, (B, 8))
    actions = normalizer.normalize_action(actions)
    proprio = normalizer.normalize_proprio(proprio)
    _check("normalized actions", actions, (B, 10, 7))
    _check("normalized proprio", proprio, (B, 8))
    encoded = encoder.encode_observation(input_ids=ids, attention_mask=mask.bool(), pixel_values=pixels)
    features, feature_mask = encoded["features"], encoded["feature_mask"]
    if features.ndim != 3 or features.shape[1] < 1:
        raise ValueError("features must be nonempty [B,T,C]")
    _check("features", features, (B, features.shape[1], flow_head.vlm_hidden_dim))
    _check("feature_mask", feature_mask, features.shape[:2])
    if feature_mask.dtype != torch.bool or not feature_mask.any(dim=1).all():
        raise ValueError("feature_mask must be boolean with valid tokens in each sample")
    sampled = sample_flow_matching_inputs(actions)
    prediction = flow_head(features, sampled["x_t"], proprio, sampled["t"], feature_mask)
    _check("predicted velocity", prediction, (B, 10, 7))
    loss = compute_flow_matching_loss(prediction, sampled["target_velocity"])
    _check("loss", loss, ())
    return loss, {"loss": loss.detach(), "t_mean": sampled["t"].mean().detach(),
                  "velocity_rms": prediction.detach().square().mean().sqrt(),
                  "batch_size": B}


def hybrid_optimizer_step(encoder, flow_head, batch, normalizer, optimizer):
    """Single smoke step; use make_hybrid_smoke_optimizer first.

    Caller may wrap this in bf16 autocast, or use the pure loss function for
    custom precision handling. No GradScaler/fp16 scaling policy is introduced.
    Non-finite gradients abort before step; missing gradients in either model
    also fail instead of silently accepting a disconnected path.
    """
    encoder_params, flow_params = _hybrid_parameters(encoder, flow_head)
    expected = {id(p) for p in encoder_params + flow_params}
    actual = [p for group in optimizer.param_groups for p in group["params"]]
    if len(actual) != len(expected) or {id(p) for p in actual} != expected:
        raise ValueError("Optimizer must contain exactly the Hybrid parameter set, without duplicates")
    encoder.train()
    flow_head.train()
    optimizer.zero_grad(set_to_none=True)
    loss, diagnostics = hybrid_flow_loss(encoder, flow_head, batch, normalizer)
    loss.backward()
    for name, parameters in (("encoder", encoder_params), ("flow_head", flow_params)):
        grads = [p.grad for p in parameters if p.grad is not None]
        if not grads or not any(torch.count_nonzero(g).item() for g in grads):
            raise RuntimeError(f"No nonzero {name} gradients")
        if any(not torch.isfinite(g).all() for g in grads):
            raise RuntimeError(f"Non-finite {name} gradients; optimizer step aborted")
        diagnostics[f"{name}_grad_norm"] = torch.stack([g.detach().float().norm() for g in grads]).norm()
    optimizer.step()
    return loss.detach(), diagnostics
