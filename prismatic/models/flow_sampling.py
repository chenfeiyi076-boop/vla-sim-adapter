"""Normalized-space Euler inference for SimVLAFlowActionHead.

Uses SimVLA's noise-to-data convention (velocity = noise - action).
No VLM encoding, normalization, gripper processing, or action execution.
"""

from typing import Optional

import torch
from torch import nn


@torch.no_grad()
def sample_actions_euler(
    flow_head: nn.Module,
    vlm_features: torch.Tensor,
    proprio: torch.Tensor,
    feature_mask: Optional[torch.Tensor] = None,
    *,
    num_steps: int = 10,
    initial_noise: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Return normalized actions [B, head.action_horizon, head.action_dim].

    Features [B,T,C] and normalized proprio [B,P] must share a device. Noise,
    time tensors, Euler accumulation and output use proprio's floating dtype
    and device, as in the reference sampler. Supplied noise is detached,
    moved/cast to that device/dtype, and cloned; the caller's tensor is untouched.
    The optional generator must match the sampling device. With explicit noise
    it is unused and no random draw is made by this helper.

    Model weights, features and proprio are never cast here. Callers retain
    control of model/input precision and any enclosing autocast context; each
    predicted velocity is cast to the accumulator dtype before the update.
    Temporarily evaluates the head to disable dropout, then restores every
    module's original training flag, including on failure. No gradients are
    recorded. Do not concurrently train the same head while sampling.

    Exactly N calls at t_i = 1 - i/N (i=0,...,N-1), with dt=-1/N,
    land at t=0 after the last update. Integer-indexed times avoid cumulative
    Python floating-point drift; the model is not evaluated again at t=0.
    """
    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps <= 0:
        raise ValueError("num_steps must be a positive integer")
    if vlm_features.ndim != 3 or not vlm_features.is_floating_point() or vlm_features.shape[0] == 0:
        raise ValueError("vlm_features must be a nonempty floating [B,T,C] tensor")
    B = vlm_features.shape[0]
    if proprio.ndim != 2 or proprio.shape[0] != B or not proprio.is_floating_point():
        raise ValueError("proprio must be a floating [B,P] tensor matching the feature batch")
    if proprio.device != vlm_features.device:
        raise ValueError("proprio and vlm_features must share a device")
    if feature_mask is not None:
        if feature_mask.shape != vlm_features.shape[:2] or feature_mask.dtype != torch.bool:
            raise ValueError("feature_mask must be boolean with exact shape [B,T]")
        if feature_mask.device != vlm_features.device:
            raise ValueError("feature_mask must be on the same device as vlm_features")

    shape = (B, flow_head.action_horizon, flow_head.action_dim)
    if initial_noise is None:
        x = torch.randn(shape, device=proprio.device, dtype=proprio.dtype, generator=generator)
    else:
        if initial_noise.shape != shape:
            raise ValueError(f"initial_noise must have shape {shape}, got {tuple(initial_noise.shape)}")
        if not initial_noise.is_floating_point():
            raise ValueError("initial_noise must have a floating dtype")
        x = initial_noise.detach().to(device=proprio.device, dtype=proprio.dtype).clone()

    dt = -1.0 / num_steps
    training_flags = [(module, module.training) for module in flow_head.modules()]
    try:
        flow_head.eval()
        for step in range(num_steps):
            current_t = 1.0 + step * dt
            t_batch = torch.full((B,), current_t, device=x.device, dtype=x.dtype)
            velocity = flow_head(vlm_features, x, proprio, t_batch, feature_mask)
            if velocity.shape != shape or not velocity.is_floating_point():
                raise ValueError(f"flow_head velocity must be a floating tensor with exact shape {shape}")
            if velocity.device != x.device:
                raise ValueError("flow_head velocity must be on the same device as the noise")
            x = x + dt * velocity.to(dtype=x.dtype)
    finally:
        # Assign flags individually instead of recursive train(flag), preserving
        # any intentionally mixed train/eval configuration of the caller.
        for module, was_training in training_flags:
            module.training = was_training
    return x.detach()
