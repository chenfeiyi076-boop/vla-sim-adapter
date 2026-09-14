"""SimVLA z-score rule applied to canonical VLA action/proprio values.

Supply one dataset's RLDS statistics, not the outer mapping of dataset names.
The old action normalization mask is intentionally ignored: all seven action
dimensions, including the canonical VLA gripper, use the same rule. Inputs must
not already have undergone BOUNDS/BOUNDS_Q99 normalization.
"""

import torch


class HybridZScoreNormalizer:
    """Float32 statistics, explicit per-call device transfer, no clipping.

    Statistics are owned CPU copies; operations move temporary float32 copies
    to the input device and return float32 tensors. No global/default device or
    mutable module device is involved. Input gradients are preserved.
    """

    def __init__(self, dataset_statistics):
        for kind, dim in (("action", 7), ("proprio", 8)):
            for stat in ("mean", "std"):
                value = torch.as_tensor(dataset_statistics[kind][stat], dtype=torch.float32).detach().cpu().clone()
                if value.shape != (dim,):
                    raise ValueError(f"{kind}.{stat} must have shape [{dim}], got {tuple(value.shape)}")
                if not torch.isfinite(value).all() or (stat == "std" and (value < 0).any()):
                    raise ValueError(f"{kind}.{stat} must be finite; standard deviations must be nonnegative")
                setattr(self, f"{kind}_{stat}", value)

    def _apply(self, x, kind, inverse=False):
        dim = self.action_mean.numel() if kind == "action" else self.proprio_mean.numel()
        if not isinstance(x, torch.Tensor) or x.ndim < 1 or x.shape[-1] != dim:
            raise ValueError(f"{kind} must be a tensor with shape [...,{dim}]")
        if not x.is_floating_point():
            raise ValueError(f"{kind} must be floating point")
        mean = getattr(self, f"{kind}_mean").to(device=x.device, dtype=torch.float32)
        std = getattr(self, f"{kind}_std").to(device=x.device, dtype=torch.float32)
        x = x.float()
        if inverse:
            return x * (std + 1e-6) + mean
        return (x - mean) / (std + 1e-6)

    def normalize_action(self, x):
        return self._apply(x, "action")

    def normalize_proprio(self, x):
        return self._apply(x, "proprio")

    def denormalize_action(self, x):
        return self._apply(x, "action", inverse=True)
