"""Standalone SimVLA concat action head and normalized-space flow utilities.

Adapted from SimVLA models/transformer_smolvlm.py and
models/modeling_smolvlm_vla.py, commit
32700d0ad8991996e123e4b685abe370ce6e9aab (Apache-2.0).
Source: https://github.com/LUOyk1999/SimVLA
License: ../../third_party/licenses/SimVLA-LICENSE.txt
Changes: concat-only implementation, configurable dimensions/dropout, strict
input validation, VLM padding masks, and independently testable flow helpers.
No dataset normalization, gripper convention, or inference integration lives here.
"""

import math
from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 100) -> torch.Tensor:
    """SimVLA sinusoidal embedding of unscaled t [B]; result is [B, dim]."""
    if t.ndim != 1 or not t.is_floating_point():
        raise ValueError("t must be a floating tensor with shape [B]")
    if dim < 2 or max_period <= 0:
        raise ValueError("dim must be >= 2 and max_period must be positive")
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=t.dtype, device=t.device) / half
    )
    args = t[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class Mlp(nn.Module):
    """Reference GELU(tanh) MLP with dropout after activation and fc2."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU(approximate="tanh")
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


class Attention(nn.Module):
    """Non-causal self-attention; valid_mask True means an allowed key/value."""

    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = hasattr(F, "scaled_dot_product_attention")
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        # Reference concat blocks use zero output-projection dropout.
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        B, S, C = x.shape
        if valid_mask.shape != (B, S) or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be a boolean tensor of exact shape [B, S]")
        if valid_mask.device != x.device:
            raise ValueError("valid_mask and attention inputs must be on the same device")
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # Explicit [B,1,1,S] expansion broadcasts only over heads and queries.
        # SDPA boolean True = ALLOW (opposite of MultiheadAttention's
        # key_padding_mask True = IGNORE). Fallback masks the inverse below.
        allowed_keys = valid_mask[:, None, None, :]
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask=allowed_keys,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=False,
            )
        else:
            scores = (q * self.scale) @ k.transpose(-2, -1)
            scores = scores.masked_fill(~allowed_keys, float("-inf"))
            x = self.attn_drop(scores.softmax(dim=-1)) @ v
        x = x.transpose(1, 2).reshape(B, S, C)
        return self.proj_drop(self.proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, dropout)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), valid_mask)
        return x + self.mlp(self.norm2(x))


def _basic_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class SimVLAFlowActionHead(nn.Module):
    """SimVLA concat head. Small defaults; no dependency on a VLM implementation.

    The caller supplies normalized actions/proprio and last hidden VLM tokens.
    vlm_proj is the sole C_vlm -> hidden_dim projection. Large can be selected
    with hidden_dim=1024, depth=24, num_heads=16.
    """

    def __init__(
        self,
        vlm_hidden_dim: int = 896,
        hidden_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        action_horizon: int = 10,
        action_dim: int = 7,
        proprio_dim: int = 8,
        time_dim: int = 32,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        dimensions = (vlm_hidden_dim, hidden_dim, depth, num_heads, action_horizon,
                      action_dim, proprio_dim, max_seq_len)
        if any(d <= 0 for d in dimensions) or time_dim < 2:
            raise ValueError("dimensions, depth and horizon must be positive; time_dim must be >= 2")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not 0 <= dropout < 1 or mlp_ratio <= 0 or int(hidden_dim * mlp_ratio) < 1:
            raise ValueError("dropout must be in [0,1) and mlp_ratio must produce a positive MLP width")
        self.vlm_hidden_dim = vlm_hidden_dim
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        self.time_dim = time_dim
        self.max_seq_len = max_seq_len
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])
        self.vlm_proj = nn.Linear(vlm_hidden_dim, hidden_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        nn.init.normal_(self.pos_emb, std=0.02)
        self.norm = nn.LayerNorm(hidden_dim)
        self.action_encoder = nn.Linear(action_dim + proprio_dim + time_dim, hidden_dim)
        self.action_decoder = nn.Linear(hidden_dim, action_dim)
        self.apply(_basic_init)

    def forward(
        self,
        vlm_features: torch.Tensor,
        noisy_action: torch.Tensor,
        proprio: torch.Tensor,
        t: torch.Tensor,
        feature_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """[B,T,C], [B,H,A], [B,P], [B], optional bool [B,T] -> [B,H,A]."""
        if vlm_features.ndim != 3 or vlm_features.shape[-1] != self.vlm_hidden_dim:
            raise ValueError(f"vlm_features must have shape [B,T,{self.vlm_hidden_dim}]")
        B, T, _ = vlm_features.shape
        H = self.action_horizon
        if B == 0:
            raise ValueError("batch size must be positive")
        for name, value, shape in (
            ("noisy_action", noisy_action, (B, H, self.action_dim)),
            ("proprio", proprio, (B, self.proprio_dim)),
            ("t", t, (B,)),
        ):
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
        if any(not value.is_floating_point() for value in (vlm_features, noisy_action, proprio, t)):
            raise ValueError("features, actions, proprio and t must be floating tensors")
        if any(value.device != vlm_features.device for value in (noisy_action, proprio, t)):
            raise ValueError("features, actions, proprio and t must be on the same device")
        if H + T > self.max_seq_len:
            raise ValueError(f"H + T = {H} + {T} = {H + T} exceeds max_seq_len={self.max_seq_len}")
        if feature_mask is None:
            feature_mask = torch.ones(B, T, dtype=torch.bool, device=vlm_features.device)
        elif feature_mask.shape != (B, T) or feature_mask.dtype != torch.bool:
            raise ValueError(f"feature_mask must be boolean with exact shape {(B, T)}")
        elif feature_mask.device != vlm_features.device:
            raise ValueError("feature_mask must be on the same device as vlm_features")

        time_tokens = timestep_embedding(t, self.time_dim)[:, None].expand(B, H, self.time_dim)
        proprio_tokens = proprio[:, None].expand(B, H, self.proprio_dim)
        action_tokens = torch.cat([noisy_action, proprio_tokens, time_tokens], dim=-1)
        x = self.action_encoder(action_tokens)  # [B,H,A+P+time_dim] -> [B,H,D]
        # Replace padding before projection as well, so even non-finite padding
        # cannot contaminate matrix products. No valid feature is changed.
        features = vlm_features.masked_fill(~feature_mask[:, :, None], 0)
        x = torch.cat([x, self.vlm_proj(features)], dim=1)  # ACTION first, VLM second
        valid_mask = torch.cat([
            torch.ones(B, H, dtype=torch.bool, device=x.device), feature_mask
        ], dim=1)
        x = x + self.pos_emb[:, :H + T]
        for block in self.blocks:
            x = block(x, valid_mask)
        return self.action_decoder(self.norm(x[:, :H]))


def sample_flow_matching_inputs(action: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Sample SimVLA flow inputs for already-normalized floating actions [B,H,A].

    Randomness uses PyTorch's RNG (torch.manual_seed for reproducible tests).
    Returns t [B] and noise/x_t/target_velocity [B,H,A]. No preprocessing.
    """
    if action.ndim != 3 or not action.is_floating_point() or any(d == 0 for d in action.shape):
        raise ValueError("action must be a nonempty floating tensor with shape [B,H,A]")
    beta = torch.distributions.Beta(
        torch.tensor(1.5, device=action.device), torch.tensor(1.0, device=action.device)
    )
    t = beta.sample((action.shape[0],)) * 0.999 + 0.001
    noise = torch.randn_like(action)
    t3 = t.view(-1, 1, 1)
    return {
        "t": t,
        "noise": noise,
        "x_t": t3 * noise + (1 - t3) * action,
        "target_velocity": noise - action,
    }


def compute_flow_matching_loss(pred_velocity: torch.Tensor, target_velocity: torch.Tensor) -> torch.Tensor:
    """Full-dimension scalar MSE; no gripper weighting, masks, or normalization."""
    if pred_velocity.shape != target_velocity.shape or pred_velocity.ndim != 3:
        raise ValueError("pred_velocity and target_velocity must have identical [B,H,A] shapes")
    if any(d == 0 for d in pred_velocity.shape):
        raise ValueError("velocity tensors must be nonempty")
    if not pred_velocity.is_floating_point() or not target_velocity.is_floating_point():
        raise ValueError("velocity tensors must be floating point")
    if pred_velocity.device != target_velocity.device:
        raise ValueError("velocity tensors must be on the same device")
    return torch.mean(torch.square(pred_velocity - target_velocity))
