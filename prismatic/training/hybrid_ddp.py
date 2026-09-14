"""Outer training module: the only differentiable entry point for Hybrid DDP."""

from torch import nn

from prismatic.training.hybrid_step import hybrid_flow_loss, hybrid_parameter_groups


class HybridFlowTrainingModule(nn.Module):
    def __init__(self, encoder, flow_head):
        super().__init__()
        self.encoder = encoder
        self.flow_head = flow_head
        # Apply the unchanged Phase-5A policy BEFORE DDP builds its reducer.
        hybrid_parameter_groups(self.encoder, self.flow_head)

    def forward(self, batch, normalizer):
        return hybrid_flow_loss(self.encoder, self.flow_head, batch, normalizer)
