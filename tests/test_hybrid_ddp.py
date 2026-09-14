"""CPU-only architecture/diagnostic tests; collectives are fakes, not NCCL proof."""

import ast
from collections import OrderedDict
from contextlib import nullcontext
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from test_hybrid_step import ROOT, inputs, standalone, training


@pytest.fixture
def wrapper(training, monkeypatch):
    monkeypatch.setitem(sys.modules, "prismatic.training.hybrid_step", training)
    return standalone("prismatic/training/hybrid_ddp.py")


def test_outer_forward_delegates(wrapper, inputs, monkeypatch):
    encoder, head, batch, normalizer, _ = inputs
    calls = []
    def loss(*args):
        calls.append(args)
        return torch.tensor(2., requires_grad=True), {"tag": "delegated"}
    monkeypatch.setattr(wrapper, "hybrid_flow_loss", loss)
    model = wrapper.HybridFlowTrainingModule(encoder, head)
    result = model(batch, normalizer)
    assert calls == [(encoder, head, batch, normalizer)]
    assert result[1] == {"tag": "delegated"}
    assert dict(model.named_children()) == {"encoder": encoder, "flow_head": head}
    assert "encoder.action_queries.weight" in model.state_dict()
    assert "flow_head.action_decoder.weight" in model.state_dict()


def test_exact_phase5a_policy_and_differentiable_loss(wrapper, training, inputs):
    encoder, head, batch, normalizer, _ = inputs
    expected = training.hybrid_parameter_groups(encoder, head)
    expected_ids = {id(p) for group in expected for p in group["params"]}
    model = wrapper.HybridFlowTrainingModule(encoder, head)
    assert {id(p) for p in model.parameters() if p.requires_grad} == expected_ids
    loss, _ = model(batch, normalizer)
    loss.backward()
    assert encoder.projector.weight.grad.abs().sum() > 0
    assert head.action_decoder.weight.grad.abs().sum() > 0
    for module in (encoder.action_queries, encoder.proprio_projector, encoder.action_head, encoder.language_model.lm_head):
        assert all(not p.requires_grad and p.grad is None for p in module.parameters())


def test_wrapper_imports_only_training_loss_and_torch():
    tree = ast.parse((ROOT / "prismatic/training/hybrid_ddp.py").read_text())
    imports = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert imports == {"torch", "prismatic.training.hybrid_step"}
    forward = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "forward")
    assert len(forward.body) == 1 and isinstance(forward.body[0], ast.Return)
    assert forward.body[0].value.func.id == "hybrid_flow_loss"


@pytest.fixture
def smoke(monkeypatch):
    module = standalone("vla-scripts/hybrid_ddp_smoke.py")
    def gather(outputs, local):
        for output in outputs:
            output.copy_(local)
    def gather_object(outputs, local):
        outputs[:] = [local.copy() for _ in outputs]
    monkeypatch.setattr(module, "dist", SimpleNamespace(
        get_world_size=lambda: 4, get_rank=lambda: 0, all_gather=gather,
        all_gather_object=gather_object, all_reduce=lambda tensor, op: None,
        ReduceOp=SimpleNamespace(MIN="min")))
    return module


def test_sync_slice_detects_different_rank(smoke):
    assert smoke.synchronized_slice(torch.ones(128), "gradient", "cpu") == 0
    def unequal(outputs, local):
        for output in outputs:
            output.copy_(local)
        outputs[-1][0] += .01
    smoke.dist.all_gather = unequal
    with pytest.raises(RuntimeError, match="not synchronized"):
        smoke.synchronized_slice(torch.ones(128), "gradient", "cpu")


def test_smoke_enters_outer_forward_and_steps_once(wrapper, training, inputs, smoke, monkeypatch):
    encoder, head, batch, normalizer, _ = inputs
    encoder.projector = nn.Sequential(OrderedDict(fc1=encoder.projector))
    model = wrapper.HybridFlowTrainingModule(encoder, head)
    groups = training.hybrid_parameter_groups(encoder, head)
    batch = {key: value[:1] if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

    class FakeDDP(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.calls = 0
        def forward(self, *args):
            self.calls += 1
            return self.module(*args)

    class CountAdamW(torch.optim.AdamW):
        steps = 0
        zeros = 0
        def zero_grad(self, set_to_none):
            assert set_to_none
            self.zeros += 1
            return super().zero_grad(set_to_none=set_to_none)
        def step(self):
            self.steps += 1
            return super().step()

    ddp = FakeDDP(model)
    optimizer = CountAdamW(groups, lr=1e-6)
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 123)
    reports = smoke.one_step(ddp, batch, normalizer, optimizer, groups, "cpu")
    assert ddp.calls == optimizer.steps == optimizer.zeros == 1
    assert len(reports) == 4
    for report in reports:
        assert report["encoder_changed"] and report["flow_head_changed"]
        assert report["action_queries_no_grad"] and report["excluded_no_grad"]
        assert report["encoder_gradient_max_diff"] == report["flow_head_gradient_max_diff"] == 0
        assert report["encoder_parameter_max_diff"] == report["flow_head_parameter_max_diff"] == 0


def test_missing_trainable_grad_is_rejected(wrapper, training, inputs, smoke):
    encoder, head, batch, normalizer, _ = inputs
    # Represents an unused vision tail still selected by the unchanged 5A policy.
    encoder.vision_backbone.register_parameter("unused_tail", nn.Parameter(torch.ones(1)))
    model = wrapper.HybridFlowTrainingModule(encoder, head)
    groups = training.hybrid_parameter_groups(encoder, head)
    loss, _ = model(batch, normalizer)
    loss.backward()
    with pytest.raises(RuntimeError, match="unused_tail"):
        smoke.check_gradients(model, groups, "cpu")


def test_runner_ddp_configuration_and_no_bypass():
    tree = ast.parse((ROOT / "vla-scripts/hybrid_ddp_smoke.py").read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    ddp = next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id == "DDP")
    kwargs = {item.arg: item.value for item in ddp.keywords}
    assert ast.literal_eval(kwargs["find_unused_parameters"]) is False
    assert ast.literal_eval(kwargs["gradient_as_bucket_view"]) is True
    assert ddp.args[0].id == "model"
    assert kwargs["device_ids"].elts[0].id == kwargs["output_device"].id == "local_rank"
    step = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "one_step")
    step_calls = [n for n in ast.walk(step) if isinstance(n, ast.Call)]
    assert sum(isinstance(n.func, ast.Name) and n.func.id == "ddp_model" for n in step_calls) == 1
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert "encode_observation" not in attrs
    assert "sample_actions_euler" not in attrs
    assert "DistributedSampler" not in {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
