"""CPU sampler tests using toy vector fields and the real tiny flow head."""

import ast
import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]


def load_file(relative):
    # Avoid the package's eager optional VLM/RLDS dependencies.
    spec = importlib.util.spec_from_file_location("euler_test_module", ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sample = load_file("prismatic/models/flow_sampling.py").sample_actions_euler
flow = load_file("prismatic/models/flow_action_head.py")


class ConstantVelocity(nn.Module):
    action_horizon = 10
    action_dim = 7

    def __init__(self):
        super().__init__()
        self.times, self.states, self.masks = [], [], []

    def forward(self, features, x, proprio, t, feature_mask):
        assert not torch.is_grad_enabled()
        assert not self.training
        self.times.append(t.clone())
        self.states.append(x.clone())
        self.masks.append(feature_mask)
        return torch.full_like(x, 2)


def inputs(T=9):
    return torch.randn(2, T, 11), torch.randn(2, 8), torch.ones(2, T, dtype=torch.bool)


@pytest.mark.parametrize("steps", [1, 4, 10, 17])
def test_constant_velocity_direction_and_endpoint(steps):
    head = ConstantVelocity()
    args = inputs()
    args[-1][:, -2:] = False
    noise = torch.randn(2, 10, 7, requires_grad=True)
    saved = noise.detach().clone()
    result = sample(head, *args, num_steps=steps, initial_noise=noise)
    torch.testing.assert_close(result, saved - 2, rtol=1e-5, atol=1e-6)
    assert result.shape == (2, 10, 7) and torch.isfinite(result).all()
    assert result.requires_grad is False
    assert len(head.times) == steps
    times = torch.stack(head.times)[:, 0]
    torch.testing.assert_close(times, 1 - torch.arange(steps) / steps)
    assert times[0] == 1 and (times > 0).all()
    assert (times[1:] < times[:-1]).all()
    assert abs(times[-1].item() - 1 / steps) < 1e-6  # final update lands at 0
    states = head.states + [result]
    for old, new in zip(states, states[1:]):
        torch.testing.assert_close(new - old, torch.full_like(old, -2 / steps), rtol=1e-5, atol=1e-6)
    assert all(mask is args[-1] for mask in head.masks)
    assert head.training  # Restored to the caller's mode.
    torch.testing.assert_close(noise, saved, rtol=0, atol=0)


def test_four_step_times():
    head = ConstantVelocity()
    sample(head, *inputs(), num_steps=4)
    assert [t[0].item() for t in head.times] == [1., .75, .5, .25]


@pytest.fixture
def real_head():
    torch.manual_seed(10)
    return flow.SimVLAFlowActionHead(vlm_hidden_dim=11, hidden_dim=24, depth=2, num_heads=3)


@pytest.mark.parametrize("T", [5, 37])
def test_real_head_fixed_noise_seed_and_no_grad(real_head, T):
    args = inputs(T)
    args[0].requires_grad_(True)
    args[-1][:, -2:] = False
    noise = torch.randn(2, 10, 7)
    # Starts in training mode with dropout enabled; sampler handles evaluation.
    first = sample(real_head, *args, initial_noise=noise)
    second = sample(real_head, *args, initial_noise=noise)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert first.shape == (2, 10, 7) and torch.isfinite(first).all()
    assert not first.requires_grad and args[0].grad is None
    assert all(p.grad is None for p in real_head.parameters())
    assert real_head.training
    seeded1 = sample(real_head, *args, generator=torch.Generator().manual_seed(42))
    seeded2 = sample(real_head, *args, generator=torch.Generator().manual_seed(42))
    torch.testing.assert_close(seeded1, seeded2, rtol=0, atol=0)
    different = sample(real_head, *args, generator=torch.Generator().manual_seed(43))
    assert not torch.allclose(seeded1, different)


def test_explicit_noise_cast_and_generator_not_consumed():
    args = inputs()
    noise = torch.randn(2, 10, 7, dtype=torch.float64)
    original = noise.clone()
    generator = torch.Generator().manual_seed(9)
    state = generator.get_state().clone()
    result = sample(ConstantVelocity(), *args, initial_noise=noise, generator=generator)
    assert result.dtype == args[1].dtype and result.device == args[1].device
    torch.testing.assert_close(result, noise.float() - 2, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(noise, original, rtol=0, atol=0)
    assert torch.equal(state, generator.get_state())


def test_autocast_and_original_module_flags(real_head):
    real_head.blocks[0].eval()  # Mixed modes must survive the sampler.
    flags = [m.training for m in real_head.modules()]
    args = inputs()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = sample(real_head, *args, num_steps=3)
    assert result.dtype == args[1].dtype and torch.isfinite(result).all()
    assert [m.training for m in real_head.modules()] == flags
    assert all(p.dtype == torch.float32 for p in real_head.parameters())


def test_float64_model_not_forced_to_float32(real_head):
    real_head.double().eval()
    features, proprio, mask = inputs()
    result = sample(real_head, features.double(), proprio.double(), mask, num_steps=2)
    assert result.dtype == torch.float64 and torch.isfinite(result).all()
    assert not real_head.training


def test_dimensions_read_from_head():
    head = ConstantVelocity()
    head.action_horizon, head.action_dim = 3, 4
    features, proprio, _ = inputs()
    result = sample(head, features, proprio, num_steps=2)
    assert result.shape == (2, 3, 4)
    assert len(head.masks) == 2 and all(mask is None for mask in head.masks)


@pytest.mark.parametrize("steps", [0, -1, 2.5, True])
def test_bad_step_count(steps):
    with pytest.raises(ValueError, match="positive integer"):
        sample(ConstantVelocity(), *inputs(), num_steps=steps)


@pytest.mark.parametrize("noise", [torch.zeros(1, 10, 7), torch.zeros(2, 7, 10)])
def test_bad_noise_shape(noise):
    with pytest.raises(ValueError, match="initial_noise must have shape"):
        sample(ConstantVelocity(), *inputs(), initial_noise=noise)


def test_nonfloating_noise():
    with pytest.raises(ValueError, match="floating dtype"):
        sample(ConstantVelocity(), *inputs(), initial_noise=torch.zeros(2, 10, 7, dtype=torch.long))


def test_velocity_broadcast_rejected_and_mode_restored():
    class BrokenHead(ConstantVelocity):
        def forward(self, *args):
            return torch.zeros(1, 10, 7)

    head = BrokenHead()
    with pytest.raises(ValueError, match="velocity.*exact shape"):
        sample(head, *inputs())
    assert head.training


def test_training_utilities_remain_independent():
    tree = ast.parse((ROOT / "prismatic/models/flow_action_head.py").read_text(encoding="utf-8"))
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for name in ("sample_flow_matching_inputs", "compute_flow_matching_loss"):
        assert name in functions
        assert not any(isinstance(n, ast.Name) and n.id == "sample_actions_euler" for n in ast.walk(functions[name]))
    action = torch.randn(2, 10, 7)
    sampled = flow.sample_flow_matching_inputs(action)
    torch.testing.assert_close(sampled["target_velocity"], sampled["noise"] - action)
    t = sampled["t"][:, None, None]
    torch.testing.assert_close(sampled["x_t"], t * sampled["noise"] + (1 - t) * action)
    pred = torch.randn_like(action, requires_grad=True)
    loss = flow.compute_flow_matching_loss(pred, sampled["target_velocity"])
    torch.testing.assert_close(loss, ((pred - sampled["target_velocity"]) ** 2).mean())
    loss.backward()
    assert torch.isfinite(pred.grad).all()
