"""Phase 5A CPU tests: real encoder methods, tiny backbones, real flow/AdamW."""

import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Dict

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]


def standalone(path):
    spec = importlib.util.spec_from_file_location("hybrid_step_test_module", ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


flow = standalone("prismatic/models/flow_action_head.py")
norm = standalone("prismatic/vla/hybrid_normalization.py")


@pytest.fixture
def training(monkeypatch):
    # Import only the real flow dependency, without eager optional package imports.
    monkeypatch.setitem(sys.modules, "prismatic.models.flow_action_head", flow)
    return standalone("prismatic/training/hybrid_step.py")


def tiny_encoder():
    path = ROOT / "prismatic/extern/hf/modeling_prismatic.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PrismaticForConditionalGeneration")
    names = {"encode_observation", "get_input_embeddings", "get_decoder",
             "_process_vision_features", "_build_multimodal_attention"}
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {"torch": torch, "nn": nn, "Dict": Dict}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)
    Encoder = type("TinyPrismatic", (nn.Module,), {name: namespace[name] for name in names})

    class Decoder(nn.Linear):
        def forward(self, inputs_embeds, **kwargs):
            return SimpleNamespace(last_hidden_state=super().forward(inputs_embeds))

    class LM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(20, 12)
            self.decoder = Decoder(12, 12)
            self.lm_head = nn.Linear(12, 20, bias=False)

        def get_input_embeddings(self):
            return self.embedding

        def get_decoder(self):
            return self.decoder

        def forward(self, *args, **kwargs):
            raise AssertionError("Legacy vocabulary path is unused")

    encoder = Encoder()
    encoder.vision_backbone = nn.Sequential(nn.Flatten(2), nn.Linear(4, 6))
    encoder.projector = nn.Linear(6, 12)
    encoder.language_model = LM()
    encoder.action_queries = nn.Embedding(56, 12)
    encoder.proprio_projector = nn.Linear(8, 12)
    encoder.action_head = nn.Linear(12, 7)
    return encoder


@pytest.fixture
def inputs():
    torch.manual_seed(7)
    encoder = tiny_encoder()
    head = flow.SimVLAFlowActionHead(vlm_hidden_dim=12, hidden_dim=24, depth=1,
                                   num_heads=3, dropout=0, max_seq_len=32)
    batch = dict(input_ids=torch.tensor([[1, 2, 3], [4, 5, 0]]),
                 attention_mask=torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool),
                 pixel_values=torch.randn(2, 12, 2, 2), actions=torch.randn(2, 10, 7),
                 proprio=torch.randn(2, 8), dataset_names=["libero_spatial_no_noops"] * 2)
    stats = {kind: {"mean": torch.full((dim,), .7, requires_grad=True),
                    "std": torch.full((dim,), 1.3, requires_grad=True)}
             for kind, dim in (("action", 7), ("proprio", 8))}
    return encoder, head, batch, norm.HybridZScoreNormalizer(stats), stats


def test_loss_backward_both_models_and_batch_unchanged(training, inputs):
    encoder, head, batch, normalizer, stats = inputs
    before = {k: v.clone() for k, v in batch.items() if isinstance(v, torch.Tensor)}
    loss, diagnostics = training.hybrid_flow_loss(encoder, head, batch, normalizer)
    assert loss.shape == () and torch.isfinite(loss) and loss.requires_grad
    assert not diagnostics["loss"].requires_grad
    loss.backward()
    for module in (encoder.vision_backbone, encoder.projector, encoder.get_input_embeddings(),
                   encoder.get_decoder(), head):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert any(g.abs().sum() > 0 for g in gradients)
    assert encoder.action_queries.weight.grad is None
    for key, value in before.items():
        torch.testing.assert_close(batch[key], value, rtol=0, atol=0)
    for kind in stats:
        for name in stats[kind]:
            assert stats[kind][name].grad is None
            assert not getattr(normalizer, f"{kind}_{name}").requires_grad


def test_normalization_mask_and_exact_flow_equations(training, inputs, monkeypatch):
    encoder, head, batch, normalizer, _ = inputs
    captured = {}
    real_sample = flow.sample_flow_matching_inputs
    real_loss = flow.compute_flow_matching_loss

    def sample(action):
        torch.testing.assert_close(action, (batch["actions"] - .7) / 1.300001)
        state = torch.random.get_rng_state()
        result = real_sample(action)
        torch.random.set_rng_state(state)
        t = torch.distributions.Beta(torch.tensor(1.5), torch.tensor(1.)).sample((2,)) * .999 + .001
        eps = torch.randn_like(action)
        torch.testing.assert_close(result["t"], t)
        torch.testing.assert_close(result["noise"], eps)
        torch.testing.assert_close(result["x_t"], t[:, None, None] * eps + (1 - t[:, None, None]) * action)
        torch.testing.assert_close(result["target_velocity"], eps - action)
        captured.update(result)
        return result

    def mse(prediction, target):
        torch.testing.assert_close(target, captured["target_velocity"])
        loss = real_loss(prediction, target)
        torch.testing.assert_close(loss, (prediction - target).square().mean())
        captured["mse_called"] = True
        return loss

    def encoder_hook(module, args, kwargs):
        # The real method returns a dict and this exact mask must reach the head.
        encoded = original(**kwargs)
        captured["feature_mask"] = encoded["feature_mask"]
        return encoded

    original = encoder.encode_observation
    encoder.encode_observation = lambda **kwargs: encoder_hook(encoder, (), kwargs)
    def inspect_head(module, args):
        features, xt, proprio, t, mask = args
        assert features.requires_grad
        assert mask is captured["feature_mask"] and not mask[1, -1]
        torch.testing.assert_close(xt, captured["x_t"])
        torch.testing.assert_close(t, captured["t"])
        torch.testing.assert_close(proprio, (batch["proprio"] - .7) / 1.300001)
    head.register_forward_pre_hook(inspect_head)
    monkeypatch.setattr(training, "sample_flow_matching_inputs", sample)
    monkeypatch.setattr(training, "compute_flow_matching_loss", mse)
    training.hybrid_flow_loss(encoder, head, batch, normalizer)
    assert captured["mse_called"]


@pytest.mark.parametrize("autocast", [False, True])
def test_adamw_updates_both_and_excludes_legacy(training, inputs, autocast):
    encoder, head, batch, normalizer, _ = inputs
    encoder.action_queries.weight.grad = torch.ones_like(encoder.action_queries.weight)
    optimizer = training.make_hybrid_smoke_optimizer(encoder, head)
    before = {f"{root}.{name}": p.detach().clone() for root, model in (("encoder", encoder), ("head", head))
              for name, p in model.named_parameters()}
    dtypes = [p.dtype for p in encoder.parameters()]
    output_dtypes = []
    head.register_forward_hook(lambda module, args, output: output_dtypes.append(output.dtype))
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        loss, diagnostics = training.hybrid_optimizer_step(encoder, head, batch, normalizer, optimizer)
    assert torch.isfinite(loss)
    for root, model in (("encoder", encoder), ("head", head)):
        assert any(not torch.equal(p, before[f"{root}.{name}"]) for name, p in model.named_parameters())
    assert diagnostics["encoder_grad_norm"] > 0 and diagnostics["flow_head_grad_norm"] > 0
    assert dtypes == [p.dtype for p in encoder.parameters()]
    assert output_dtypes == [torch.bfloat16 if autocast else torch.float32]
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    for name, p in encoder.named_parameters():
        excluded = name.startswith(("action_queries.", "proprio_projector.", "action_head.", "language_model.lm_head."))
        assert (id(p) not in optimized) == excluded
        if excluded:
            assert not p.requires_grad and p.grad is None
            torch.testing.assert_close(p, before[f"encoder.{name}"], rtol=0, atol=0)


def test_tied_embeddings_included_once(training, inputs):
    encoder, head, *_ = inputs
    encoder.language_model.lm_head.weight = encoder.get_input_embeddings().weight
    groups = training.hybrid_parameter_groups(encoder, head)
    ids = [id(p) for group in groups for p in group["params"]]
    assert len(ids) == len(set(ids))
    assert id(encoder.get_input_embeddings().weight) in ids
    assert encoder.get_input_embeddings().weight.requires_grad


@pytest.mark.parametrize("depths", [(2, 3), (5, 4), (3, None)])
def test_structural_timm_tail_exclusions(training, inputs, depths):
    encoder, head, *_ = inputs

    def featurizer(depth, pool=False):
        model = nn.Module()
        model.blocks = nn.ModuleList([nn.Linear(6, 6) for _ in range(depth)])
        model.norm = nn.LayerNorm(6)
        model.patch_embed = nn.Linear(4, 6)
        # An unfamiliar module must not be filtered by speculative name rules.
        model.future_module = nn.Linear(6, 6)
        if pool:
            model.attn_pool = nn.Linear(6, 6)
        return model

    vision = nn.Module()
    vision.featurizer = featurizer(depths[0])
    if depths[1] is not None:
        vision.fused_featurizer = featurizer(depths[1], pool=True)
    encoder.vision_backbone = vision
    encoder.language_model.lm_head.weight = encoder.get_input_embeddings().weight
    tail = []
    earlier = []
    for model in vision.children():
        tail.extend(model.blocks[-1].parameters())
        tail.extend(model.norm.parameters())
        if hasattr(model, "attn_pool"):
            tail.extend(model.attn_pool.parameters())
        earlier.extend(model.blocks[:-1].parameters())
        earlier.extend(model.patch_embed.parameters())
        earlier.extend(model.future_module.parameters())
    for p in tail:
        p.grad = torch.ones_like(p)
    groups = training.hybrid_parameter_groups(encoder, head)
    ids = [id(p) for group in groups for p in group["params"]]
    assert len(ids) == len(set(ids))
    assert {id(p) for p in training._hybrid_vision_parameters(vision)} == {id(p) for p in earlier}
    for p in tail:
        assert id(p) not in ids and not p.requires_grad and p.grad is None
    for module in (encoder.projector, encoder.get_input_embeddings(), encoder.get_decoder(), head):
        earlier.extend(module.parameters())
    assert all(p.requires_grad and id(p) in ids for p in earlier)
    for module in (encoder.action_queries, encoder.proprio_projector, encoder.action_head):
        assert all(not p.requires_grad and id(p) not in ids for p in module.parameters())


def test_structural_timm_optional_norm_and_pool(training):
    vision = nn.Module()
    vision.featurizer = nn.Module()
    vision.featurizer.blocks = nn.ModuleList([nn.Linear(3, 3) for _ in range(4)])
    assert {id(p) for p in training._hybrid_vision_parameters(vision)} == {
        id(p) for p in vision.featurizer.blocks[:-1].parameters()}


def test_sequential_vision_keeps_all_parameters(training, inputs):
    encoder, head, *_ = inputs
    expected = {id(p) for p in encoder.vision_backbone.parameters()}
    assert {id(p) for p in training._hybrid_vision_parameters(encoder.vision_backbone)} == expected
    groups = training.hybrid_parameter_groups(encoder, head)
    selected = {id(p) for group in groups for p in group["params"]}
    assert expected <= selected
    assert all(p.requires_grad for p in encoder.vision_backbone.parameters())


@pytest.mark.parametrize("bad", ["horizon", "action_dim", "proprio_dim", "head_horizon", "nan"])
def test_dimensions_and_finite_validation(training, inputs, bad):
    encoder, head, batch, normalizer, _ = inputs
    if bad == "horizon":
        batch["actions"] = torch.zeros(2, 8, 7)
    elif bad == "action_dim":
        batch["actions"] = torch.zeros(2, 10, 6)
    elif bad == "proprio_dim":
        batch["proprio"] = torch.zeros(2, 7)
    elif bad == "head_horizon":
        head.action_horizon = 8
    else:
        batch["actions"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        training.hybrid_flow_loss(encoder, head, batch, normalizer)


def test_nonfinite_gradient_aborts_step(training, inputs):
    encoder, head, batch, normalizer, _ = inputs
    optimizer = training.make_hybrid_smoke_optimizer(encoder, head)
    before = head.action_decoder.weight.detach().clone()
    encoder.projector.weight.register_hook(lambda grad: grad * float("nan"))
    with pytest.raises(RuntimeError, match="Non-finite encoder gradients"):
        training.hybrid_optimizer_step(encoder, head, batch, normalizer, optimizer)
    torch.testing.assert_close(head.action_decoder.weight, before, rtol=0, atol=0)
    assert not optimizer.state


def test_optimizer_rejects_legacy_parameters(training, inputs):
    encoder, head, batch, normalizer, _ = inputs
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(head.parameters()))
    with pytest.raises(ValueError, match="exactly the Hybrid"):
        training.hybrid_optimizer_step(encoder, head, batch, normalizer, optimizer)


def test_disconnected_encoder_aborts_step(training, inputs):
    encoder, head, batch, normalizer, _ = inputs
    optimizer = training.make_hybrid_smoke_optimizer(encoder, head)
    original = encoder.encode_observation
    def disconnected(**kwargs):
        encoded = original(**kwargs)
        encoded["features"] = encoded["features"].detach()
        return encoded
    encoder.encode_observation = disconnected
    with pytest.raises(RuntimeError, match="No nonzero encoder gradients"):
        training.hybrid_optimizer_step(encoder, head, batch, normalizer, optimizer)
    assert not optimizer.state


def test_no_euler_or_evaluator_dependency(training, inputs, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Euler must never be used for training")
    monkeypatch.setitem(sys.modules, "prismatic.models.flow_sampling", SimpleNamespace(sample_actions_euler=forbidden))
    encoder, head, batch, normalizer, _ = inputs
    training.hybrid_flow_loss(encoder, head, batch, normalizer)
    tree = ast.parse((ROOT / "prismatic/training/hybrid_step.py").read_text())
    imports = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    imports.update(alias.name for n in ast.walk(tree) if isinstance(n, ast.Import) for alias in n.names)
    assert imports == {"torch", "prismatic.models.flow_action_head"}
