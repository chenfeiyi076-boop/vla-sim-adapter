"""CPU-only tests; no checkpoint, CUDA, RLDS, LIBERO, or network required.

Load the standalone file directly because prismatic/__init__.py eagerly imports
model loaders and optional training dependencies. Tests require only torch and
pytest; a missing dependency is an error, not a skipped/pass result.
"""

import ast
import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
HEAD_PATH = ROOT / "prismatic/models/flow_action_head.py"
ENCODER_PATH = ROOT / "prismatic/extern/hf/modeling_prismatic.py"
spec = importlib.util.spec_from_file_location("standalone_flow_action_head", HEAD_PATH)
flow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flow)


@pytest.fixture(params=[False, True], ids=["fallback", "sdpa"])
def head(request):
    if request.param and not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        pytest.skip("This PyTorch version does not provide SDPA")
    torch.manual_seed(7)
    # Real input dimensions, tiny Transformer for fast CPU unit tests.
    model = flow.SimVLAFlowActionHead(hidden_dim=32, depth=2, num_heads=4, max_seq_len=64)
    for block in model.blocks:
        block.attn.fused_attn = request.param
    return model


def inputs(T=37):
    return (
        torch.randn(2, T, 896), torch.randn(2, 10, 7),
        torch.randn(2, 8), torch.tensor([0.1, 0.9]),
        torch.ones(2, T, dtype=torch.bool),
    )


def test_shape_finite_loss_and_backward(head):
    features, _, proprio, _, mask = inputs()
    samples = flow.sample_flow_matching_inputs(torch.randn(2, 10, 7))
    output = head(features, samples["x_t"], proprio, samples["t"], mask)
    assert output.shape == (2, 10, 7)
    assert torch.isfinite(output).all()
    loss = flow.compute_flow_matching_loss(output, samples["target_velocity"])
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    for module in (head.vlm_proj, head.action_encoder, head.blocks, head.action_decoder):
        for parameter in module.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
        assert any(torch.count_nonzero(p.grad) > 0 for p in module.parameters())


def test_mask_invariance_and_padded_length(head):
    head.eval()
    features, action, proprio, t, mask = inputs(T=5)
    mask[:, 3:] = False
    baseline = head(features, action, proprio, t, mask)
    changed = features.clone()
    changed[:, 3:] = torch.randn_like(changed[:, 3:]) * 1e6
    torch.testing.assert_close(head(changed, action, proprio, t, mask), baseline)
    # Appending masked tokens must not change valid tokens' positions/results.
    torch.testing.assert_close(
        head(features[:, :3], action, proprio, t), baseline, rtol=1e-5, atol=1e-6
    )
    # Guard against NaNs in masked data contaminating attention matrix products.
    changed[:, 3:] = float("nan")
    torch.testing.assert_close(head(changed, action, proprio, t, mask), baseline)


def test_action_tokens_always_valid_and_concat_order(head):
    head.eval()
    features, action, proprio, t, mask = inputs(T=5)
    mask[:] = False
    captured = {}

    def capture_encoder(module, args):
        captured["action_inputs"] = args[0].detach().clone()

    def capture_block(module, args):
        captured["tokens"], captured["mask"] = args

    hook1 = head.action_encoder.register_forward_pre_hook(capture_encoder)
    hook2 = head.blocks[0].register_forward_pre_hook(capture_block)
    try:
        output = head(features, action, proprio, t, mask)
    finally:
        hook1.remove()
        hook2.remove()
    expected_inputs = torch.cat([
        action, proprio[:, None].expand(2, 10, 8),
        flow.timestep_embedding(t, 32)[:, None].expand(2, 10, 32),
    ], dim=-1)
    torch.testing.assert_close(captured["action_inputs"], expected_inputs)
    assert captured["action_inputs"].shape == (2, 10, 47)
    expected_tokens = torch.cat([
        head.action_encoder(expected_inputs), head.vlm_proj(torch.zeros_like(features)),
    ], dim=1) + head.pos_emb[:, :15]
    torch.testing.assert_close(captured["tokens"], expected_tokens)
    assert captured["mask"].shape == (2, 15)
    assert captured["mask"][:, :10].all()
    assert not captured["mask"][:, 10:].any()
    assert torch.isfinite(output).all()
    assert not torch.allclose(output, head(features, action + 1, proprio, t, mask))


def test_masked_feature_gradients_are_zero(head):
    features, action, proprio, t, mask = inputs(T=5)
    features.requires_grad_(True)
    mask[:, 3:] = False
    head(features, action, proprio, t, mask).square().mean().backward()
    assert torch.count_nonzero(features.grad[:, 3:]) == 0
    assert torch.count_nonzero(features.grad[:, :3]) > 0


def test_sequence_capacity(head):
    head.eval()
    assert head(*inputs(T=54)).shape == (2, 10, 7)  # Exactly 64 tokens.
    with pytest.raises(ValueError, match=r"H \+ T = 10 \+ 55 = 65 exceeds max_seq_len=64"):
        head(*inputs(T=55))


@pytest.mark.parametrize("bad_mask", [torch.ones(5, dtype=torch.bool), torch.ones(2, 1, dtype=torch.bool),
                                      torch.ones(2, 5)])
def test_reject_mask_broadcast_and_non_bool(head, bad_mask):
    features, action, proprio, t, _ = inputs(T=5)
    with pytest.raises(ValueError, match="feature_mask"):
        head(features, action, proprio, t, bad_mask)


def test_reject_wrong_action_shape(head):
    features, action, proprio, t, mask = inputs()
    with pytest.raises(ValueError, match="noisy_action"):
        head(features, action[:, :1], proprio, t, mask)


def test_sdpa_and_fallback_agree():
    if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        pytest.skip("This PyTorch version does not provide SDPA")
    model = flow.SimVLAFlowActionHead(hidden_dim=32, depth=2, num_heads=4).eval()
    args = inputs(T=5)
    args[-1][:, 3:] = False
    sdpa = model(*args)
    for block in model.blocks:
        block.attn.fused_attn = False
    torch.testing.assert_close(model(*args), sdpa, rtol=1e-5, atol=1e-6)


def test_flow_sampling_matches_reference_rng_and_equations():
    action = torch.arange(140, dtype=torch.float32).reshape(2, 10, 7) / 100
    torch.manual_seed(123)
    result = flow.sample_flow_matching_inputs(action)
    torch.manual_seed(123)
    expected_t = torch.distributions.Beta(torch.tensor(1.5), torch.tensor(1.0)).sample((2,)) * .999 + .001
    expected_noise = torch.randn_like(action)
    assert result["t"].shape == (2,)
    assert ((result["t"] >= .001) & (result["t"] <= 1)).all()
    torch.testing.assert_close(result["t"], expected_t)
    torch.testing.assert_close(result["noise"], expected_noise)
    t3 = expected_t[:, None, None]
    torch.testing.assert_close(result["x_t"], t3 * expected_noise + (1 - t3) * action)
    torch.testing.assert_close(result["target_velocity"], expected_noise - action)
    for key in ("noise", "x_t", "target_velocity"):
        assert result[key].shape == action.shape


def test_mse_treats_all_seven_dimensions_equally_and_rejects_broadcast():
    target = torch.zeros(2, 10, 7)
    for channel in range(7):
        pred = target.clone()
        pred[..., channel] = 2
        torch.testing.assert_close(flow.compute_flow_matching_loss(pred, target), torch.tensor(4 / 7))
    with pytest.raises(ValueError, match="identical"):
        flow.compute_flow_matching_loss(target, target[:1])


def test_timestep_unscaled_and_odd_dimension():
    t = torch.tensor([0., .5, 1.], dtype=torch.float64)
    frequencies = 100 ** (-torch.arange(16, dtype=t.dtype) / 16)
    expected = torch.cat([(t[:, None] * frequencies).cos(), (t[:, None] * frequencies).sin()], -1)
    torch.testing.assert_close(flow.timestep_embedding(t, 32), expected)
    odd = flow.timestep_embedding(t, 33)
    torch.testing.assert_close(odd[:, :32], expected)
    assert torch.count_nonzero(odd[:, -1]) == 0


def test_small_defaults_and_constructor_dimensions():
    defaults = inspect.signature(flow.SimVLAFlowActionHead).parameters
    for name, expected in dict(hidden_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                               dropout=.1, action_horizon=10, action_dim=7,
                               proprio_dim=8, time_dim=32, max_seq_len=1024).items():
        assert defaults[name].default == expected
    # Non-default dimensions prove that the implementation does not hardcode 896/10/7.
    model = flow.SimVLAFlowActionHead(vlm_hidden_dim=11, hidden_dim=24, depth=1, num_heads=3,
                                    action_horizon=3, action_dim=4, proprio_dim=2, time_dim=6)
    assert model(torch.randn(1, 9, 11), torch.randn(1, 3, 4), torch.randn(1, 2), torch.rand(1)).shape == (1, 3, 4)
    with pytest.raises(ValueError, match="divisible"):
        flow.SimVLAFlowActionHead(hidden_dim=25, num_heads=3)


def test_no_old_action_path_dependency():
    tree = ast.parse(HEAD_PATH.read_text(encoding="utf-8"))
    forbidden = {"action_queries", "L1RegressionActionHead", "ProprioProjector", "MLPResNet_Pro",
                 "MLPResNetBlock_Pro", "L1Loss", "l1_loss"}
    identifiers = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    identifiers |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not identifiers & forbidden
    imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    for node in imports:
        names = [node.module] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names]
        assert all(n.split(".")[0] in {"math", "typing", "torch"} for n in names)


def test_encode_observation_source_with_tiny_modules():
    # Exercise the exact canonical methods without importing timm/transformers or
    # constructing Prismatic. This checks packing/interface only, not real Qwen.
    tree = ast.parse(ENCODER_PATH.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PrismaticForConditionalGeneration")
    names = {"encode_observation", "get_input_embeddings", "get_decoder",
             "_process_vision_features", "_build_multimodal_attention"}
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(methods) == len(names)
    namespace = {"torch": torch, "nn": nn, "Dict": Dict}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(ENCODER_PATH), "exec"), namespace)
    Encoder = type("TinyObservationEncoder", (nn.Module,), {n: namespace[n] for n in names})

    class TinyDecoder(nn.Module):
        def forward(self, inputs_embeds, attention_mask, **kwargs):
            assert kwargs == dict(use_cache=False, output_attentions=False,
                                  output_hidden_states=False, return_dict=True)
            assert attention_mask.dtype == torch.bool
            return SimpleNamespace(last_hidden_state=inputs_embeds + 1)

    class NoLogitsLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(20, 11)
            self.decoder = TinyDecoder()

        def get_input_embeddings(self):
            return self.embedding

        def get_decoder(self):
            return self.decoder

        def forward(self, *args, **kwargs):
            raise AssertionError("The vocabulary-logit wrapper must not be called")

    encoder = Encoder()
    encoder.language_model = NoLogitsLM()
    encoder.vision_backbone = nn.Sequential(nn.Flatten(2),)  # [B,3,2,2] -> [B,3,4]
    encoder.projector = nn.Linear(4, 11)
    ids = torch.tensor([[1, 2, 3, 4, 0], [1, 2, 3, 0, 0]])
    mask = ids.ne(0)
    pixels = torch.randn(2, 3, 2, 2)
    result = encoder.encode_observation(ids, mask.long(), pixels)
    text = encoder.get_input_embeddings()(ids)
    images = encoder.projector(encoder.vision_backbone(pixels))
    expected = torch.cat([text[:, :1], images, text[:, 1:]], dim=1) + 1
    torch.testing.assert_close(result["features"], expected)
    assert result["features"].shape == (2, 8, 11)
    expected_mask = torch.cat([mask[:, :1], torch.ones(2, 3, dtype=torch.bool), mask[:, 1:]], 1)
    assert torch.equal(result["feature_mask"], expected_mask)
    # There is no action-query or proprio module on this stub at all.
    result["features"].sum().backward()
    assert encoder.projector.weight.grad is not None
    assert encoder.language_model.embedding.weight.grad is not None
    with pytest.raises(ValueError, match="same.*shape"):
        encoder.encode_observation(ids, mask[:1], pixels)

