"""Simulator-free bridge tests; use actual evaluator helpers via AST extraction."""

import json
import math
import sys
from abc import ABC, abstractmethod
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from test_hybrid_data import definitions, standalone


bridge = standalone("experiments/robot/libero/hybrid_policy.py")
runner = standalone("experiments/robot/libero/run_hybrid_episode.py")
normalization = standalone("prismatic/vla/hybrid_normalization.py")
sampling = standalone("prismatic/models/flow_sampling.py")
flow = standalone("prismatic/models/flow_action_head.py")


@pytest.fixture
def dependencies(monkeypatch):
    ns = dict(ABC=ABC, abstractmethod=abstractmethod, np=np, math=math, Image=Image)
    definitions("prismatic/models/backbones/llm/prompting/base_prompter.py", {"PromptBuilder"}, ns)
    definitions("prismatic/models/backbones/llm/prompting/qwen_prompter.py", {"QwenPromptBuilder", "SYS_PROMPTS"}, ns)
    definitions("experiments/robot/robot_utils.py", {"normalize_gripper_action", "invert_gripper_action"}, ns)
    definitions("experiments/robot/libero/libero_utils.py", {
        "quat2axisangle", "get_libero_image", "get_libero_wrist_image", "get_libero_dummy_action"}, ns)
    # Only resize is replaced: synthetic inputs already have the requested size.
    ns.update(resize_image_for_policy=lambda image, size: image, OPENVLA_IMAGE_SIZE=224)
    definitions("experiments/robot/openvla_utils.py", {"check_image_format", "prepare_images_for_vla"}, ns)
    definitions("experiments/robot/libero/run_libero_eval.py", {"prepare_observation", "process_action"}, ns)
    ns.update(TASK_MAX_STEPS={"libero_spatial": 18}, get_image_resize_size=lambda cfg: 224)
    for name, module in {
        "experiments.robot.openvla_utils": SimpleNamespace(**ns),
        "experiments.robot.libero.run_libero_eval": SimpleNamespace(**ns),
        "prismatic.models.backbones.llm.prompting": SimpleNamespace(**ns),
        "prismatic.models.flow_sampling": sampling,
        "prismatic.vla.hybrid_normalization": normalization,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return ns


def statistics():
    return {kind: {"mean": [0.25] * dim, "std": [2.] * dim}
            for kind, dim in (("action", 7), ("proprio", 8))}


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def encode_observation(self, **inputs):
        self.inputs = inputs
        return {"features": torch.zeros(1, 3, 896), "feature_mask": torch.ones(1, 3, dtype=torch.bool)}


class Processor:
    def __init__(self):
        self.prompts = []

    def __call__(self, prompt, image):
        self.prompts.append(prompt)
        return dict(input_ids=torch.tensor([[1, 2, 3]]), attention_mask=torch.ones(1, 3),
                    pixel_values=torch.full((1, 6, 224, 224), float(np.asarray(image)[0, 0, 0])))


def observation():
    return dict(full_image=np.zeros((224, 224, 3), dtype=np.uint8),
                wrist_image=np.ones((224, 224, 3), dtype=np.uint8), state=np.arange(8, dtype=np.float32))


def policy():
    head = flow.SimVLAFlowActionHead(hidden_dim=16, depth=1, num_heads=2, dropout=0)
    return bridge.HybridLiberoPolicy(Encoder(), Processor(), head,
                                    normalization.HybridZScoreNormalizer(statistics()),
                                    center_crop=False, num_steps=3)


def test_policy_real_euler_and_normalizer(dependencies, monkeypatch):
    p = policy()
    calls = []
    real_sample = sampling.sample_actions_euler

    def record(head, features, proprio, mask, **kwargs):
        torch.testing.assert_close(proprio, (torch.arange(8)[None] - .25) / 2.000001)
        value = real_sample(head, features, proprio, mask, **kwargs)
        calls.append(value.clone())
        return value

    monkeypatch.setattr(sampling, "sample_actions_euler", record)
    obs = observation()
    result = p(obs, "PICK UP")
    assert result.shape == (10, 7) and np.isfinite(result).all()
    np.testing.assert_allclose(result, (calls[0] * 2.000001 + .25)[0].numpy())
    np.testing.assert_array_equal(obs["state"], np.arange(8))
    pixels = p.encoder.inputs["pixel_values"]
    assert (pixels[:, :6] == 0).all() and (pixels[:, 6:] == 1).all()
    prompt = p.processor.prompts[0]
    assert p.processor.prompts == [prompt, prompt]
    assert "What action should the robot take to pick up?" in prompt
    assert prompt.endswith("<|im_start|>assistant\n") and "<|endoftext|>" not in prompt


def test_episode_generator_advances_and_isolates_flow_rng(dependencies, monkeypatch):
    p = policy()
    actual_sampler = sampling.sample_actions_euler
    generators = []
    def sampled(*args, **kwargs):
        assert torch.is_inference_mode_enabled()
        generators.append(kwargs["generator"])
        return actual_sampler(*args, **kwargs)
    monkeypatch.setattr(sampling, "sample_actions_euler", sampled)
    p.begin_episode(7, debug=True)
    first = p(observation(), "task")
    torch.randn(1000)  # Unrelated global RNG does not alter episode sampling.
    second = p(observation(), "task")
    hashes = list(p.debug_trace["noise_hashes"])
    assert hashes[0] != hashes[1] and generators[0] is generators[1]
    p.begin_episode(7, debug=True)
    np.testing.assert_array_equal(first, p(observation(), "task"))
    np.testing.assert_array_equal(second, p(observation(), "task"))
    assert p.debug_trace["noise_hashes"] == hashes
    # Reference stream and generator stream agree on CPU for the same sequence.
    torch.manual_seed(7)
    p.begin_episode(7, reference=True, debug=True)
    np.testing.assert_array_equal(first, p(observation(), "task"))
    np.testing.assert_array_equal(second, p(observation(), "task"))
    assert p.debug_trace["noise_hashes"] == hashes


def test_encoder_dict_contract_does_not_depend_on_iteration_order(dependencies):
    p = policy()
    # Reversed insertion order and extra metadata must not affect named access.
    p.encoder.encode_observation = lambda **kw: {
        "feature_mask": torch.ones(1, 3, dtype=torch.bool),
        "metadata": None,
        "features": torch.zeros(1, 3, 896),
    }
    assert p(observation(), "pick up").shape == (10, 7)


def test_native_weight_renames_and_strict_missing_validation():
    names = [
        ("vision_backbone.dino_featurizer.gamma", "vision_backbone.featurizer.scale_factor"),
        ("vision_backbone.siglip_featurizer.weight", "vision_backbone.fused_featurizer.weight"),
        ("llm_backbone.llm.weight", "language_model.weight"),
        ("projector.projector.0.weight", "projector.fc1.weight"),
        ("projector.projector.2.weight", "projector.fc2.weight"),
        ("projector.projector.4.weight", "projector.fc3.weight"),
    ]
    encoder = torch.nn.Module()
    for name in [target for _, target in names] + ["action_queries.weight"]:
        parent = encoder
        *parts, parameter = name.split(".")
        for part in parts:
            if not hasattr(parent, part):
                parent.add_module(part, torch.nn.Module())
            parent = getattr(parent, part)
        parent.register_parameter(parameter, torch.nn.Parameter(torch.zeros(1)))
    weights = {name: torch.tensor([float(i + 1)]) for i, (name, _) in enumerate(names)}
    native = SimpleNamespace(state_dict=lambda: weights)
    runner.load_native_weights(encoder, native)
    for source, target in names:
        torch.testing.assert_close(encoder.state_dict()[target], weights[source])
    weights["unexpected.weight"] = torch.ones(1)
    with pytest.raises(ValueError, match="unexpected.weight"):
        runner.load_native_weights(encoder, native)
    del weights["unexpected.weight"]
    del weights["llm_backbone.llm.weight"]
    with pytest.raises(ValueError, match="language_model.weight"):
        runner.load_native_weights(encoder, native)
    weights["llm_backbone.llm.weight"] = torch.ones(2)
    with pytest.raises(RuntimeError, match="size mismatch"):
        runner.load_native_weights(encoder, native)


def test_load_policy_uses_native_and_config_assets(dependencies, monkeypatch):
    calls = []
    cfg = SimpleNamespace(text_config=SimpleNamespace(hidden_size=896, model_type="qwen2"),
                          image_sizes=[224, 224], use_fused_vision_backbone=True,
                          vision_backbone_id="dinosiglip-vit-so-224px")

    def native_load(path, **kwargs):
        calls.append(("native", path, kwargs))
        return SimpleNamespace(state_dict=lambda: {})

    class HFEncoder(Encoder):
        def __init__(self, config):
            super().__init__()
            assert config is cfg
            self.vision_backbone = SimpleNamespace(set_num_images_in_input=lambda n: calls.append(("views", n)))

        def load_state_dict(self, weights, strict):
            assert weights == {} and strict is False
            return ["action_queries.weight"], []

    def asset_loader(kind, result):
        def load(path, **kwargs):
            calls.append((kind, path, kwargs))
            return result
        return SimpleNamespace(from_pretrained=load)

    for name, module in {
        "transformers": SimpleNamespace(AutoTokenizer=asset_loader("tokenizer", "tokenizer")),
        "prismatic.models": SimpleNamespace(load=native_load),
        "prismatic.extern.hf.configuration_prismatic": SimpleNamespace(OpenVLAConfig=asset_loader("config", cfg)),
        "prismatic.extern.hf.modeling_prismatic": SimpleNamespace(PrismaticForConditionalGeneration=HFEncoder),
        "prismatic.extern.hf.processing_prismatic": SimpleNamespace(
            PrismaticImageProcessor=asset_loader("processor", "images"),
            PrismaticProcessor=lambda image, tokenizer: (image, tokenizer)),
        "prismatic.models.flow_action_head": SimpleNamespace(SimVLAFlowActionHead=lambda:
            flow.SimVLAFlowActionHead(hidden_dim=16, depth=1, num_heads=2)),
        "experiments.robot.libero.hybrid_policy": SimpleNamespace(
            HybridLiberoPolicy=bridge.HybridLiberoPolicy,
            load_hybrid_normalizer=lambda path: normalization.HybridZScoreNormalizer(statistics())),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    p = runner.load_policy("native-assets", "hf-assets", "stats.json", "cpu", 10, True)
    assert p.processor == ("images", "tokenizer")
    assert ("native", "native-assets", {"hf_token": "", "load_for_training": True}) in calls
    assert ("views", 2) in calls
    for kind in ("config", "processor", "tokenizer"):
        call = next(call for call in calls if call[0] == kind)
        assert call[1] == "hf-assets" and call[2]["local_files_only"] is True


@pytest.mark.parametrize("kind", ["state_shape", "state_nan", "pixels", "features", "mask",
                                  "normalized_shape", "normalized_nan", "denormalized_shape", "denormalized_nan"])
def test_reject_invalid_tensors(dependencies, monkeypatch, kind):
    p, obs = policy(), observation()
    if kind == "state_shape":
        obs["state"] = np.zeros((1, 8))
    elif kind == "state_nan":
        obs["state"][0] = np.nan
    elif kind == "pixels":
        p.processor = lambda *args: dict(input_ids=torch.ones(1, 2, dtype=torch.long),
                                        attention_mask=torch.ones(1, 2), pixel_values=torch.zeros(1, 3, 224, 224))
    elif kind in ("features", "mask"):
        p.encoder.encode_observation = lambda **kw: {
            "features": torch.full((1, 3, 896), float("nan") if kind == "features" else 0.),
            "feature_mask": torch.zeros(1, 3, dtype=torch.bool)}
    elif kind.startswith("normalized"):
        monkeypatch.setattr(sampling, "sample_actions_euler", lambda *a, **kw:
                            torch.zeros(1, 8, 7) if kind.endswith("shape") else torch.full((1, 10, 7), float("nan")))
    else:
        monkeypatch.setattr(p.normalizer, "denormalize_action", lambda x:
                            torch.zeros(1, 8, 7) if kind.endswith("shape") else x * float("nan"))
    with pytest.raises(ValueError):
        p(obs, "pick up")


def test_exact_dataset_key(dependencies, tmp_path):
    path = tmp_path / "dataset_statistics.json"
    path.write_text(json.dumps({"libero_spatial": statistics()}))
    with pytest.raises(ValueError, match="no fallback"):
        bridge.load_hybrid_normalizer(path)
    path.write_text(json.dumps({bridge.DATASET_KEY: statistics()}))
    norm = bridge.load_hybrid_normalizer(path)
    assert norm.action_mean.tolist() == [.25] * 7


class Environment:
    def __init__(self, done_at=None):
        self.steps = []
        self.resets = 0
        self.done_at = done_at

    def reset(self):
        self.resets += 1

    def set_init_state(self, state):
        self.initial_state = state
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        image[-1, -1] = 13
        return dict(agentview_image=image, robot0_eye_in_hand_image=image + 1,
                    robot0_eef_pos=np.array([1, 2, 3]), robot0_eef_quat=np.array([0., 0., 0., 1.]),
                    robot0_gripper_qpos=np.array([.1, .2]))

    def step(self, action):
        assert isinstance(action, list) and len(action) == 7 and np.isfinite(action).all()
        self.steps.append(action)
        return self.set_init_state(self.initial_state), 0, len(self.steps) == self.done_at, {}


@pytest.mark.parametrize("done_at,expected_steps", [(None, 18), (11, 9)])
def test_rollout_reuses_canonical_conversion_and_termination(dependencies, done_at, expected_steps):
    env = Environment(done_at)
    cfg = SimpleNamespace(task_suite_name="libero_spatial", model_family="openvla",
                          num_steps_wait=2, num_open_loop_steps=8)
    calls = []

    def canonical(obs, language):
        calls.append(obs)
        assert obs["full_image"][0, 0, 0] == 13 and obs["wrist_image"][0, 0, 0] == 14
        np.testing.assert_allclose(obs["state"], [1, 2, 3, 0, 0, 0, .1, .2])
        chunk = np.tile(np.arange(7, dtype=np.float32), (10, 1))
        chunk[:, -1] = np.arange(10) % 2
        return chunk

    result = runner.run_single_episode(cfg, env, "task", canonical, "initial")
    assert env.resets == 1 and result["action_steps"] == expected_steps
    assert result["success"] == (done_at is not None)
    assert result["policy_calls"] == len(calls) == math.ceil(expected_steps / 8)
    for i, action in enumerate(env.steps[2:]):
        np.testing.assert_equal(action[:6], np.arange(6))
        assert action[-1] == (1 if i % 2 == 0 else -1)


def test_rollout_errors_propagate(dependencies):
    cfg = SimpleNamespace(task_suite_name="libero_spatial", model_family="openvla",
                          num_steps_wait=0, num_open_loop_steps=8)
    def broken(*args):
        raise RuntimeError("inference failed")
    with pytest.raises(RuntimeError, match="inference failed"):
        runner.run_single_episode(cfg, Environment(), "task", broken, "initial")
