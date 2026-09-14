"""CPU synthetic contracts; no TFDS data, TensorFlow, checkpoints or network.

As in Phase 1, extract selected canonical definitions to avoid Prismatic's
eager optional imports. A small NumPy-backed TF fake exercises the *actual*
RLDS restructure/chunk functions, not a second implementation of the pipeline.
This does not replace a future real TensorFlow/RLDS integration test.
"""

import __future__
import ast
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Sequence

import numpy as np
from PIL import Image
import pytest
import torch
from torch.nn.utils.rnn import pad_sequence


ROOT = Path(__file__).resolve().parents[1]
HYBRID = "prismatic/vla/datasets/hybrid_datasets.py"
RLDS = "prismatic/vla/datasets/rlds/dataset.py"


def definitions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    selected = [node for node in tree.body if
                (isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names) or
                (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets))]
    namespace.setdefault("__name__", __name__)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(ROOT / path), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace


def standalone(path):
    spec = importlib.util.spec_from_file_location("hybrid_test_module", ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tree_map(fn, value):
    return {k: tree_map(fn, v) for k, v in value.items()} if isinstance(value, dict) else fn(value)


def fake_tf():
    return SimpleNamespace(
        shape=np.shape, range=np.arange, repeat=np.repeat, concat=np.concatenate,
        cast=lambda x, dtype: np.asarray(x, dtype=dtype), float32=np.float32,
        bool=np.bool_, string=np.dtype("S8"), zeros=np.zeros, tile=np.tile,
        convert_to_tensor=np.asarray, broadcast_to=np.broadcast_to, maximum=np.maximum,
        minimum=np.minimum, fill=np.full, gather=lambda x, i: np.asarray(x)[i],
        nest=SimpleNamespace(map_structure=tree_map), data=SimpleNamespace(AUTOTUNE=-1),
    )


@pytest.fixture
def bridge():
    ns = dict(torch=torch, np=np, Image=Image, dataclass=dataclass, deepcopy=deepcopy,
              pad_sequence=pad_sequence, Callable=Callable, Sequence=Sequence, ABC=ABC,
              abstractmethod=abstractmethod)
    definitions("prismatic/models/backbones/llm/prompting/base_prompter.py", {"PromptBuilder"}, ns)
    definitions("prismatic/models/backbones/llm/prompting/qwen_prompter.py", {"QwenPromptBuilder", "SYS_PROMPTS"}, ns)
    # Real legacy constructor + captured interleaver, so inherited view-loading
    # configuration and Hybrid overrides are tested together without TFDS.
    captured = []

    def oxe(root, mixture, **kwargs):
        assert kwargs["load_camera_views"] == ("primary", "wrist")
        assert kwargs["load_proprio"] and kwargs["load_language"]
        return [{"name": name, "standardize_fn": "original_standardizer"} for name, _ in mixture], [1.]

    def interleave(**config):
        captured.append(config)
        return "dataset", 123, {"libero_spatial_no_noops": stats()}

    ns.update(IterableDataset=torch.utils.data.IterableDataset, OXE_NAMED_MIXTURES={},
              NUM_ACTIONS_CHUNK=8, ACTION_PROPRIO_NORMALIZATION_TYPE="bounds_q99",
              get_oxe_dataset_kwargs_and_weights=oxe, make_interleaved_dataset=interleave)
    definitions("prismatic/vla/datasets/datasets.py", {"RLDSDataset"}, ns)
    definitions(HYBRID, {"HybridRLDSDataset", "HybridRLDSBatchTransform", "PaddedCollatorForHybridFlow",
                         "HYBRID_ACTION_HORIZON", "HYBRID_ACTION_DIM", "HYBRID_PROPRIO_DIM"}, ns)
    ns["captured"] = captured
    return ns


def stats():
    return {kind: dict(mean=np.arange(dim, dtype=np.float32),
                       std=np.arange(dim, dtype=np.float32) / 2,
                       mask=[True] * (dim - 1) + [False]) for kind, dim in (("action", 7), ("proprio", 8))}


def frame():
    return dict(action=np.arange(70, dtype=np.float32).reshape(10, 7),
                observation=dict(proprio=np.arange(8, dtype=np.float32)[None],
                                 image_primary=np.zeros((1, 4, 4, 3), dtype=np.uint8),
                                 image_wrist=np.ones((1, 4, 4, 3), dtype=np.uint8)),
                task=dict(language_instruction=b"Put the bowl on the plate"),
                dataset_name=b"libero_spatial_no_noops")


class Tokenizer:
    def __call__(self, text, add_special_tokens):
        assert add_special_tokens is True
        self.prompt = text
        self.ids = list(range(1, len(text) + 1))
        return SimpleNamespace(input_ids=self.ids)


def image_transform(image):
    rgb = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1)
    return torch.cat([rgb, rgb + 10], 0)  # Distinguishable DINO/SigLIP channels.


def test_independent_horizon_and_inherited_configuration(bridge):
    constants = definitions("prismatic/vla/constants.py", {"LIBERO_CONSTANTS"},
                            {"NormalizationType": SimpleNamespace(BOUNDS_Q99="bounds_q99")})
    assert constants["LIBERO_CONSTANTS"]["NUM_ACTIONS_CHUNK"] == 8
    assert bridge["HYBRID_ACTION_HORIZON"] == 10
    args = (Path("unused"), "libero_spatial_no_noops", lambda x: x, (224, 224))
    legacy = bridge["RLDSDataset"](*args)
    hybrid = bridge["HybridRLDSDataset"](*args)
    old, new = bridge["captured"]
    assert old["traj_transform_kwargs"]["future_action_window_size"] == 7
    assert new["traj_transform_kwargs"]["future_action_window_size"] == 9
    assert new["traj_transform_kwargs"]["window_size"] == 1
    assert "normalize_action_proprio" not in old["dataset_kwargs_list"][0]
    assert new["dataset_kwargs_list"][0]["normalize_action_proprio"] is False
    assert new["dataset_kwargs_list"][0]["standardize_fn"] == "original_standardizer"
    assert legacy.dataset_length == hybrid.dataset_length == 123
    assert "libero_spatial_no_noops" in hybrid.dataset_statistics


def test_exact_current_action_alignment():
    chunk = definitions("prismatic/vla/datasets/rlds/traj_transforms.py", {"chunk_act_obs"},
                        {"tf": fake_tf()})["chunk_act_obs"]
    labels = np.arange(14)
    trajectory = dict(action=np.repeat(labels[:, None], 7, axis=1),
                      observation={"timestep": labels}, task={"language_instruction": labels},
                      dataset_name=labels, absolute_action_mask=np.zeros((14, 7), dtype=bool))
    result = chunk(trajectory, window_size=1, future_action_window_size=9)
    assert result["action"].shape == (5, 10, 7)
    for i in range(5):
        assert result["observation"]["timestep"][i, 0] == i
        np.testing.assert_array_equal(result["action"][i, :, 0], np.arange(i, i + 10))
        assert result["action"][i, 0, 0] == i  # Explicitly not i+1.


@pytest.mark.parametrize("normalize", [None, False])
def test_raw_switch_preserves_standardization_restructure_and_statistics(normalize):
    raw = dict(action=np.ones((12, 7)), observation={"state": np.ones((12, 8)), "rgb": np.zeros((12, 2, 2, 3))})
    calls = []

    class Dataset:
        def __init__(self, value):
            self.value = value

        @staticmethod
        def from_rlds(builder, **kwargs):
            return Dataset(deepcopy(raw))

        def traj_map(self, fn, *args):
            self.value = fn(self.value)
            return self

    def standardize(traj):
        calls.append("standardize")
        traj["action"] *= 3
        traj["observation"]["state"] *= 5
        return traj

    def statistics(ds, **kwargs):
        calls.append("statistics")
        assert "proprio" in ds.value["observation"]
        np.testing.assert_array_equal(ds.value["action"], np.full((12, 7), 3))
        return {"action": {"mean": ds.value["action"].mean(0)},
                "proprio": {"mean": ds.value["observation"]["proprio"].mean(0)}}

    def old_normalize(traj, **kwargs):
        calls.append("normalize")
        traj["action"] += 100
        return traj

    ns = dict(tf=fake_tf(), tfds=SimpleNamespace(builder=lambda *a, **k: SimpleNamespace(info="info", data_dir="unused")),
              dl=SimpleNamespace(DLataset=Dataset), np=np, inspect=inspect, tree_map=tree_map,
              partial=partial, get_dataset_statistics=statistics, normalize_action_and_proprio=old_normalize)
    make = definitions(RLDS, {"make_dataset_from_rlds"}, ns)["make_dataset_from_rlds"]
    assert inspect.signature(make).parameters["normalize_action_proprio"].default is True
    kwargs = {} if normalize is None else {"normalize_action_proprio": normalize}
    ds, metadata = make("libero_spatial_no_noops", "unused", train=True, standardize_fn=standardize,
                        state_obs_keys=["state"], image_obs_keys={"primary": "rgb"},
                        action_proprio_normalization_type="bounds_q99", **kwargs)
    assert calls.count("standardize") == 2 and calls.count("statistics") == 1
    assert calls.count("normalize") == (1 if normalize is None else 0)
    np.testing.assert_array_equal(metadata["action"]["mean"], np.full(7, 3))
    np.testing.assert_array_equal(metadata["proprio"]["mean"], np.full(8, 5))
    np.testing.assert_array_equal(ds.value["action"], np.full((12, 7), 103 if normalize is None else 3))
    assert ds.value["dataset_name"][0] == "libero_spatial_no_noops"
    assert "image_primary" in ds.value["observation"]


def test_observation_prompt_shapes_and_no_action_leakage(bridge):
    tokenizer = Tokenizer()
    transform = bridge["HybridRLDSBatchTransform"](tokenizer, image_transform)
    sample = transform(frame())
    expected = ("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\nWhat action should the robot take to put the bowl on the plate?<|im_end|>\n"
                "<|im_start|>assistant\n")
    assert tokenizer.prompt == expected
    assert sample["input_ids"].tolist() == tokenizer.ids
    assert "labels" not in sample
    assert sample["actions"].shape == (10, 7) and sample["proprio"].shape == (8,)
    assert sample["dataset_name"] == b"libero_spatial_no_noops"
    changed = frame()
    changed["action"] += 1000
    assert torch.equal(transform(changed)["input_ids"], sample["input_ids"])
    assert "ActionTokenizer" not in (ROOT / HYBRID).read_text(encoding="utf-8")
    changed["action"] = changed["action"][:8]
    with pytest.raises(ValueError, match="actions"):
        transform(changed)
    changed = frame()
    changed["observation"]["proprio"] = np.zeros((2, 8))
    with pytest.raises(ValueError, match="proprio"):
        transform(changed)


def batch(bridge):
    transform = bridge["HybridRLDSBatchTransform"](Tokenizer(), image_transform)
    a, b = transform(frame()), transform(frame())
    a["input_ids"] = torch.tensor([1, 0, 2])
    b["input_ids"] = torch.tensor([3])
    return bridge["PaddedCollatorForHybridFlow"](pad_token_id=0)([a, b])


def test_collator_padding_and_camera_order(bridge):
    result = batch(bridge)
    assert result["input_ids"].tolist() == [[1, 0, 2], [3, 0, 0]]
    assert result["attention_mask"].dtype == torch.bool
    assert result["attention_mask"].tolist() == [[True, True, True], [True, False, False]]
    assert result["pixel_values"].shape == (2, 12, 4, 4)
    assert result["pixel_values"][0, :, 0, 0].tolist() == [0.] * 3 + [10.] * 3 + [1.] * 3 + [11.] * 3
    assert result["actions"].shape == (2, 10, 7)
    assert result["proprio"].shape == (2, 8)
    assert result["dataset_names"] == [b"libero_spatial_no_noops"] * 2
    assert "labels" not in result


def test_normalization_equation_inverse_zero_std_and_gripper():
    normalizer = standalone("prismatic/vla/hybrid_normalization.py").HybridZScoreNormalizer(stats())
    x = torch.arange(140, dtype=torch.float32).reshape(2, 10, 7) / 10
    expected = (x - torch.arange(7)) / (torch.arange(7) / 2 + 1e-6)
    actual = normalizer.normalize_action(x)
    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual).all() and actual.abs().max() > 1  # No clipping, even at std=0.
    torch.testing.assert_close(actual[..., 6], expected[..., 6])  # Old mask=False is intentionally ignored.
    torch.testing.assert_close(normalizer.denormalize_action(actual), x, rtol=1e-5, atol=1e-6)
    p = torch.randn(2, 8)
    torch.testing.assert_close(normalizer.normalize_proprio(p), (p - torch.arange(8)) / (torch.arange(8) / 2 + 1e-6))
    assert normalizer.normalize_action(x.double()).dtype == torch.float32
    assert normalizer.normalize_action(x).device == x.device
    with pytest.raises(ValueError, match="shape"):
        normalizer.normalize_action(torch.zeros(2, 8))
    bad = stats()
    bad["proprio"]["mean"] = [0.] * 7
    with pytest.raises(ValueError, match="proprio.mean"):
        type(normalizer)(bad)


def test_legacy_action_transform_and_collator_still_present():
    tree = ast.parse((ROOT / "prismatic/vla/datasets/datasets.py").read_text(encoding="utf-8"))
    legacy = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RLDSBatchTransform")
    assert any(isinstance(n, ast.Attribute) and n.attr == "action_tokenizer" for n in ast.walk(legacy))
    assert any(isinstance(n, ast.Name) and n.id == "labels" for n in ast.walk(legacy))
    collator = ast.parse((ROOT / "prismatic/util/data_utils.py").read_text(encoding="utf-8"))
    assert any(isinstance(n, ast.ClassDef) and n.name == "PaddedCollatorForActionPrediction" for n in collator.body)


def test_synthetic_bridge_to_flow_backward(bridge):
    flow = standalone("prismatic/models/flow_action_head.py")
    normalizer = standalone("prismatic/vla/hybrid_normalization.py").HybridZScoreNormalizer(stats())
    data = batch(bridge)
    action = normalizer.normalize_action(data["actions"])
    proprio = normalizer.normalize_proprio(data["proprio"])
    # Constant state coordinates equal their mean here; keep this smoke focused
    # on the bridge, not large synthetic magnitudes from the zero-std test.
    samples = flow.sample_flow_matching_inputs(action)
    head = flow.SimVLAFlowActionHead(vlm_hidden_dim=11, hidden_dim=24, depth=1, num_heads=3)
    features = torch.randn(2, 5, 11, requires_grad=True)
    velocity = head(features, samples["x_t"], proprio, samples["t"], torch.ones(2, 5, dtype=torch.bool))
    loss = flow.compute_flow_matching_loss(velocity, samples["target_velocity"])
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(features.grad).all()
    assert head.action_decoder.weight.grad is not None
