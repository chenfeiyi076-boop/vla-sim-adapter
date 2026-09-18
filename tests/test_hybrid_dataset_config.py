"""Dataset identity tests only; no TFDS, CUDA or real non-Spatial data."""

import ast
import sys
from types import SimpleNamespace

import pytest
import torch

from test_hybrid_step import ROOT, standalone
from test_hybrid_formal import statistics


formal = standalone("prismatic/training/hybrid_formal.py")
trainer = standalone("vla-scripts/train_hybrid_spatial.py")
CLI = ["--vlm-path", "native", "--hf-config", "config", "--data-root", "data",
       "--run-dir", "run", "--max-steps", "40000"]


def metadata(**kwargs):
    return formal.build_formal_metadata(source_splits=[str(i) for i in range(4)],
                                       statistics=statistics(), max_steps=40000, **kwargs)


def test_default_and_explicit_spatial_identity():
    default = trainer.parse_args(CLI)
    explicit = trainer.parse_args(CLI + ["--dataset-key", "libero_spatial_no_noops"])
    assert vars(default) == vars(explicit)
    assert default.dataset_key == "libero_spatial_no_noops"
    assert metadata() == metadata(dataset_key=default.dataset_key)
    assert metadata()["experiment"] == "hybrid_spatial_formal_v1"
    assert metadata()["dataset_key"] == "libero_spatial_no_noops"


@pytest.mark.parametrize("key,suite", [("libero_spatial_no_noops", "spatial"),
    ("libero_object_no_noops", "object"), ("libero_goal_no_noops", "goal"), ("libero_10_no_noops", "10")])
def test_registered_keys_and_metadata(key, suite):
    assert trainer.parse_args(CLI + ["--dataset-key", key]).dataset_key == key
    result = metadata(dataset_key=key)
    assert result["experiment"] == f"hybrid_{suite}_formal_v1"
    assert result["dataset_key"] == key
    assert {k: v for k, v in result.items() if k not in ("experiment", "dataset_key")} == {
        k: v for k, v in metadata().items() if k not in ("experiment", "dataset_key")}
    for file in ("configs.py", "mixtures.py", "transforms.py"):
        tree = ast.parse((ROOT / "prismatic/vla/datasets/rlds/oxe" / file).read_text())
        dict_keys = {k.value for n in ast.walk(tree) if isinstance(n, ast.Dict)
                     for k in n.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}
        assert key in dict_keys


@pytest.mark.parametrize("key", ["unknown", "libero_long_no_noops", "libero_4_task_suites_no_noops", ""])
def test_unknown_key_fails_without_fallback(key):
    with pytest.raises(SystemExit):
        trainer.parse_args(CLI + ["--dataset-key", key])
    with pytest.raises(ValueError, match="Unsupported Hybrid formal dataset key"):
        metadata(dataset_key=key)


@pytest.mark.parametrize("key", list(formal.HYBRID_DATASET_CONFIGS))
@pytest.mark.parametrize("missing", [False, True])
def test_make_data_selected_key_and_statistics(monkeypatch, key, missing):
    args = trainer.parse_args(CLI + ["--dataset-key", key])
    selected = statistics()
    entries = {} if missing else {key: selected}
    dataset = SimpleNamespace(dataset_statistics=entries, resolved_source_split="train[0:25%]")
    calls = {}
    def create(root, received_key, transform, **kwargs):
        calls["dataset"] = (root, received_key, kwargs)
        return dataset
    def loader(received, **kwargs):
        assert received is dataset
        calls["loader"] = kwargs
        return "loader"
    monkeypatch.setitem(sys.modules, "prismatic.vla.datasets.hybrid_datasets", SimpleNamespace(
        HybridRLDSDataset=create, HybridRLDSBatchTransform=lambda *a: a,
        PaddedCollatorForHybridFlow=lambda pad: pad))
    monkeypatch.setitem(sys.modules, "prismatic.vla.hybrid_normalization", SimpleNamespace(
        HybridZScoreNormalizer=lambda stats: calls.setdefault("normalizer", stats)))
    monkeypatch.setattr(torch.utils.data, "DataLoader", loader)
    invoke = lambda: trainer.make_data(args, SimpleNamespace(pad_token_id=0),
                                      SimpleNamespace(apply_transform=None), rank=0, world_size=4)
    if missing:
        with pytest.raises(ValueError, match=f"Dataset statistics do not contain requested dataset key: {key}"):
            invoke()
        assert "loader" not in calls and "normalizer" not in calls
    else:
        result = invoke()
        assert result[2] is selected and calls["normalizer"] is selected
        assert calls["loader"]["num_workers"] == 0 and calls["loader"]["batch_size"] == 1
    assert calls["dataset"][0:2] == (args.data_root, key)
    assert calls["dataset"][2]["rank"] == 0 and calls["dataset"][2]["world_size"] == 4


def test_cross_dataset_resume_manifest_rejected(tmp_path):
    formal.prepare_run_manifest(tmp_path, metadata())
    formal.prepare_run_manifest(tmp_path, metadata(dataset_key="libero_spatial_no_noops"), resume=True)
    with pytest.raises(ValueError, match="incompatible"):
        formal.prepare_run_manifest(tmp_path, metadata(dataset_key="libero_object_no_noops"), resume=True)


def test_main_passes_selected_key_to_metadata():
    tree = ast.parse((ROOT / "vla-scripts/train_hybrid_spatial.py").read_text())
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == "build_formal_metadata")
    assert ast.unparse(next(k.value for k in call.keywords if k.arg == "dataset_key")) == "args.dataset_key"
