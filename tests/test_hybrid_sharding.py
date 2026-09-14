"""Source-level sharding contracts without installed TensorFlow/TFDS or RLDS data."""

import ast
import copy
from functools import partial
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_hybrid_data import ROOT, RLDS, bridge, definitions, fake_tf, standalone, tree_map


@pytest.fixture
def source():
    events = []
    splits = {}
    def even_splits(base, *, n, drop_remainder):
        events.append(("even_splits", base, n, drop_remainder))
        # Opaque instructions: tests do not rely on TFDS's internal formatting.
        return splits.setdefault((base, n), [object() for _ in range(n)])

    class Dataset:
        def __init__(self, split):
            self.split = split
        @staticmethod
        def from_rlds(builder, *, split, **kwargs):
            events.append(("source", split))
            return Dataset(split)
        def traj_map(self, fn, *args):
            events.append(("traj_map", self.split))
            return self
        def repeat(self):
            events.append(("repeat", self.split))
            return self
        def flatten(self, **kwargs):
            events.append(("flatten", self.split))
            return self
        @staticmethod
        def sample_from_datasets(datasets, weights):
            events.append(("interleave", datasets[0].split))
            return datasets[0]
        def shuffle(self, count):
            events.append(("shuffle", self.split))
            return self
        def with_ram_budget(self, budget):
            return self
        def take(self, count):
            return self
        def cache(self):
            return self

    statistics = dict(action=dict(mean=np.zeros(7), std=np.ones(7)),
                      proprio=dict(mean=np.zeros(8), std=np.ones(8)), num_transitions=100, num_trajectories=10)
    def get_statistics(dataset, **kwargs):
        events.append(("statistics", dataset.split))
        assert dataset.split == "all"
        return copy.deepcopy(statistics)
    def transform(stage):
        def apply(dataset, **kwargs):
            events.append((stage, dataset.split))
            return dataset
        return apply
    ns = dict(tf=fake_tf(), tfds=SimpleNamespace(even_splits=even_splits,
              builder=lambda *a, **kw: SimpleNamespace(info="info", data_dir="unused")),
              dl=SimpleNamespace(DLataset=Dataset), np=np, copy=copy, inspect=inspect, partial=partial,
              tree_map=tree_map, get_dataset_statistics=get_statistics,
              normalize_action_and_proprio=lambda *a, **kw: None,
              allocate_threads=lambda n, weights: [1] * len(weights),
              pprint_data_mixture=lambda *args: None, overwatch=SimpleNamespace(info=lambda *a: None),
              apply_trajectory_transforms=transform("window"),
              apply_per_dataset_frame_transforms=transform("per_dataset_frame"),
              apply_frame_transforms=transform("decode_resize"))
    definitions(RLDS, {"_resolve_rlds_split", "make_dataset_from_rlds", "make_interleaved_dataset"}, ns)
    return SimpleNamespace(ns=ns, events=events, splits=splits, statistics=statistics)


@pytest.mark.parametrize("train,base", [(True, "train"), (False, "val")])
def test_split_resolution(source, train, base):
    resolve = source.ns["_resolve_rlds_split"]
    assert resolve(train=train) == base
    assert resolve(train=train, shard_rank=0, shard_world_size=1) == base
    assert not source.events
    shards = [resolve(train=train, shard_rank=r, shard_world_size=4) for r in range(4)]
    assert len(set(shards)) == 4
    assert shards == source.splits[(base, 4)]
    assert all(event == ("even_splits", base, 4, False) for event in source.events)


@pytest.mark.parametrize("rank,size", [(-1, 4), (4, 4), (0, 0), (0, -1), (0, None), (None, 1),
                                      (None, 4), (1.5, 4), (0, 4.), (True, 4), (0, True), ("0", 4)])
def test_invalid_configuration(source, rank, size):
    with pytest.raises(ValueError):
        source.ns["_resolve_rlds_split"](train=True, shard_rank=rank, shard_world_size=size)
    assert not source.events


@pytest.mark.parametrize("train", [True, False])
def test_statistics_all_actual_source_sharded(source, train):
    _, statistics = source.ns["make_dataset_from_rlds"](
        "libero_spatial_no_noops", "unused", train=train, shard_rank=2, shard_world_size=4,
        normalize_action_proprio=False, action_proprio_normalization_type="bounds_q99")
    split = source.splits[("train" if train else "val", 4)][2]
    assert [event for event in source.events if event[0] == "source"] == [("source", "all"), ("source", split)]
    assert ("statistics", "all") in source.events
    np.testing.assert_equal(statistics, source.statistics)


@pytest.mark.parametrize("rank,size", [(None, None), (0, 1), (3, 4)])
def test_interleaver_global_pass_then_source_before_transforms(source, rank, size):
    _, length, statistics = source.ns["make_interleaved_dataset"](
        [dict(name="libero_spatial_no_noops", data_dir="unused", normalize_action_proprio=False,
              action_proprio_normalization_type="bounds_q99")],
        train=True, shuffle_buffer_size=100, traj_transform_kwargs={"future_action_window_size": 9},
        frame_transform_kwargs={}, shard_rank=rank, shard_world_size=size)
    split = source.splits[("train", 4)][rank] if size == 4 else "train"
    events = [event for event in source.events if event[0] != "even_splits"]
    assert [event for event in events if event[0] == "source"] == [
        ("source", "all"), ("source", "train"), ("source", split)]
    assert events.count(("statistics", "all")) == 1
    stages = [name for name, _ in events]
    assert stages.index("repeat") < stages.index("window") < stages.index("flatten")
    assert stages.index("flatten") < stages.index("interleave") < stages.index("shuffle") < stages.index("decode_resize")
    assert all(value == split for name, value in events if name in ("repeat", "window", "flatten", "shuffle", "decode_resize"))
    assert length == 100  # Never divide the global effective length by rank count.
    assert statistics["libero_spatial_no_noops"]["num_transitions"] == 100


@pytest.mark.parametrize("rank,size", [(0, 1), (2, 4)])
def test_hybrid_forwarding_and_worker_contract(bridge, source, rank, size):
    bridge["tfds"] = source.ns["tfds"]
    dataset = bridge["HybridRLDSDataset"](
        Path("unused"), "libero_spatial_no_noops", lambda x: x, (224, 224), rank=rank, world_size=size)
    config = bridge["captured"][-1]
    assert (config["shard_rank"], config["shard_world_size"]) == (rank, size)
    assert config["traj_transform_kwargs"]["future_action_window_size"] == 9
    assert config["dataset_kwargs_list"][0]["normalize_action_proprio"] is False
    assert dataset.dataset_length == 123
    dataset.dataset = SimpleNamespace(as_numpy_iterator=lambda: iter(["frame"]))
    assert list(dataset) == ["frame"]
    bridge["get_worker_info"] = lambda: SimpleNamespace(id=0, num_workers=2)
    with pytest.raises(RuntimeError, match="worker-level sharding.*num_workers=0"):
        iter(dataset)


def test_hybrid_validation_precedes_parent_construction(bridge):
    with pytest.raises(ValueError):
        bridge["HybridRLDSDataset"](Path("unused"), "libero_spatial_no_noops", None, (224, 224), rank=4, world_size=4)
    assert not bridge["captured"]


def test_interleaver_rejects_misplaced_shard_options(source):
    with pytest.raises(ValueError, match="not dataset_kwargs_list"):
        source.ns["make_interleaved_dataset"](
            [dict(name="libero_spatial_no_noops", shard_rank=1)], train=True,
            shuffle_buffer_size=10, traj_transform_kwargs={}, frame_transform_kwargs={})
    assert not source.events


def test_no_late_sharding_or_hidden_distributed_state():
    for path in (RLDS, "prismatic/vla/datasets/hybrid_datasets.py", "vla-scripts/hybrid_sharding_smoke.py"):
        tree = ast.parse((ROOT / path).read_text())
        attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        assert "shard" not in attrs and "islice" not in attrs and "DistributedSampler" not in names
        if path.endswith("hybrid_datasets.py"):
            assert not attrs & {"get_rank", "get_world_size", "distributed"}
        if path.endswith("hybrid_sharding_smoke.py"):
            assert not attrs & {"backward", "load_state_dict", "cuda", "step"}


def test_smoke_fingerprint_and_pass_conditions():
    smoke = standalone("vla-scripts/hybrid_sharding_smoke.py")
    batch = dict(input_ids=torch.tensor([[1, 2]]), actions=torch.zeros(1, 10, 7),
                 proprio=torch.zeros(1, 8), pixel_values=torch.zeros(1, 12, 224, 224))
    original = smoke.fingerprint(batch)
    assert original == smoke.fingerprint(batch) and len(original) == 64
    for key, index in (("input_ids", (0, 0)), ("actions", (0, 0, 0)), ("proprio", (0, 0)),
                       ("pixel_values", (0, 0, 0, 0)), ("pixel_values", (0, 6, 0, 0))):
        changed = {k: v.clone() for k, v in batch.items()}
        changed[key][index] += 1
        assert smoke.fingerprint(changed) != original
    reports = [dict(rank=r, source_split=f"opaque-{r}", statistics_key=smoke.DATASET_KEY,
                    dimensions_valid=True, error=None, fingerprints=[str(r)] * 2) for r in range(4)]
    result = smoke.summarize(reports, 2)
    assert result["rank_sharding_validated"]
    assert all(r["within_rank_duplicate_count"] == 1 for r in result["ranks"])
    reports[1]["fingerprints"] = reports[0]["fingerprints"]
    assert not smoke.summarize(reports, 2)["rank_sharding_validated"]
    reports[1]["fingerprints"] = ["1"]  # Incomplete consumption fails.
    assert not smoke.summarize(reports, 2)["rank_sharding_validated"]
    with pytest.raises(ValueError, match="actions"):
        smoke.fingerprint({**batch, "actions": torch.zeros(1, 8, 7)})
