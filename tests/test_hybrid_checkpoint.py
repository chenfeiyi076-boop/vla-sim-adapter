"""Tiny CPU training-state persistence; RNG control is a test technique only."""

import ast
import copy

import pytest
import torch

from test_hybrid_step import ROOT, inputs, standalone, training
from test_hybrid_multistep import setup, step_module, FakeDDP, CountAdamW


checkpoint = standalone("prismatic/training/hybrid_checkpoint.py")
META = dict(dataset_key="libero_spatial_no_noops", world_size=4, local_batch_size=1,
            effective_global_batch_size=4, action_horizon=10, action_dim=7, proprio_dim=8,
            optimizer="AdamW", learning_rate=1e-6, gradient_accumulation=1, scheduler=None,
            source_splits=["split0", "split1", "split2", "split3"])


def train(bundle, count=3):
    model, batch, normalizer, optimizer = bundle
    for _ in range(count):
        step_module.hybrid_training_step(model, batch, normalizer, optimizer, device_type="cpu", autocast_enabled=False)


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


@pytest.fixture
def saved(setup, tmp_path):
    train(setup)
    model, _, _, optimizer = setup
    path = tmp_path / "hybrid.pt"
    checkpoint.save_hybrid_checkpoint(path, model, optimizer, global_step=3, metadata=META)
    return setup, path


def test_full_round_trip(saved):
    bundle, path = saved
    model, _, _, optimizer = bundle
    expected_model = copy.deepcopy(model.module.state_dict())
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    with torch.no_grad():
        for p in model.parameters():
            p.add_(3)
    train(bundle, 1)
    result = checkpoint.load_hybrid_checkpoint(path, model, optimizer, expected_metadata=META)
    assert result == dict(format_version=1, global_step=3, metadata=META)
    equal(model.module.state_dict(), expected_model)
    equal(optimizer.state_dict(), expected_optimizer)
    assert all(state["step"].item() == 3 for state in optimizer.state.values())


def test_true_next_update_continuation(saved, training):
    bundle, path = saved
    a, batch, normalizer, optimizer_a = bundle
    b = FakeDDP(copy.deepcopy(a.module))
    with torch.no_grad():
        for p in b.parameters():
            p.zero_()
    optimizer_b = CountAdamW(training.hybrid_parameter_groups(b.module.encoder, b.module.flow_head), lr=1e-6)
    checkpoint.load_hybrid_checkpoint(path, b, optimizer_b, expected_metadata=META)
    equal(a.module.state_dict(), b.module.state_dict())
    equal(optimizer_a.state_dict(), optimizer_b.state_dict())
    rng = torch.random.get_rng_state()
    train(bundle, 1)
    torch.random.set_rng_state(rng)
    train((b, batch, normalizer, optimizer_b), 1)
    equal(a.module.state_dict(), b.module.state_dict())
    equal(optimizer_a.state_dict(), optimizer_b.state_dict())


@pytest.mark.parametrize("bad", ["missing", "excluded", "group_order", "parameter_order", "duplicate"])
def test_layout_rejected_before_model_mutation(saved, bad):
    bundle, path = saved
    model, _, _, optimizer = bundle
    with torch.no_grad():
        model.module.encoder.projector.fc1.weight.add_(1)
    before = copy.deepcopy(model.module.state_dict())
    params = optimizer.param_groups[0]["params"]
    if bad == "missing":
        params.pop()
    elif bad == "excluded":
        params.append(model.module.encoder.action_queries.weight)
    elif bad == "group_order":
        optimizer.param_groups.reverse()
    elif bad == "parameter_order":
        params.reverse()
    else:
        params.append(params[0])
    with pytest.raises(ValueError):
        checkpoint.load_hybrid_checkpoint(path, model, optimizer)
    equal(model.module.state_dict(), before)


@pytest.mark.parametrize("key,value", [("dataset_key", "other"), ("world_size", 2), ("action_horizon", 8),
                                       ("action_dim", 6), ("proprio_dim", 7), ("learning_rate", 1e-4),
                                       ("source_splits", ["wrong"] * 4)])
def test_metadata_mismatch_before_restore(saved, key, value):
    bundle, path = saved
    model, _, _, optimizer = bundle
    before = copy.deepcopy(model.module.state_dict())
    with pytest.raises(ValueError, match=key):
        checkpoint.load_hybrid_checkpoint(path, model, optimizer, expected_metadata={key: value})
    equal(model.module.state_dict(), before)


@pytest.mark.parametrize("bad", ["not_dict", "missing_key", "version", "negative", "bool", "layout", "step_mismatch", "missing_step"])
def test_invalid_payload(saved, bad):
    bundle, path = saved
    model, _, _, optimizer = bundle
    payload = torch.load(path, weights_only=True)
    if bad == "not_dict":
        payload = []
    elif bad == "missing_key":
        del payload["encoder"]
    elif bad == "version":
        payload["format_version"] = 2
    elif bad in ("negative", "bool", "step_mismatch"):
        payload["global_step"] = {"negative": -1, "bool": True, "step_mismatch": 4}[bad]
    elif bad == "layout":
        payload["optimizer_parameter_names"] = [{"parameters": "broken"}]
    else:
        del next(iter(payload["optimizer"]["state"].values()))["step"]
    torch.save(payload, path)
    before = copy.deepcopy(model.module.state_dict())
    with pytest.raises(ValueError):
        checkpoint.load_hybrid_checkpoint(path, model, optimizer)
    equal(model.module.state_dict(), before)


@pytest.mark.parametrize("component", ["encoder", "flow_head"])
@pytest.mark.parametrize("bad", ["missing", "extra", "shape"])
def test_strict_model_state(saved, component, bad):
    bundle, path = saved
    model, _, _, optimizer = bundle
    payload = torch.load(path, weights_only=True)
    state = payload[component]
    key = next(iter(state))
    if bad == "missing":
        del state[key]
    elif bad == "extra":
        state["unexpected.weight"] = torch.ones(1)
    else:
        state[key] = torch.ones(123)
    torch.save(payload, path)
    with pytest.raises(RuntimeError):
        checkpoint.load_hybrid_checkpoint(path, model, optimizer)


def test_cpu_snapshot_and_atomic_success(setup, tmp_path, monkeypatch):
    train(setup)
    model, _, _, optimizer = setup
    path = tmp_path / "new" / "model.pt"
    original_save = torch.save
    captured = []
    def capture(payload, stream):
        snapshot = copy.deepcopy(payload)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1)
        for state in optimizer.state.values():
            state["exp_avg"].add_(2)
        equal(payload, snapshot)
        def assert_cpu(value):
            if isinstance(value, torch.Tensor):
                assert value.device.type == "cpu" and not value.requires_grad
            elif isinstance(value, dict):
                for x in value.values():
                    assert_cpu(x)
            elif isinstance(value, (tuple, list)):
                for x in value:
                    assert_cpu(x)
        assert_cpu(payload)
        captured.append(snapshot)
        original_save(payload, stream)
    monkeypatch.setattr(torch, "save", capture)
    checkpoint.save_hybrid_checkpoint(path, model.module, optimizer, global_step=3, metadata=META)
    assert list(path.parent.iterdir()) == [path]
    equal(torch.load(path, weights_only=True), captured[0])
    with pytest.raises(FileExistsError):
        checkpoint.save_hybrid_checkpoint(path, model, optimizer, global_step=3, metadata=META)
    assert len(captured) == 1


@pytest.mark.parametrize("existing", [False, True])
def test_atomic_save_failure_preserves_final(setup, tmp_path, monkeypatch, existing):
    model, _, _, optimizer = setup
    path = tmp_path / "model.pt"
    if existing:
        checkpoint.save_hybrid_checkpoint(path, model, optimizer, global_step=0, metadata=META)
    before = path.read_bytes() if existing else None
    def fail(payload, stream):
        stream.write(b"partial")
        raise OSError("simulated disk failure")
    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="disk failure"):
        checkpoint.save_hybrid_checkpoint(path, model, optimizer, global_step=0, metadata=META, overwrite=existing)
    assert list(tmp_path.iterdir()) == ([path] if existing else [])
    assert (path.read_bytes() if path.exists() else None) == before


@pytest.mark.parametrize("step,metadata", [(-1, {}), (True, {}), (1.5, {}), (0, [])])
def test_invalid_save_never_creates_directory(setup, tmp_path, step, metadata):
    model, _, _, optimizer = setup
    path = tmp_path / "not_created" / "model.pt"
    with pytest.raises(ValueError):
        checkpoint.save_hybrid_checkpoint(path, model, optimizer, global_step=step, metadata=metadata)
    assert not path.parent.exists()


def test_no_forbidden_dependencies():
    tree = ast.parse((ROOT / "prismatic/training/hybrid_checkpoint.py").read_text())
    imports = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    imports.update(a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names)
    assert imports == {"copy", "os", "pathlib", "tempfile", "torch"}
    runner = ast.parse((ROOT / "vla-scripts/hybrid_checkpoint_smoke.py").read_text())
    attrs = {n.attr for n in ast.walk(runner) if isinstance(n, ast.Attribute)}
    assert not attrs & {"encode_observation", "save_state", "load_state_dict", "no_sync"}
    calls = [n for n in ast.walk(runner) if isinstance(n, ast.Call)]
    assert sum(isinstance(n.func, ast.Name) and n.func.id == "save_hybrid_checkpoint" for n in calls) == 1
    assert sum(isinstance(n.func, ast.Name) and n.func.id == "iter" for n in calls) == 1


def test_eval_only_full_state_without_optimizer(saved):
    bundle, path = saved
    model, _, _, _ = bundle
    encoder = copy.deepcopy(model.module.encoder)
    head = copy.deepcopy(model.module.flow_head)
    with torch.no_grad():
        for p in list(encoder.parameters()) + list(head.parameters()):
            p.zero_()
    info = checkpoint.load_hybrid_model_checkpoint(path, encoder, head, expected_metadata=META)
    assert info == dict(format_version=1, global_step=3, metadata=META)
    equal(encoder.state_dict(), model.module.encoder.state_dict())
    equal(head.state_dict(), model.module.flow_head.state_dict())


@pytest.mark.parametrize("bad", ["metadata", "version", "missing_schema", "encoder_key", "flow_shape"])
def test_eval_loader_rejections(saved, bad):
    bundle, path = saved
    model, _, _, _ = bundle
    encoder, head = model.module.encoder, model.module.flow_head
    before = copy.deepcopy(model.module.state_dict())
    payload = torch.load(path, weights_only=True)
    expected = META
    if bad == "metadata":
        expected = META | {"world_size": 2}
    elif bad == "version":
        payload["format_version"] = 2
    elif bad == "missing_schema":
        del payload["metadata"]
    elif bad == "encoder_key":
        payload["encoder"]["unexpected.weight"] = torch.zeros(1)
    else:
        payload["flow_head"][next(iter(payload["flow_head"]))] = torch.zeros(999)
    torch.save(payload, path)
    with pytest.raises((ValueError, RuntimeError)):
        checkpoint.load_hybrid_model_checkpoint(path, encoder, head, expected_metadata=expected)
    if bad in ("metadata", "version", "missing_schema"):
        equal(model.module.state_dict(), before)
