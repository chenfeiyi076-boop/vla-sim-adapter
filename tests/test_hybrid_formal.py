"""CPU tests for the formal recipe, orchestration and artifact contracts."""

import ast
import copy
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from test_hybrid_step import ROOT, standalone


formal = standalone("prismatic/training/hybrid_formal.py")
trainer = standalone("vla-scripts/train_hybrid_spatial.py")


def test_normalization_numeric_compatibility():
    expected = formal.normalization_metadata(statistics())
    actual = copy.deepcopy(expected)
    for kind, field, delta in (("action", "mean", 4.8e-7), ("action", "std", 2.4e-6),
                               ("proprio", "mean", 7.2e-6), ("proprio", "std", 2.7e-7)):
        actual[kind][field][0] += delta
    formal.validate_normalization_metadata(expected, expected)
    formal.validate_normalization_metadata(expected, actual, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("kind,field", [("action", "mean"), ("action", "std"), ("proprio", "mean"), ("proprio", "std")])
def test_normalization_material_mismatch(kind, field):
    expected = formal.normalization_metadata(statistics())
    actual = copy.deepcopy(expected)
    actual[kind][field][0] += 1e-3
    with pytest.raises(ValueError, match=rf"{kind}\.{field}, max_abs_diff="):
        formal.validate_normalization_metadata(expected, actual)


@pytest.mark.parametrize("side", ["expected", "actual"])
@pytest.mark.parametrize("error", ["missing_kind", "extra_kind", "missing_field", "extra_field", "dimension", "nan", "inf", "malformed"])
def test_normalization_invalid_metadata(side, error):
    pair = {name: formal.normalization_metadata(statistics()) for name in ("expected", "actual")}
    bad = pair[side]
    if error == "missing_kind":
        del bad["action"]
    elif error == "extra_kind":
        bad["extra"] = {}
    elif error == "missing_field":
        del bad["proprio"]["mean"]
    elif error == "extra_field":
        bad["action"]["mask"] = [True] * 7
    elif error == "dimension":
        bad["proprio"]["mean"] = [0.] * 7
    elif error in ("nan", "inf"):
        bad["action"]["std"][0] = float(error)
    else:
        bad["action"] = None
    with pytest.raises(ValueError):
        formal.validate_normalization_metadata(**pair)


def statistics():
    return {kind: {"mean": [0.123456789] * dim, "std": [1.] * dim, "mask": [False] * dim}
            for kind, dim in (("action", 7), ("proprio", 8))}


def test_lr_exact_boundary_and_resume():
    for step in (0, 29999):
        assert formal.hybrid_learning_rate(step, base_lr=1e-6, decay_step=30000) == 1e-6
    for step in (30000, 40000):
        assert formal.hybrid_learning_rate(step, base_lr=1e-6, decay_step=30000) == 1e-7
    assert formal.hybrid_learning_rate(99999, base_lr=1e-6) == 1e-6
    rates = [formal.hybrid_learning_rate(step, base_lr=1e-6, decay_step=75) for step in range(100)]
    assert rates == [1e-6] * 75 + [1e-7] * 25


@pytest.mark.parametrize("kwargs", [dict(global_step=-1), dict(global_step=True), dict(global_step=1.5),
    dict(base_lr=0), dict(base_lr=float("nan")), dict(base_lr=float("inf")), dict(base_lr=True),
    dict(decay_step=0), dict(decay_step=True), dict(decay_step=1.5), dict(decay_factor=0),
    dict(decay_factor=1.1), dict(decay_factor=float("nan"))])
def test_invalid_lr(kwargs):
    with pytest.raises(ValueError):
        formal.hybrid_learning_rate(**(dict(global_step=0, base_lr=1e-6) | kwargs))


def test_lr_updates_both_groups():
    optimizer = torch.optim.AdamW([{"params": [torch.nn.Parameter(torch.ones(1))]},
                                  {"params": [torch.nn.Parameter(torch.ones(1))]}])
    formal.set_optimizer_learning_rate(optimizer, 1e-7)
    assert [g["lr"] for g in optimizer.param_groups] == [1e-7, 1e-7]


@pytest.mark.parametrize("kind,name", [("action", "mean"), ("action", "std"), ("proprio", "mean"), ("proprio", "std")])
def test_normalization_metadata_shape_validation(kind, name):
    stats = statistics()
    stats[kind][name] = [0.] * 6
    with pytest.raises(ValueError):
        formal.normalization_metadata(stats)


def test_metadata_and_atomic_manifest(tmp_path):
    metadata = formal.build_formal_metadata(source_splits=[str(i) for i in range(4)], statistics=statistics(), max_steps=40000)
    assert metadata["normalization_statistics"]["action"]["mean"] == torch.tensor([.123456789] * 7).tolist()
    assert metadata["image_aug"] is True and metadata["effective_global_batch_size"] == 4
    assert metadata["scheduler"] is metadata["gradient_clipping"] is None
    assert metadata["warmup_steps"] == 0 and "epoch" not in metadata
    assert metadata["lr_policy"] == "single_step_decay" and metadata["lr_decay_step"] == 30000
    assert set(metadata["normalization_statistics"]["proprio"]) == {"mean", "std"}
    formal.prepare_run_manifest(tmp_path, metadata)
    formal.prepare_run_manifest(tmp_path, metadata, resume=True)
    assert json.loads((tmp_path / "run_config.json").read_text()) == metadata
    assert list(tmp_path.iterdir()) == [tmp_path / "run_config.json"]
    with pytest.raises(ValueError, match="incompatible"):
        formal.prepare_run_manifest(tmp_path, metadata | {"max_steps": 100})
    (tmp_path / "train.jsonl").write_text("{}\n")
    with pytest.raises(FileExistsError):
        formal.prepare_run_manifest(tmp_path, metadata)
    formal.prepare_run_manifest(tmp_path, metadata, resume=True)


def test_checkpoint_boundaries():
    assert formal.checkpoint_name(10000) == "step-00010000.pt"
    assert [s for s in range(1, 106) if formal.checkpoint_due(s, max_steps=105, save_every=50)] == [50, 100, 105]
    assert [s for s in range(1, 101) if formal.checkpoint_due(s, max_steps=100, save_every=50)] == [50, 100]


@pytest.mark.parametrize("world_size", [1, 4])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_physical_batch_metadata_and_startup_log(world_size, batch_size, capsys):
    metadata = formal.build_formal_metadata(source_splits=[str(i) for i in range(world_size)],
        statistics=statistics(), max_steps=40000, world_size=world_size, local_batch_size=batch_size)
    assert metadata["effective_global_batch_size"] == world_size * batch_size
    assert metadata["gradient_accumulation"] == 1 and metadata["scheduler"] is None
    # Execute the actual rank-zero startup logging block without loading CUDA/models.
    tree = ast.parse((ROOT / "vla-scripts/train_hybrid_spatial.py").read_text())
    block = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                 and "per_device_batch_size" in ast.unparse(n) and ast.unparse(n.test) == "rank == 0")
    exec(compile(ast.Module(body=[block], type_ignores=[]), "startup", "exec"),
         dict(json=json, rank=0, world_size=world_size, args=SimpleNamespace(per_device_batch_size=batch_size,
              learning_rate=1e-6, flow_head_learning_rate=1e-6)))
    assert json.loads(capsys.readouterr().out) == dict(world_size=world_size,
        per_device_batch_size=batch_size, gradient_accumulation_steps=1,
        effective_global_batch_size=world_size * batch_size, vlm_learning_rate=1e-6, flow_head_learning_rate=1e-6)


@pytest.mark.parametrize("batch_size", [0, -1])
def test_invalid_physical_batch_rejected(batch_size):
    with pytest.raises(SystemExit):
        trainer.parse_args(["--vlm-path", "native", "--hf-config", "config", "--data-root", "data",
            "--run-dir", "run", "--max-steps", "10", "--per-device-batch-size", str(batch_size)])
    with pytest.raises(ValueError, match="local_batch_size"):
        formal.build_formal_metadata(source_splits=["train"], statistics=statistics(),
            max_steps=10, world_size=1, local_batch_size=batch_size)


def test_batch_change_rejects_resume_manifest(tmp_path):
    metadata = formal.build_formal_metadata(source_splits=[str(i) for i in range(4)],
                                           statistics=statistics(), max_steps=40000)
    formal.prepare_run_manifest(tmp_path, metadata)
    changed = formal.build_formal_metadata(source_splits=[str(i) for i in range(4)],
        statistics=statistics(), max_steps=40000, local_batch_size=2)
    with pytest.raises(ValueError, match="incompatible"):
        formal.prepare_run_manifest(tmp_path, changed, resume=True)


@pytest.mark.parametrize("start", [0, 75])
@pytest.mark.parametrize("log_every", [10, 20])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_continuous_loop_and_100_step_schedule(monkeypatch, start, log_every, batch_size):
    monkeypatch.setitem(sys.modules, "prismatic.training.hybrid_formal", formal)
    rates, forwards, logs, saves = [], [], [], []
    logged_rates = {}
    optimizer = SimpleNamespace(param_groups=[{"lr": 0, "lr_role": "vlm"},
        {"lr": 0, "lr_role": "flow_head"}], zero_grad=lambda **kw: None)
    batch = dict(actions=torch.zeros(batch_size, 10, 7), proprio=torch.zeros(batch_size, 8))
    class Loader:
        iterations = 0
        samples = 0
        def __iter__(self):
            self.iterations += 1
            return self
        def __next__(self):
            self.samples += 1
            return batch
    loader = Loader()
    model = object()
    def step(training_model, received, normalizer, opt, **kwargs):
        assert training_model is model and received is batch
        assert kwargs == dict(device_type="cuda", autocast_dtype=torch.bfloat16, autocast_enabled=True)
        rates.append(opt.param_groups[0]["lr"])
        assert opt.param_groups[0]["lr"] == opt.param_groups[1]["lr"]
        forwards.append(1)
        return {"loss": 1.}
    monkeypatch.setitem(sys.modules, "prismatic.training.hybrid_multistep", SimpleNamespace(hybrid_training_step=step))
    args = SimpleNamespace(max_steps=100, learning_rate=1e-6, lr_decay_step=75, lr_decay_factor=.1,
                           log_every=log_every, save_every=50, per_device_batch_size=batch_size)
    def saved(step):
        assert step == start + len(forwards)
        saves.append(step)
    def logged(step, lr, diag):
        logs.append(step)
        logged_rates[step] = lr
    end = trainer.training_loop(model, loader, None, optimizer, args=args, global_step=start,
                               log_callback=logged, checkpoint_callback=saved)
    assert end == 100 and loader.iterations == 1 and loader.samples == 100 - start
    assert rates == ([1e-6] * 75 + [1e-7] * 25)[start:]
    assert saves == ([50, 100] if start == 0 else [100])
    assert logs[-1] == 100
    expected_logs = sorted({s for s in range(start + 1, 101) if s % log_every == 0} |
                           {s for s in (75, 76, 100) if s > start})
    assert logs == expected_logs
    if start == 0:
        assert 75 in logs and logged_rates[75] == 1e-6
    assert 76 in logs and logged_rates[76] == 1e-7


def test_failed_step_does_not_log_or_checkpoint(monkeypatch):
    monkeypatch.setitem(sys.modules, "prismatic.training.hybrid_formal", formal)
    def fail(*args, **kwargs):
        raise RuntimeError("step failed")
    monkeypatch.setitem(sys.modules, "prismatic.training.hybrid_multistep", SimpleNamespace(hybrid_training_step=fail))
    seen = []
    args = SimpleNamespace(max_steps=1, learning_rate=1e-6, lr_decay_step=75, lr_decay_factor=.1,
                           log_every=1, save_every=1, per_device_batch_size=1)
    with pytest.raises(RuntimeError, match="step failed"):
        trainer.training_loop(None, [dict(actions=torch.zeros(1, 10, 7), proprio=torch.zeros(1, 8))], None,
            SimpleNamespace(param_groups=[{"lr_role": "vlm"}]), args=args, global_step=0,
            log_callback=lambda *a: seen.append(a), checkpoint_callback=lambda *a: seen.append(a))
    assert not seen


def test_formal_cli_and_source_policy():
    args = trainer.parse_args(["--vlm-path", "native", "--hf-config", "config", "--data-root", "data",
                               "--run-dir", "run", "--max-steps", "40000"])
    assert args.image_aug and args.learning_rate == 1e-6 and args.lr_decay_step == 30000
    assert args.save_every == 10000 and args.log_every == 20 and args.shuffle_buffer_size == 10000
    assert args.per_device_batch_size == 1 and args.benchmark_warmup_steps is None
    tree = ast.parse((ROOT / "vla-scripts/train_hybrid_spatial.py").read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    dataset = next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id == "HybridRLDSDataset")
    kw = {k.arg: k.value for k in dataset.keywords}
    assert kw["rank"].id == "rank" and kw["world_size"].id == "world_size"
    assert kw["image_aug"].attr == "image_aug"
    loader = next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id == "DataLoader")
    loader_kw = {k.arg: k.value for k in loader.keywords}
    assert ast.literal_eval(loader_kw["num_workers"]) == 0
    assert ast.unparse(loader_kw["batch_size"]) == "args.per_device_batch_size"
    for path in ("prismatic/training/hybrid_formal.py", "vla-scripts/train_hybrid_spatial.py"):
        identifiers = {n.id for n in ast.walk(ast.parse((ROOT / path).read_text())) if isinstance(n, ast.Name)}
        assert not identifiers & {"GradScaler", "DistributedSampler", "StepLR", "CosineAnnealingLR", "clip_grad_norm_"}


def test_distributed_initialization_precedes_prismatic_imports():
    tree = ast.parse((ROOT / "vla-scripts/train_hybrid_spatial.py").read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    calls = {ast.unparse(n.func): n.lineno for n in ast.walk(main) if isinstance(n, ast.Call)}
    first_import = min(n.lineno for n in ast.walk(main) if isinstance(n, ast.ImportFrom)
                       and n.module.startswith("prismatic."))
    assert calls["torch.cuda.set_device"] < calls["dist.init_process_group"]
    assert calls["dist.init_process_group"] < calls["tf.config.set_visible_devices"] < first_import
    assert not any(isinstance(n, ast.ImportFrom) and n.module.startswith("prismatic.") for n in tree.body)
