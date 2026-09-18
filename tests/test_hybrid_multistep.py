"""CPU synthetic multi-step contracts; no CUDA, TFDS, NCCL, weights or network."""

import ast
from collections import OrderedDict
from contextlib import nullcontext
import copy
import math
import json
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from test_hybrid_step import ROOT, inputs, standalone, training


step_module = standalone("prismatic/training/hybrid_multistep.py")
smoke = standalone("vla-scripts/hybrid_multistep_smoke.py")


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


@pytest.fixture
def setup(training, inputs, monkeypatch):
    encoder, head, batch, normalizer, _ = inputs
    encoder.projector = nn.Sequential(OrderedDict(fc1=encoder.projector))
    monkeypatch.setitem(sys.modules, "prismatic.training.hybrid_step", training)
    wrapper = standalone("prismatic/training/hybrid_ddp.py")
    model = wrapper.HybridFlowTrainingModule(encoder, head)
    optimizer = CountAdamW(training.hybrid_parameter_groups(encoder, head), lr=1e-6)
    return FakeDDP(model), batch, normalizer, optimizer


@pytest.mark.parametrize("autocast", [False, True])
def test_consecutive_steps_enter_wrapper_and_return_only_scalars(setup, autocast):
    ddp, batch, normalizer, optimizer = setup
    before = {n: p.detach().clone() for n, p in ddp.module.named_parameters()}
    for i in range(1, 4):
        result = step_module.hybrid_training_step(ddp, batch, normalizer, optimizer,
                                                  device_type="cpu", autocast_enabled=autocast)
        assert ddp.calls == optimizer.steps == optimizer.zeros == i
        assert isinstance(result["loss"], float) and math.isfinite(result["loss"])
        assert all(not isinstance(value, torch.Tensor) for value in result.values())
        assert result["missing_trainable_grads"] == []
        assert result["excluded_no_grad"] and result["action_queries_no_grad"]
        assert result["encoder_grad_norm"] > 0 and result["flow_head_grad_norm"] > 0
        assert all(p.grad is None for p in ddp.module.parameters() if not p.requires_grad)
    for prefix in ("encoder.", "flow_head."):
        assert any(not torch.equal(p, before[n]) for n, p in ddp.module.named_parameters() if n.startswith(prefix))
    assert smoke.optimizer_step_range(optimizer) == (3, 3)


@pytest.mark.parametrize("failure", ["missing", "nan_grad", "inf_grad", "zero_grad", "excluded_grad",
                                      "query_grad", "nan_loss", "vector_loss"])
def test_invalid_loss_or_gradient_never_steps(setup, failure):
    ddp, batch, normalizer, optimizer = setup
    module = ddp.module
    if failure == "missing":
        module.encoder.register_parameter("disconnected", nn.Parameter(torch.ones(1)))
        optimizer.param_groups[0]["params"].append(module.encoder.disconnected)
    elif failure in ("nan_grad", "inf_grad", "zero_grad"):
        value = {"nan_grad": float("nan"), "inf_grad": float("inf"), "zero_grad": 0.}[failure]
        targets = list(module.flow_head.parameters()) if failure == "zero_grad" else [module.flow_head.action_decoder.weight]
        for parameter in targets:
            parameter.register_hook(lambda g: g * value)
    elif failure in ("excluded_grad", "query_grad"):
        target = module.encoder.action_queries.weight if failure == "query_grad" else module.encoder.proprio_projector.weight
        target.grad = torch.ones_like(target)
    else:
        original = module.forward
        def invalid(*args):
            loss, diagnostics = original(*args)
            return (loss * float("nan") if failure == "nan_loss" else loss[None]), diagnostics
        module.forward = invalid
    before = [p.detach().clone() for p in module.parameters()]
    with pytest.raises((RuntimeError, ValueError)):
        step_module.hybrid_training_step(ddp, batch, normalizer, optimizer, device_type="cpu", autocast_enabled=False)
    assert optimizer.steps == 0 and not optimizer.state
    for p, old in zip(module.parameters(), before):
        assert torch.equal(p, old)


@pytest.mark.parametrize("bad", ["missing", "excluded", "duplicate"])
def test_optimizer_exactness_before_forward(setup, bad):
    ddp, batch, normalizer, optimizer = setup
    params = optimizer.param_groups[0]["params"]
    if bad == "missing":
        params.pop()
    elif bad == "excluded":
        params.append(ddp.module.encoder.action_queries.weight)
    else:
        params.append(params[0])
    with pytest.raises(ValueError, match="exactly"):
        step_module.hybrid_training_step(ddp, batch, normalizer, optimizer, device_type="cpu")
    assert optimizer.steps == ddp.calls == optimizer.zeros == 0


def test_plain_outer_module_supported(setup):
    ddp, batch, normalizer, optimizer = setup
    result = step_module.hybrid_training_step(ddp.module, batch, normalizer, optimizer,
                                              device_type="cpu", autocast_enabled=False)
    assert math.isfinite(result["loss"]) and optimizer.steps == 1


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("benchmark", [False, True])
def test_formal_loop_physical_batch_one_update(setup, monkeypatch, capsys, batch_size, benchmark):
    ddp, batch, normalizer, optimizer = setup
    batch = {k: v[:batch_size] for k, v in batch.items()}
    trainer = standalone("vla-scripts/train_hybrid_spatial.py")
    formal = standalone("prismatic/training/hybrid_formal.py")
    monkeypatch.setitem(sys.modules, "prismatic.training.hybrid_formal", formal)
    shapes = []
    def record_shapes(module, args, output):
        features, noisy, proprio, t, mask = args
        shapes.append((noisy.shape, proprio.shape, t.shape, output.shape))
    ddp.module.flow_head.register_forward_hook(record_shapes)
    def cpu_step(*args, **kwargs):
        return step_module.hybrid_training_step(*args, device_type="cpu", autocast_enabled=True)
    monkeypatch.setitem(sys.modules, "prismatic.training.hybrid_multistep",
                        SimpleNamespace(hybrid_training_step=cpu_step))
    logs, saves = [], []
    args = SimpleNamespace(max_steps=3, per_device_batch_size=batch_size, learning_rate=1e-6,
        lr_decay_step=30000, lr_decay_factor=.1, log_every=1, save_every=3)
    if benchmark:
        args.benchmark_warmup_steps = 1
        # Warmup is deliberately slow; measured steps have unequal data/compute times.
        ticks = iter([0, 10, 11, 31, 32,
                      40, 42, 42.01, 48.01, 48.02,
                      50, 54, 54.01, 64.01, 64.02])
        monkeypatch.setattr(trainer.time, "perf_counter", lambda: next(ticks))
        sync_calls = []
        monkeypatch.setattr(torch.cuda, "synchronize", lambda: sync_calls.append(optimizer.steps))
        monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 2**30)
        monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 2 * 2**30)
        monkeypatch.setattr(trainer.dist, "get_rank", lambda: 0)
    end = trainer.training_loop(ddp, [batch] * 3, normalizer, optimizer, args=args, global_step=0,
        log_callback=lambda s, lr, diag: logs.append((s, lr, diag)), checkpoint_callback=saves.append)
    assert end == ddp.calls == optimizer.steps == 3
    assert smoke.optimizer_step_range(optimizer) == (3, 3)
    assert shapes == [((batch_size, 10, 7), (batch_size, 8), (batch_size,), (batch_size, 10, 7))] * 3
    assert [s for s, _, _ in logs] == [1, 2, 3] and saves == [3]
    assert all(lr == 1e-6 and diag["batch_size"] == batch_size and math.isfinite(diag["loss"])
               for _, lr, diag in logs)
    assert optimizer.zeros == 6  # One pre-backward clear and the existing post-update cleanup per step.
    if benchmark:
        report = json.loads(capsys.readouterr().out)
        assert report == pytest.approx(dict(benchmark_rank=0, measured_steps=2,
            seconds_per_optimizer_step=11.02, samples_per_sec=batch_size / 11.02,
            mean_data_wait_sec=3., mean_compute_sec=8.,
            data_wait_fraction=3. / 11.02, compute_fraction=8. / 11.02,
            cuda_peak_allocated_gib=1., cuda_peak_reserved_gib=2.))
        assert sync_calls == [0, 1, 1, 2, 2, 3]
        assert report["seconds_per_optimizer_step"] - report["mean_data_wait_sec"] - report["mean_compute_sec"] == pytest.approx(.02)


def test_runner_continuous_iterator_and_memory_reporting(setup, monkeypatch):
    ddp, batch, normalizer, optimizer = setup
    batch = {k: v[:1] if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    class Loader:
        iters = 0
        samples = 0
        def __iter__(self):
            self.iters += 1
            return self
        def __next__(self):
            self.samples += 1
            return batch
    loader = Loader()
    sync_calls = []
    def sync(value, name, device):
        sync_calls.append(name)
        assert torch.isfinite(value).all()
        return 0.
    # Run the real utility and real optimizer on CPU; only CUDA/collective surfaces are fakes.
    def cpu_step(*args, **kwargs):
        kwargs["device_type"] = "cpu"
        return step_module.hybrid_training_step(*args, **kwargs)
    monkeypatch.setitem(sys.modules, "hybrid_ddp_smoke", SimpleNamespace(synchronized_slice=sync))
    monkeypatch.setitem(sys.modules, "prismatic.training.hybrid_multistep", SimpleNamespace(hybrid_training_step=cpu_step))
    monkeypatch.setattr(smoke, "dist", SimpleNamespace(get_world_size=lambda: 4, get_rank=lambda: 0,
        all_reduce=lambda tensor, op: tensor.mul_(4), ReduceOp=SimpleNamespace(SUM="sum")))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    allocations = iter([1., 2., 3., 4., 4.1])
    def allocated(device):
        assert all(p.grad is None for p in ddp.parameters())
        return next(allocations) * 2**30
    monkeypatch.setattr(torch.cuda, "memory_allocated", allocated)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 6 * 2**30)
    report, sync_records = smoke.run_steps(ddp, loader, normalizer, optimizer, "cpu", 5)
    assert loader.iters == 1 and loader.samples == ddp.calls == optimizer.steps == 5
    assert optimizer.zeros == 10  # Before forward and after distributed diagnostics.
    assert len(sync_calls) == 20 and len(sync_records) == 5
    assert report["steady_memory_span_gib"] == pytest.approx(.1)
    assert report["encoder_changed"] and report["flow_head_changed"]
    assert report["optimizer_state_step_min"] == report["optimizer_state_step_max"] == 5


def acceptance_report(steps=20):
    return dict(rank=0, samples_consumed=steps, losses=[1.] * steps,
                encoder_all_steps_finite_nonzero=True, flow_head_all_steps_finite_nonzero=True,
                all_steps_missing_trainable_grads=[], excluded_no_grad_all_steps=True,
                action_queries_no_grad_all_steps=True, encoder_changed=True, flow_head_changed=True,
                optimizer_state_step_min=steps, optimizer_state_step_max=steps,
                steady_memory_span_gib=0.1 if steps > 3 else None)


@pytest.mark.parametrize("failure", [None, "memory", "state", "sync", "split", "unchanged"])
def test_formal_acceptance_and_failures(failure):
    reports = [dict(acceptance_report(), rank=rank) for rank in range(4)]
    records = [dict(global_step=i, global_mean_loss=1., encoder_gradient_max_diff=0., flow_head_gradient_max_diff=0.,
                    encoder_parameter_max_diff=0., flow_head_parameter_max_diff=0.) for i in range(1, 21)]
    splits = [str(i) for i in range(4)]
    if failure == "memory":
        reports[3]["steady_memory_span_gib"] = .5001
    elif failure == "state":
        reports[0]["optimizer_state_step_min"] = 19
    elif failure == "sync":
        records[0]["encoder_gradient_max_diff"] = 1e-4
    elif failure == "split":
        splits[1] = splits[0]
    elif failure == "unchanged":
        reports[0]["flow_head_changed"] = False
    result = smoke.summarize(reports, records, splits, 20)
    assert result["phase6b_passed"] == (failure is None)
    assert smoke.memory_span([100., 50., 20., 4., 4.5]) == .5


def test_no_forbidden_behaviors_or_per_parameter_host_sync():
    tree = ast.parse((ROOT / "prismatic/training/hybrid_multistep.py").read_text())
    imports = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    imports.update(a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names)
    assert imports == {"math", "torch"}
    for loop in (n for n in ast.walk(tree) if isinstance(n, (ast.For, ast.ListComp))):
        assert not any(isinstance(n, ast.Attribute) and n.attr in {"item", "cpu", "tolist"} for n in ast.walk(loop))
    runner = ast.parse((ROOT / "vla-scripts/hybrid_multistep_smoke.py").read_text())
    attrs = {n.attr for n in ast.walk(runner) if isinstance(n, ast.Attribute)}
    assert not attrs & {"empty_cache", "save", "load_state_dict", "encode_observation", "no_sync"}
    calls = [n for n in ast.walk(runner) if isinstance(n, ast.Call)]
    dataset = next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id == "HybridRLDSDataset")
    kw = {k.arg: k.value for k in dataset.keywords}
    assert kw["rank"].id == "rank" and kw["world_size"].id == "world_size"
    ddp = next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id == "DDP")
    kw = {k.arg: k.value for k in ddp.keywords}
    assert ast.literal_eval(kw["find_unused_parameters"]) is False
    assert ast.literal_eval(kw["gradient_as_bucket_view"]) is True
