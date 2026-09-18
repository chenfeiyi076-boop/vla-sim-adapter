"""Role-based Hybrid LR policy, logging and real CPU AdamW resume contracts."""

import copy
import ast
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from test_hybrid_step import inputs, standalone, training
from test_hybrid_step import ROOT
from test_hybrid_multistep import setup, step_module
from test_hybrid_formal import statistics


formal = standalone("prismatic/training/hybrid_formal.py")
trainer = standalone("vla-scripts/train_hybrid_spatial.py")
checkpoint = standalone("prismatic/training/hybrid_checkpoint.py")
CLI = ["--vlm-path", "native", "--hf-config", "config", "--data-root", "data",
       "--run-dir", "run", "--max-steps", "5"]


def metadata(head=None):
    return formal.build_formal_metadata(source_splits=[str(i) for i in range(4)],
        statistics=statistics(), max_steps=5, lr_decay_step=3, flow_head_learning_rate=head)


def test_cli_default_and_metadata_compatibility():
    args = trainer.parse_args(CLI)
    assert args.learning_rate == args.flow_head_learning_rate == 1e-6
    assert trainer.parse_args(CLI + ["--learning-rate", "2e-6"]).flow_head_learning_rate == 2e-6
    args = trainer.parse_args(CLI + ["--flow-head-learning-rate", "1e-5"])
    assert args.learning_rate == 1e-6 and args.flow_head_learning_rate == 1e-5
    assert metadata() == metadata(1e-6)
    assert "flow_head_learning_rate" not in metadata()
    assert metadata(1e-5) == dict(metadata(), flow_head_learning_rate=1e-5)


@pytest.mark.parametrize("flag", ["--learning-rate", "--flow-head-learning-rate"])
@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_bad_cli_rate_rejected(flag, value):
    with pytest.raises(SystemExit):
        trainer.parse_args(CLI + [flag, value])


def test_multiple_shuffled_groups_and_invalid_roles():
    groups = [{"lr_role": role, "lr": 9.} for role in ("flow_head", "vlm", "flow_head", "vlm")]
    opt = SimpleNamespace(param_groups=groups)
    for step in (0, 2, 3, 4):
        vlm = formal.hybrid_learning_rate(step, base_lr=1e-6, decay_step=3)
        head = formal.hybrid_learning_rate(step, base_lr=1e-5, decay_step=3)
        formal.set_optimizer_learning_rates(opt, vlm_lr=vlm, flow_head_lr=head)
        factor = 1 if step < 3 else .1
        for group in groups:
            assert group["lr"] == pytest.approx((1e-6 if group["lr_role"] == "vlm" else 1e-5) * factor)
    for bad in ({}, {"lr_role": "unknown"}):
        invalid = SimpleNamespace(param_groups=[dict(lr_role="vlm", lr=9.), bad])
        with pytest.raises(ValueError, match="lr_role"):
            formal.set_optimizer_learning_rates(invalid, vlm_lr=1e-6, flow_head_lr=1e-5)
        assert invalid.param_groups[0]["lr"] == 9.  # Validate before mutating any LR.


def test_role_parameter_coverage_and_initial_lr(training, inputs):
    encoder, head, *_ = inputs
    groups = training.hybrid_parameter_groups(encoder, head)
    formal.set_group_learning_rates(groups, vlm_lr=1e-6, flow_head_lr=1e-5)
    opt = torch.optim.AdamW(groups, lr=1e-6)
    all_ids = [id(p) for g in groups for p in g["params"]]
    assert len(all_ids) == len(set(all_ids))
    for role, module, lr in (("vlm", encoder, 1e-6), ("flow_head", head, 1e-5)):
        assert {id(p) for g in groups if g["lr_role"] == role for p in g["params"]} == {
            id(p) for p in module.parameters() if p.requires_grad}
        assert all(g["lr"] == lr for g in opt.param_groups if g["lr_role"] == role)
    assert all(g["betas"] == (.9, .999) and g["weight_decay"] == .01 for g in opt.param_groups)


def test_real_loop_decay_and_log_fields(setup, monkeypatch):
    model, batch, normalizer, opt = setup
    monkeypatch.setitem(sys.modules, "prismatic.training.hybrid_formal", formal)
    def step(*args, **kwargs):
        return step_module.hybrid_training_step(*args, device_type="cpu", autocast_enabled=True)
    monkeypatch.setitem(sys.modules, "prismatic.training.hybrid_multistep", SimpleNamespace(hybrid_training_step=step))
    args = trainer.parse_args(CLI + ["--flow-head-learning-rate", "1e-5", "--lr-decay-step", "3",
                                   "--log-every", "1", "--per-device-batch-size", "2"])
    records = []
    def logged(s, lr, diag):
        assert diag["vlm_learning_rate"] == lr
        assert all(g["lr"] == diag[g["lr_role"] + "_learning_rate"] for g in opt.param_groups)
        records.append((s, lr, diag["flow_head_learning_rate"]))
    end = trainer.training_loop(model, [batch] * 5, normalizer, opt, args=args, global_step=0,
        log_callback=logged, checkpoint_callback=lambda s: None)
    assert end == opt.steps == model.calls == 5
    for s, vlm, head in records:
        factor = 1 if s <= 3 else .1
        assert vlm == pytest.approx(1e-6 * factor) and head == pytest.approx(1e-5 * factor)


@pytest.mark.parametrize("saved_head,requested_head,legacy,allowed", [
    (None, None, True, True), (None, 1e-5, True, False),
    (1e-5, 1e-5, False, True), (1e-5, 3e-5, False, False), (1e-5, None, False, False)])
def test_checkpoint_resume_lr_contract(setup, tmp_path, saved_head, requested_head, legacy, allowed):
    model, batch, normalizer, opt = setup
    formal.set_optimizer_learning_rates(opt, vlm_lr=1e-6, flow_head_lr=saved_head or 1e-6)
    for _ in range(3):
        step_module.hybrid_training_step(model, batch, normalizer, opt, device_type="cpu", autocast_enabled=False)
    path = tmp_path / "checkpoint.pt"
    checkpoint.save_hybrid_checkpoint(path, model, opt, global_step=3, metadata=metadata(saved_head))
    if legacy:
        payload = torch.load(path, weights_only=True)
        for group in payload["optimizer"]["param_groups"]:
            del group["lr_role"]
        torch.save(payload, path)
    before = copy.deepcopy(model.state_dict())
    if not allowed:
        with pytest.raises(ValueError, match="flow_head_learning_rate"):
            checkpoint.load_hybrid_checkpoint(path, model, opt, expected_metadata=metadata(requested_head))
        assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())
        return
    state_before = copy.deepcopy(opt.state_dict()["state"])
    info = checkpoint.load_hybrid_checkpoint(path, model, opt, expected_metadata=metadata(requested_head))
    assert info["global_step"] == 3
    for key, state in opt.state_dict()["state"].items():
        for field, value in state.items():
            assert torch.equal(value, state_before[key][field])
    assert {g["lr_role"] for g in opt.param_groups} == {"vlm", "flow_head"}
    # Next update uses completed global_step, not stale checkpoint group LR.
    formal.set_optimizer_learning_rates(opt,
        vlm_lr=formal.hybrid_learning_rate(info["global_step"], base_lr=1e-6, decay_step=3),
        flow_head_lr=formal.hybrid_learning_rate(info["global_step"], base_lr=requested_head or 1e-6, decay_step=3))
    for group in opt.param_groups:
        assert group["lr"] == pytest.approx((1e-6 if group["lr_role"] == "vlm" else requested_head or 1e-6) * .1)


def test_json_log_contains_both_rates(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer.dist, "all_reduce", lambda *a, **k: None)
    monkeypatch.setattr(trainer.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(trainer, "rank_zero_io", lambda fn: fn())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a: 0)
    diag = dict(loss=1., t_mean=.5, velocity_rms=1., encoder_grad_norm=1., flow_head_grad_norm=1.,
                vlm_learning_rate=1e-6, flow_head_learning_rate=1e-5)
    trainer.write_log(tmp_path, 1, 1e-6, diag, "cpu")
    row = json.loads((tmp_path / "train.jsonl").read_text())
    assert row["learning_rate"] == row["vlm_learning_rate"] == 1e-6
    assert row["flow_head_learning_rate"] == 1e-5


def test_startup_prints_differential_rates(capsys):
    args = trainer.parse_args(CLI + ["--flow-head-learning-rate", "1e-5"])
    tree = ast.parse((ROOT / "vla-scripts/train_hybrid_spatial.py").read_text())
    block = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                 and ast.unparse(n.test) == "rank == 0" and "per_device_batch_size" in ast.unparse(n))
    exec(compile(ast.Module(body=[block], type_ignores=[]), "startup", "exec"),
         dict(json=json, rank=0, world_size=1, args=args))
    row = json.loads(capsys.readouterr().out)
    assert row["vlm_learning_rate"] == 1e-6 and row["flow_head_learning_rate"] == 1e-5
