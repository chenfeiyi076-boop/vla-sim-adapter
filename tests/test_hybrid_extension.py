"""Opt-in plan extension preserves recipe and real AdamW state."""
import copy
import json
import sys
import ast

import pytest
import torch
from test_hybrid_step import inputs, training, ROOT
from test_hybrid_multistep import setup, step_module
from test_hybrid_learning_rates import formal, trainer, checkpoint, metadata, CLI


@pytest.fixture(autouse=True)
def modules(monkeypatch):
    monkeypatch.setitem(sys.modules, "prismatic.training.hybrid_checkpoint", checkpoint)


def test_cli():
    assert not trainer.parse_args(CLI).allow_max_steps_extension
    with pytest.raises(SystemExit):
        trainer.parse_args(CLI + ["--allow-max-steps-extension"])
    assert trainer.parse_args(CLI + ["--resume", "old.pt", "--allow-max-steps-extension"]).allow_max_steps_extension


@pytest.mark.parametrize("requested,allow,ok", [(15000,False,False),(15000,True,True),
                                              (10000,False,True),(10000,True,True),(9000,True,False)])
def test_maximum_rules(requested, allow, ok):
    old = dict(metadata(1e-5), max_steps=10000)
    new = dict(old, max_steps=requested)
    if ok:
        checkpoint.validate_resume_metadata(old, new, allow_max_steps_extension=allow)
    else:
        with pytest.raises(ValueError):
            checkpoint.validate_resume_metadata(old, new, allow_max_steps_extension=allow)


@pytest.mark.parametrize("field", list(metadata(1e-5).keys() - {"max_steps"}))
def test_every_other_identity_field_rejected(field, tmp_path):
    old = metadata(1e-5)
    new = dict(old, max_steps=15000)
    new[field] = "changed"
    formal.prepare_run_manifest(tmp_path, old)
    before = (tmp_path / "run_config.json").read_bytes()
    with pytest.raises(ValueError):
        checkpoint.validate_resume_metadata(old, new, allow_max_steps_extension=True)
    with pytest.raises(ValueError):
        formal.prepare_run_manifest(tmp_path, new, resume=True, allow_max_steps_extension=True)
    assert (tmp_path / "run_config.json").read_bytes() == before


def test_legacy_equal_and_extra_fields():
    old = metadata()
    new = dict(old, max_steps=15, flow_head_learning_rate=1e-6)
    checkpoint.validate_resume_metadata(old, new, allow_max_steps_extension=True)
    with pytest.raises(ValueError):
        checkpoint.validate_resume_metadata(dict(old, unknown=True), new, allow_max_steps_extension=True)


@pytest.mark.parametrize("old_max", [4, 5])
@pytest.mark.parametrize("factor", [.1, 1.])
def test_real_extension_continuity(setup, tmp_path, monkeypatch, old_max, factor):
    model, batch, normalizer, opt = setup
    old = dict(metadata(1e-5), max_steps=old_max, lr_decay_factor=factor)
    new = dict(old, max_steps=6)
    for _ in range(4):
        step_module.hybrid_training_step(model, batch, normalizer, opt, device_type="cpu", autocast_enabled=False)
    path = tmp_path / "step-00000004.pt"
    checkpoint.save_hybrid_checkpoint(path, model, opt, global_step=4, metadata=old)
    original = path.read_bytes()
    saved_state = copy.deepcopy(opt.state_dict())
    saved_model = copy.deepcopy(model.state_dict())
    opt.state.clear()
    with torch.no_grad():
        next(model.parameters()).add_(1)
    formal.prepare_run_manifest(tmp_path, old)
    log = tmp_path / "train.jsonl"
    log.write_text('{"global_step":4}\n')
    info = checkpoint.load_hybrid_checkpoint(path, model, opt, expected_metadata=new, allow_max_steps_extension=True)
    assert info["global_step"] == 4
    assert all(torch.equal(v, model.state_dict()[k]) for k,v in saved_model.items())
    for key, state in saved_state["state"].items():
        for field, value in state.items():
            assert torch.equal(value, opt.state_dict()["state"][key][field])
    formal.prepare_run_manifest(tmp_path, new, resume=True, allow_max_steps_extension=True)
    assert json.loads((tmp_path / "run_config.json").read_text()) == new
    monkeypatch.setitem(sys.modules, "prismatic.training.hybrid_formal", formal)
    from types import SimpleNamespace
    def step(*args, **kwargs):
        return step_module.hybrid_training_step(*args, device_type="cpu", autocast_enabled=False)
    monkeypatch.setitem(sys.modules, "prismatic.training.hybrid_multistep", SimpleNamespace(hybrid_training_step=step))
    args = trainer.parse_args(CLI + ["--max-steps", "6", "--flow-head-learning-rate", "1e-5",
        "--lr-decay-step", "3", "--lr-decay-factor", str(factor), "--log-every", "1",
        "--save-every", "5", "--per-device-batch-size", "2"])
    records, saves = [], []
    def logged(s, lr, diag):
        records.append(s)
        assert lr == pytest.approx(1e-6 * factor)
        assert diag["flow_head_learning_rate"] == pytest.approx(1e-5 * factor)
        with log.open("a") as stream:
            stream.write(json.dumps({"global_step":s}) + "\n")
    def save(s):
        saves.append(s)
        checkpoint.save_hybrid_checkpoint(tmp_path / formal.checkpoint_name(s), model, opt, global_step=s, metadata=new)
    assert trainer.training_loop(model, [batch]*2, normalizer, opt, args=args, global_step=info["global_step"],
        log_callback=logged, checkpoint_callback=save) == 6
    assert records == saves == [5,6]
    assert all(int(state["step"]) == 6 for state in opt.state.values())
    assert path.read_bytes() == original
    assert torch.load(tmp_path / "step-00000006.pt", weights_only=True)["metadata"] == new
    assert [json.loads(row)["global_step"] for row in log.read_text().splitlines()] == [4,5,6]


def test_manifest_io_follows_collective_restore_validation():
    source = (ROOT / "vla-scripts/train_hybrid_spatial.py").read_text()
    start = source.index("if args.resume is not None:", source.index("def main"))
    assert start < source.index("errors = gather_report(error)", start) < source.index("state_checks(model, optimizer, global_step", start) < source.index("rank_zero_io(lambda: prepare_run_manifest", start)
    assert "if global_step >= args.max_steps:" in source
