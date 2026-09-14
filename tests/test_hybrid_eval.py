"""Trained Hybrid evaluation orchestration with fake LIBERO surfaces."""

import ast
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from test_hybrid_step import ROOT, standalone


evaluation = standalone("experiments/robot/libero/run_hybrid_eval.py")
formal = standalone("prismatic/training/hybrid_formal.py")
checkpoint = standalone("prismatic/training/hybrid_checkpoint.py")


@pytest.fixture
def libero(monkeypatch):
    calls, environments = [], []
    suite = SimpleNamespace(n_tasks=2, get_task=lambda index: index)
    def env(task, family, resolution):
        instance = SimpleNamespace(closed=False)
        instance.close = lambda: setattr(instance, "closed", True)
        environments.append(instance)
        return instance, f"task-{task}"
    def episode(cfg, env, description, policy, initial_state):
        calls.append((description, initial_state))
        assert cfg.num_open_loop_steps == 8
        return dict(success=initial_state[1] % 2 == 0, policy_calls=2, action_steps=16)
    api = SimpleNamespace(get_libero_env=env,
        load_initial_states=lambda cfg, suite, index: ([(index, trial) for trial in range(3)], None),
        GenerateConfig=lambda **kwargs: SimpleNamespace(model_family="openvla", env_img_res=256, center_crop=True, **kwargs),
        TaskSuite=SimpleNamespace(LIBERO_SPATIAL="libero_spatial"),
        benchmark=SimpleNamespace(get_benchmark_dict=lambda: {"libero_spatial": lambda: suite}),
        set_seed_everywhere=lambda seed: None)
    monkeypatch.setitem(sys.modules, "experiments.robot.libero.run_libero_eval", api)
    episode_api = SimpleNamespace(run_single_episode=episode)
    monkeypatch.setitem(sys.modules, "experiments.robot.libero.run_hybrid_episode", episode_api)
    return SimpleNamespace(api=api, episode_api=episode_api, suite=suite, calls=calls, environments=environments,
                           cfg=SimpleNamespace(model_family="openvla", env_img_res=256, num_open_loop_steps=8))


@pytest.mark.parametrize("task_id", [None, 1])
def test_task_selection_trials_and_aggregation(libero, task_id):
    result = evaluation.evaluate_tasks(libero.cfg, libero.suite, None, trials_per_task=3, task_id=task_id)
    selected = [0, 1] if task_id is None else [1]
    assert libero.calls == [(f"task-{index}", (index, trial)) for index in selected for trial in range(3)]
    assert result["total_trials"] == len(selected) * 3
    assert result["total_successes"] == len(selected) * 2
    assert result["overall_success_rate"] == 2 / 3
    assert all(r["total_policy_calls"] == 6 and r["total_action_steps"] == 48 for r in result["task_results"])
    assert all(env.closed for env in libero.environments)


def test_insufficient_initial_states_rejected_before_rollout(libero):
    with pytest.raises(ValueError, match="official initial states"):
        evaluation.evaluate_tasks(libero.cfg, libero.suite, None, trials_per_task=4)
    assert not libero.calls and not libero.environments


def test_runtime_errors_propagate_and_environment_closes(libero):
    def broken(*args):
        raise RuntimeError("model failure")
    libero.episode_api.run_single_episode = broken
    with pytest.raises(RuntimeError, match="model failure"):
        evaluation.evaluate_tasks(libero.cfg, libero.suite, None, trials_per_task=1)
    assert libero.environments[0].closed


@pytest.mark.parametrize("mismatch", [None, "normalization", "step"])
def test_trained_loading_validation_before_rollout(tmp_path, monkeypatch, mismatch):
    stats = {kind: {"mean": [0.] * dim, "std": [1.] * dim} for kind, dim in (("action", 7), ("proprio", 8))}
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(json.dumps({"libero_spatial_no_noops": stats}))
    normalized = formal.normalization_metadata(stats)
    if mismatch == "normalization":
        normalized["action"]["mean"][0] = .1
    policy = SimpleNamespace(encoder=torch.nn.Linear(2, 2), flow_head=torch.nn.Linear(2, 2))
    payload = dict(format_version=1, global_step=100, encoder=policy.encoder.state_dict(), flow_head=policy.flow_head.state_dict(),
        metadata=dict(experiment="hybrid_spatial_formal_v1", dataset_key="libero_spatial_no_noops",
                      action_horizon=10, action_dim=7, proprio_dim=8, normalization_statistics=normalized))
    path = tmp_path / "trained.pt"
    torch.save(payload, path)
    monkeypatch.setitem(sys.modules, "experiments.robot.libero.run_hybrid_episode", SimpleNamespace(load_policy=lambda *args: policy))
    monkeypatch.setitem(sys.modules, "prismatic.training.hybrid_checkpoint", checkpoint)
    monkeypatch.setitem(sys.modules, "prismatic.training.hybrid_formal", formal)
    args = SimpleNamespace(statistics=stats_path, checkpoint=path, expected_step=99 if mismatch == "step" else 100,
                           vlm_path="native", hf_config="config", device="cpu", num_steps=10)
    if mismatch:
        with pytest.raises(ValueError):
            evaluation.load_trained_policy(args)
    else:
        loaded, info = evaluation.load_trained_policy(args)
        assert loaded is policy and info["global_step"] == 100
        assert not loaded.encoder.training and not loaded.flow_head.training


def test_result_schema_zero_success_and_atomic_output(libero, monkeypatch, tmp_path, capsys):
    libero.episode_api.run_single_episode = lambda *args: dict(success=False, policy_calls=1, action_steps=8)
    monkeypatch.setattr(evaluation, "load_trained_policy", lambda *a, **kw: (None, {"global_step": 100}))
    monkeypatch.setitem(sys.modules, "prismatic.training.hybrid_formal", formal)
    output = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", ["eval", "--checkpoint", "trained.pt", "--vlm-path", "native",
        "--hf-config", "config", "--statistics", "stats.json", "--trials-per-task", "2", "--output", str(output)])
    evaluation.main()
    result = json.loads(output.read_text())
    assert result == json.loads(capsys.readouterr().out)
    assert result["total_trials"] == 4 and result["total_successes"] == result["overall_success_rate"] == 0
    assert result["num_euler_steps"] == 10 and result["num_open_loop_steps"] == 8 and result["global_step"] == 100
    assert list(tmp_path.iterdir()) == [output]


def test_atomic_output_failure_preserves_existing_file(tmp_path, monkeypatch):
    path = tmp_path / "result.json"
    path.write_text("original")
    def fail(*args):
        raise OSError("replace failed")
    monkeypatch.setattr(formal.os, "replace", fail)
    with pytest.raises(OSError):
        formal.atomic_write_json(path, {"new": True}, overwrite=True)
    assert path.read_text() == "original" and list(tmp_path.iterdir()) == [path]


def test_existing_execution_is_reused_without_new_conversion():
    tree = ast.parse((ROOT / "experiments/robot/libero/run_hybrid_eval.py").read_text())
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not (names | attrs) & {"ActionTokenizer", "process_action", "denormalize_action", "invert_gripper_action", "normalize_gripper_action"}
    assert "run_single_episode" in names
    episode = ast.parse((ROOT / "experiments/robot/libero/run_hybrid_episode.py").read_text())
    function = next(n for n in episode.body if isinstance(n, ast.FunctionDef) and n.name == "run_single_episode")
    assert sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "process_action" for n in ast.walk(function)) == 1
    assert any(isinstance(n, ast.Attribute) and n.attr == "num_open_loop_steps" for n in ast.walk(function))
