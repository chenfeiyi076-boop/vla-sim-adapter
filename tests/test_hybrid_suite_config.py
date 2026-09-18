"""CPU suite protocol/identity checks; no real simulator or non-Spatial weights."""

import argparse
import ast
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from test_hybrid_step import ROOT, standalone
from experiments.robot.libero.hybrid_suite_config import (
    TaskSuite, TASK_MAX_STEPS, add_task_suite_argument, get_hybrid_libero_suite_config,
    read_suite_statistics, suite_from_manifest,
)
from experiments.robot.libero import hybrid_eval_results as results
from experiments.robot.libero.compare_hybrid_eval import compare


SUITES = [("libero_spatial", TaskSuite.LIBERO_SPATIAL, 220),
          ("libero_object", TaskSuite.LIBERO_OBJECT, 280),
          ("libero_goal", TaskSuite.LIBERO_GOAL, 300),
          ("libero_10", TaskSuite.LIBERO_10, 520)]


def test_default_and_explicit_spatial():
    parser = argparse.ArgumentParser()
    add_task_suite_argument(parser)
    implicit = parser.parse_args([]).task_suite
    explicit = parser.parse_args(["--task-suite", "libero_spatial"]).task_suite
    assert implicit == explicit == "libero_spatial"
    assert get_hybrid_libero_suite_config(implicit) == get_hybrid_libero_suite_config(explicit)
    # Official evaluator re-exports the shared protocol; no replacement enum/table.
    tree = ast.parse((ROOT / "experiments/robot/libero/run_libero_eval.py").read_text())
    assert any(isinstance(n, ast.ImportFrom) and n.module == "experiments.robot.libero.libero_suite_config"
               and {a.name for a in n.names} == {"TaskSuite", "TASK_MAX_STEPS"} for n in tree.body)


@pytest.mark.parametrize("name,enum,limit", SUITES)
def test_registry_statistics_and_episode_limit(tmp_path, monkeypatch, name, enum, limit):
    config = get_hybrid_libero_suite_config(name)
    assert config.task_suite is enum and config.max_steps == TASK_MAX_STEPS[enum] == limit
    assert config.dataset_key == name + "_no_noops"
    path = tmp_path / "stats.json"
    stats = {s + "_no_noops": {"marker": i} for i, (s, _, _) in enumerate(SUITES)}
    path.write_text(json.dumps(stats))
    assert read_suite_statistics(path, name) == stats[config.dataset_key]
    del stats[config.dataset_key]
    path.write_text(json.dumps(stats))
    with pytest.raises(ValueError, match=config.dataset_key):
        read_suite_statistics(path, name)

    calls = []
    api = SimpleNamespace(TASK_MAX_STEPS=TASK_MAX_STEPS,
        get_libero_dummy_action=lambda family: [0.] * 7,
        get_image_resize_size=lambda cfg: 224,
        prepare_observation=lambda obs, size: (obs, None), process_action=lambda action, family: action)
    monkeypatch.setitem(sys.modules, "experiments.robot.libero.run_libero_eval", api)
    class Env:
        def reset(self): calls.append("reset")
        def set_init_state(self, state):
            calls.append(("initial", state))
            return {}
        def step(self, action):
            calls.append("step")
            return {}, 0, False, {}
    cfg = SimpleNamespace(task_suite_name=enum, num_steps_wait=10, model_family="openvla", num_open_loop_steps=8)
    runner = standalone("experiments/robot/libero/run_hybrid_episode.py")
    result = runner.run_single_episode(cfg, Env(), "task", lambda *a: np.zeros((10, 7)), 123)
    assert calls[:2] == ["reset", ("initial", 123)] and calls.count("step") == limit + 10
    assert result == dict(success=False, policy_calls=(limit + 7) // 8, action_steps=limit)


@pytest.mark.parametrize("name", ["libero_long", "libero_90", "unknown", ""])
def test_invalid_suite(name):
    with pytest.raises(ValueError, match="Unsupported Hybrid LIBERO task suite"):
        get_hybrid_libero_suite_config(name)
    parser = argparse.ArgumentParser()
    add_task_suite_argument(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--task-suite", name])


@pytest.mark.parametrize("name,enum,limit", SUITES)
def test_journal_merge_comparison_respect_suite_limit(tmp_path, name, enum, limit):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"identity only")
    config = get_hybrid_libero_suite_config(name)
    manifest = dict(config.identity(), max_action_steps=limit, global_step=40000, num_euler_steps=10,
        checkpoint=str(checkpoint.resolve()), checkpoint_sha256=results.file_hash(checkpoint))
    spec = dict(global_episode_id=0, task_id=0, trial_id=0, episode_seed=7, initial_state_hash="state")
    record = dict(spec, task_description="task", success=False, action_steps=limit,
                  policy_calls=(limit + 7) // 8, elapsed_sec=1.)
    path = tmp_path / "partial.json"
    journal = results.EpisodeJournal(path, manifest, [spec])
    journal.append(record)
    results.EpisodeJournal(path, manifest, [spec], resume=True).require_complete()
    assert compare([path], [path])["protocol_results_equal"]
    merged = results.merge_partials([path], checkpoint=checkpoint)
    assert merged["task_suite"] == name and merged["dataset_key"] == config.dataset_key
    assert merged["max_episode_steps"] == limit and merged["total_trials"] == 1
    with pytest.raises(ValueError, match="max steps"):
        results.validate_records([dict(record, action_steps=limit + 1, policy_calls=(limit + 8) // 8)],
                                 [spec], {0}, manifest=manifest)
    wrong = dict(manifest, task_suite="libero_object" if name != "libero_object" else "libero_spatial")
    with pytest.raises(ValueError):
        results.EpisodeJournal(path, wrong, [spec], resume=True)


def test_legacy_spatial_manifest_comparison(tmp_path):
    old = dict(task_suite_name="TaskSuite.LIBERO_SPATIAL", max_action_steps=220)
    config = suite_from_manifest(old)
    assert config.task_suite is TaskSuite.LIBERO_SPATIAL
    paths = [tmp_path / "old.json", tmp_path / "new.json"]
    spec = dict(global_episode_id=0, task_id=0, trial_id=0, episode_seed=7, initial_state_hash="state")
    for path, manifest in zip(paths, [old, dict(old, **config.identity())]):
        journal = results.EpisodeJournal(path, manifest, [spec])
        journal.append(dict(spec, task_description="task", success=True, action_steps=8, policy_calls=1, elapsed_sec=1.))
    assert compare([paths[0]], [paths[1]])["protocol_results_equal"]
    with pytest.raises(ValueError, match="Conflicting"):
        suite_from_manifest(dict(old, task_suite="libero_goal"))
