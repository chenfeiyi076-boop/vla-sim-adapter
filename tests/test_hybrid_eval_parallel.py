"""CPU/fake-env tests. These do not claim CUDA, EGL or real LIBERO equivalence."""

import json
import sys
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_hybrid_step import standalone
from test_hybrid_eval import libero
from experiments.robot.libero import hybrid_eval_results as results
from experiments.robot.libero.compare_hybrid_eval import compare

evaluation = standalone("experiments/robot/libero/run_hybrid_eval.py")
runner = standalone("experiments/robot/libero/run_hybrid_episode.py")
parallel = standalone("experiments/robot/libero/run_hybrid_eval_parallel.py")


def specs():
    return [dict(global_episode_id=t * 4 + i, task_id=t, trial_id=i,
                 episode_seed=7 + t * 100000 + i, initial_state_hash=f"{t}-{i}")
            for t in range(2) for i in range(4)]


def record(spec):
    return dict(spec, success=True, policy_calls=2, action_steps=16,
                task_description=f"task-{spec['task_id']}", elapsed_sec=.1)


def test_sharding_retains_original_task_ids():
    all_specs = [dict(global_episode_id=t * 50 + i, task_id=t) for t in range(10) for i in range(50)]
    shards = [results.assigned_ids(all_specs, i, 4) for i in range(4)]
    assert [len(s) for s in shards] == [150, 150, 100, 100]
    assert set.union(*shards) == set(range(500))
    assert sum(map(len, shards)) == len(set.union(*shards))
    for task in range(10):
        episodes = set(range(task * 50, (task + 1) * 50))
        assert [i for i, shard in enumerate(shards) if episodes & shard] == [task % 4]
        assert episodes <= shards[task % 4]
    subset = [s for s in all_specs if 200 <= s["global_episode_id"] < 250]
    assert results.assigned_ids(subset) == set(range(200, 250))
    subset = [s for s in reversed(all_specs) if s["task_id"] in (0, 4)]
    assert [results.assigned_ids(subset, i, 4) for i in range(4)] == [
        set(range(50)), set(range(200, 250)), set(), set()]


@pytest.mark.parametrize("field", ["checkpoint", "seed", "normalization", "inference", "suite", "trials"])
def test_resume_rejects_identity_change(tmp_path, field):
    path = tmp_path / "partial.json"
    manifest = {field: "original"}
    journal = results.EpisodeJournal(path, manifest, specs())
    journal.append(record(specs()[0]))
    with pytest.raises(ValueError, match="identity"):
        results.EpisodeJournal(path, {field: "changed"}, specs(), resume=True)


def test_records_reject_duplicates_missing_unknown_and_corruption():
    s = specs()
    good = [record(x) for x in s]
    results.validate_records(good, s, results.assigned_ids(s), complete=True)
    for bad in (good + good[:1], good[:-1], [dict(good[0], global_episode_id=999)],
                [dict(good[0], initial_state_hash="changed")], [dict(good[0], action_steps=0)]):
        with pytest.raises(ValueError):
            results.validate_records(bad, s, results.assigned_ids(s), complete=True)


def test_atomic_failure_preserves_partial(tmp_path, monkeypatch):
    path = tmp_path / "partial.json"
    results.atomic_json(path, {"old": 1})
    def fail(*args):
        raise OSError("disk failure")
    monkeypatch.setattr(results.os, "replace", fail)
    with pytest.raises(OSError):
        results.atomic_json(path, {"new": 2}, overwrite=True)
    assert json.loads(path.read_text()) == {"old": 1}
    assert list(tmp_path.iterdir()) == [path]


def test_single_writer_lock(tmp_path):
    path = tmp_path / "partial.json"
    with results.journal_lock(path):
        with pytest.raises(FileExistsError):
            with results.journal_lock(path):
                pytest.fail("Second writer entered")
    assert not list(tmp_path.iterdir())


def test_merge_complete_workers_and_reject_missing(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"fake checkpoint identity")
    manifest = dict(checkpoint=str(checkpoint.resolve()), checkpoint_sha256=results.file_hash(checkpoint),
                    global_step=40000, num_euler_steps=10)
    paths = []
    for worker in range(4):
        path = tmp_path / f"worker-{worker}.json"
        paths.append(path)
        journal = results.EpisodeJournal(path, manifest, specs(), worker_id=worker, num_workers=4)
        for spec in specs():
            if spec["global_episode_id"] in journal.allowed:
                journal.append(record(spec))
    merged = results.merge_partials(paths, checkpoint=checkpoint)
    assert merged["total_trials"] == 8 and merged["overall_success_rate"] == 1
    payload = json.loads(paths[0].read_text())
    payload["records"].pop()
    paths[0].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Missing"):
        results.merge_partials(paths, checkpoint=checkpoint)


def test_serial_resume_and_worker_partition(libero, tmp_path):
    # Exactly 2 tasks x 4 trials; fake environment, no GPU.
    libero.api.load_initial_states = lambda cfg, suite, index: ([(index, i) for i in range(4)], None)
    path = tmp_path / "serial.json"
    kwargs = dict(trials_per_task=4, manifest={"schema_version": 1})
    with pytest.raises(results.PlannedStop):
        evaluation.evaluate_tasks(libero.cfg, libero.suite, None, partial=path, stop_after=3, **kwargs)
    assert len(libero.calls) == 3 and all(e.closed for e in libero.environments)
    resumed = evaluation.evaluate_tasks(libero.cfg, libero.suite, None, partial=path, resume=True, **kwargs)
    assert len(libero.calls) == 8 and len(set(libero.calls)) == 8
    assert resumed["total_trials"] == 8
    paths = []
    for worker in range(4):
        part = tmp_path / f"worker{worker}.json"
        paths.append(part)
        output = evaluation.evaluate_tasks(libero.cfg, libero.suite, None,
            partial=part, worker_id=worker, num_workers=4, **kwargs)
        assert output["total_trials"] == (4 if worker < 2 else 0)
    # Comparator validates complete union and identical per-episode protocol outcomes.
    comparison = compare([path], paths)
    assert comparison["protocol_results_equal"]
    single = tmp_path / "worker-single.json"
    assert evaluation.evaluate_tasks(libero.cfg, libero.suite, None, partial=single, **kwargs) == resumed
    assert compare([path], [single])["protocol_results_equal"]


@pytest.mark.parametrize("mismatch", ["old_strategy", "worker", "workers"])
def test_resume_and_merge_reject_incompatible_assignment(tmp_path, mismatch):
    paths = [tmp_path / f"worker-{i}.json" for i in range(4)]
    for worker, path in enumerate(paths):
        journal = results.EpisodeJournal(path, {}, specs(), worker_id=worker, num_workers=4)
        for spec in specs():
            if spec["global_episode_id"] in journal.allowed:
                journal.append(record(spec))
        resumed = results.EpisodeJournal(path, {}, specs(), resume=True, worker_id=worker, num_workers=4)
        assert resumed.completed == journal.allowed
    payload = json.loads(paths[0].read_text())
    if mismatch == "old_strategy":
        del payload["assignment"]["strategy"]
    elif mismatch == "worker":
        payload["assignment"]["worker_id"] = 1
    else:
        payload["assignment"]["num_workers"] = 1
    paths[0].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="assignment"):
        results.EpisodeJournal(paths[0], {}, specs(), resume=True, worker_id=0, num_workers=4)
    with pytest.raises(ValueError, match="identity"):
        results.merge_partials(paths, checkpoint=tmp_path / "unused.pt")


def test_task_sharding_preserves_serial_env_history(libero):
    libero.suite.n_tasks = 5
    libero.api.load_initial_states = lambda cfg, suite, index: ([(index, i) for i in range(4)], None)
    histories = []
    def episode(cfg, env, description, policy, initial_state):
        if not hasattr(env, "trials"):
            env.trials = []
            histories.append((description, env.trials))
        env.trials.append(initial_state[1])
        return dict(success=len(env.trials) % 2 == 0, policy_calls=2, action_steps=16)
    libero.episode_api.run_single_episode = episode
    kwargs = dict(trials_per_task=4, task_ids=[4, 0])
    serial = evaluation.evaluate_tasks(libero.cfg, libero.suite, None, **kwargs)
    serial_history = list(histories)
    histories.clear()
    libero.environments.clear()
    outputs = [evaluation.evaluate_tasks(libero.cfg, libero.suite, None,
        worker_id=i, num_workers=4, **kwargs) for i in range(4)]
    assert histories == serial_history == [("task-0", [0, 1, 2, 3]), ("task-4", [0, 1, 2, 3])]
    assert len(libero.environments) == 2 and all(env.closed for env in libero.environments)
    assert [output["total_trials"] for output in outputs] == [4, 4, 0, 0]
    assert [task for output in outputs for task in output["task_results"]] == serial["task_results"]


@pytest.mark.parametrize("debug", [False, True])
def test_lazy_vs_old_eager_rollout_and_episode_noise(monkeypatch, debug):
    prepared = []
    hashed = []
    original_hash = results.array_hash
    def tracked_hash(value):
        hashed.append(np.asarray(value).copy())
        return original_hash(value)
    monkeypatch.setattr(results, "array_hash", tracked_hash)
    def observation(value):
        return {"agentview_image": np.asarray([value], dtype=float),
                "robot0_eye_in_hand_image": np.asarray([value + 100], dtype=float)}
    def prepare(obs, size):
        prepared.append(obs.copy())
        return {"full_image": obs["agentview_image"] + 10,
                "wrist_image": obs["robot0_eye_in_hand_image"] + 20}, None
    api = SimpleNamespace(TASK_MAX_STEPS={"libero_spatial": 18},
        get_libero_dummy_action=lambda model: [0.] * 7, get_image_resize_size=lambda cfg: 224,
        prepare_observation=prepare,
        process_action=lambda action, model: action.copy())
    monkeypatch.setitem(sys.modules, "experiments.robot.libero.run_libero_eval", api)
    class Env:
        def reset(self):
            self.steps = []
        def set_init_state(self, state):
            return observation(state)
        def step(self, action):
            self.steps.append(action)
            return observation(len(self.steps)), 0, False, {}
    class Policy:
        profile = None
        def __init__(self, reference):
            self.reference_rng = reference
            self.generator = None if reference else torch.Generator().manual_seed(7)
            self.noises = []
            self.debug_trace = {"actions": []} if debug else None
        def __call__(self, obs, description):
            noise = torch.randn(1, 10, 7, generator=self.generator)
            self.noises.append(noise.clone())
            return noise[0].numpy() + obs["full_image"][0]
    cfg = SimpleNamespace(task_suite_name="libero_spatial", model_family="openvla",
                          num_steps_wait=2, num_open_loop_steps=8)
    outcomes = []
    for reference in (True, False):
        torch.manual_seed(7)
        p, env = Policy(reference), Env()
        prepared.clear()
        hashed.clear()
        outcome = runner.run_single_episode(cfg, env, "task", p, 0)
        assert len(prepared) == (18 if reference else 3)
        assert not torch.equal(p.noises[0], p.noises[1])
        assert len(hashed) == (4 if debug else 0)
        if debug:
            # First policy call follows two settling steps, not reset or later policy calls.
            assert p.debug_trace["first_observation"] == {
                key: original_hash(np.asarray([value], dtype=float))
                for key, value in (("agentview_raw_hash", 2), ("wrist_raw_hash", 102),
                                   ("agentview_processed_hash", 12), ("wrist_processed_hash", 122))}
        else:
            assert p.debug_trace is None
        outcomes.append((outcome, env.steps, p.noises))
    assert outcomes[0][:2] == outcomes[1][:2]
    assert all(torch.equal(a, b) for a, b in zip(outcomes[0][2], outcomes[1][2]))


@pytest.mark.parametrize("physical_gpu", range(4))
def test_worker_environment_isolation(monkeypatch, physical_gpu):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5,6,7")
    monkeypatch.setenv("WORLD_SIZE", "4")
    env = parallel.worker_environment(2, physical_gpu)
    assert env["CUDA_VISIBLE_DEVICES"] == str(physical_gpu)
    assert env["MUJOCO_EGL_DEVICE_ID"] == "0"
    assert env["MUJOCO_GL"] == env["PYOPENGL_PLATFORM"] == "egl"
    assert env["OMP_NUM_THREADS"] == env["MKL_NUM_THREADS"] == "2"
    assert "WORLD_SIZE" not in env


@pytest.mark.parametrize("physical_gpu", range(4))
def test_worker_uses_local_cuda_zero(tmp_path, monkeypatch, capsys, physical_gpu):
    calls = {}
    def capture(name):
        return lambda *args: calls.__setitem__(name, args)
    monkeypatch.setitem(sys.modules, "tensorflow", SimpleNamespace(config=SimpleNamespace(
        set_visible_devices=capture("tf_visible"), threading=SimpleNamespace(
            set_intra_op_parallelism_threads=capture("tf_intra"),
            set_inter_op_parallelism_threads=capture("tf_inter")))))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        set_num_threads=capture("torch_threads"), cuda=SimpleNamespace(set_device=capture("cuda"))))
    def serial_main(argv, **kwargs):
        calls["device"] = argv[argv.index("--device") + 1]
    monkeypatch.setitem(sys.modules, "experiments.robot.libero.run_hybrid_eval", SimpleNamespace(main=serial_main))
    monkeypatch.setattr(parallel.signal, "signal", lambda *args: None)
    for key, value in parallel.worker_environment(2, physical_gpu).items():
        monkeypatch.setenv(key, value)
    parallel.main(["--worker-id", str(physical_gpu), "--devices", "0", "1", "2", "3",
                   "--threads", "2", "--checkpoint", str(tmp_path / "model.pt"),
                   "--run-dir", str(tmp_path / "run"), "--output", str(tmp_path / "out.json")])
    assert calls["cuda"] == (0,)
    assert calls["device"] == "cuda:0"
    assert calls["tf_visible"] == ([], "GPU")
    diagnostic = capsys.readouterr().err
    for field in (f"physical_gpu={physical_gpu}", f"CUDA_VISIBLE_DEVICES={physical_gpu}",
                  "torch_device=cuda:0", "EGL=0", "TF_GPU=[]"):
        assert field in diagnostic


def test_worker_crash_terminates_peer():
    class Process:
        def __init__(self, code): self.returncode = code
        def poll(self): return self.returncode
        def terminate(self): self.returncode = -15
        def wait(self, timeout=None): return self.returncode
    failed, peer = Process(1), Process(None)
    with pytest.raises(RuntimeError, match="failed"):
        parallel.wait_workers([failed, peer])
    assert peer.returncode == -15


@pytest.mark.parametrize("workers", [1, 4])
def test_parent_launches_fresh_cpu_processes_and_merges(tmp_path, monkeypatch, workers):
    """Real OS processes, fake episode records. Does not exercise CUDA/TF/LIBERO."""
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    manifest = dict(checkpoint=str(checkpoint.resolve()), checkpoint_sha256=results.file_hash(checkpoint),
                    global_step=40000, num_euler_steps=10)
    specification = tmp_path / "spec.json"
    specification.write_text(json.dumps(dict(manifest=manifest, specs=specs())))
    real_popen = subprocess.Popen
    spawned = []
    def fake_worker_command(command, **kwargs):
        worker = int(command[command.index("--worker-id") + 1])
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == str(worker)
        assert kwargs["env"]["MUJOCO_EGL_DEVICE_ID"] == "0"
        run_dir = command[command.index("--run-dir") + 1]
        program = (
            "import json,sys; from pathlib import Path; "
            "from experiments.robot.libero.hybrid_eval_results import EpisodeJournal; "
            "p=json.loads(Path(sys.argv[1]).read_text()); "
            "j=EpisodeJournal(Path(sys.argv[2])/('worker-'+sys.argv[3]+'.partial.json'),"
            "p['manifest'],p['specs'],worker_id=int(sys.argv[3]),num_workers=int(sys.argv[4])); "
            "[j.append(dict(s,success=True,policy_calls=2,action_steps=16,elapsed_sec=.1,"
            "task_description='task-'+str(s['task_id']))) for s in p['specs'] "
            "if s['global_episode_id'] in j.allowed]"
        )
        process = real_popen([sys.executable, "-c", program, str(specification), run_dir, str(worker), str(workers)], **kwargs)
        spawned.append(process)
        return process
    monkeypatch.setattr(parallel.subprocess, "Popen", fake_worker_command)
    output = tmp_path / "final.json"
    parallel.main(["--checkpoint", str(checkpoint), "--run-dir", str(tmp_path / "run"),
                   "--output", str(output), "--num-workers", str(workers),
                   "--egl-devices", *(["0"] * workers)])
    assert len({p.pid for p in spawned}) == workers
    assert all(p.returncode == 0 for p in spawned)
    assert json.loads(output.read_text())["total_trials"] == 8
    assert not (tmp_path / "run" / "launcher.lock").exists()


@pytest.mark.parametrize("observation_mode", ["equal", "different", "missing", "old"])
def test_debug_comparison_reports_numeric_action_difference(tmp_path, observation_mode):
    paths = [tmp_path / "a.json", tmp_path / "b.json"]
    for path in paths:
        journal = results.EpisodeJournal(path, {"schema_version": 1}, specs())
        for spec in specs():
            r = record(spec)
            trace = np.zeros((16, 7))
            if path == paths[1]:
                trace[0, 0] = 1e-7
            r["debug"] = dict(actions=trace.tolist(), action_hash=results.array_hash(trace), noise_hashes=["same"])
            if observation_mode != "old" and not (observation_mode == "missing" and path == paths[1]):
                r["debug"]["first_observation"] = dict(
                    agentview_raw_hash="same", wrist_raw_hash="same",
                    agentview_processed_hash="different" if observation_mode == "different" and path == paths[1] else "same",
                    wrist_processed_hash="same")
            journal.append(r)
    report = compare([paths[0]], [paths[1]])
    assert report["protocol_results_equal"]
    assert report["episodes"][0]["action_hash_equal"] is False
    assert report["episodes"][0]["max_absolute_action_difference"] == 1e-7
    for episode in report["episodes"]:
        matches = episode["first_observation_hashes_equal"]
        assert len(matches) == 4
        if observation_mode in ("missing", "old"):
            assert all(value is None for value in matches.values())
        else:
            assert matches == dict(agentview_raw_hash=True, wrist_raw_hash=True,
                agentview_processed_hash=observation_mode == "equal", wrist_processed_hash=True)
