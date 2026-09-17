"""Evaluation-only identities, atomic episode journals and strict result reduction."""

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from contextlib import contextmanager

import numpy as np


class PlannedStop(Exception):
    """A completed episode was published; stop a smoke without producing final JSON."""


@contextmanager
def journal_lock(path):
    """One writer, including resume. A hard-kill stale lock requires operator inspection."""
    if path is None:
        yield
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(path.name + ".lock")
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        lock.unlink()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(value):
    value = np.asarray(value)
    if value.dtype.hasobject:
        raise ValueError("Cannot hash object arrays")
    value = np.ascontiguousarray(value.astype(value.dtype.newbyteorder("<")))
    digest = hashlib.sha256(str((value.shape, value.dtype.str)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def atomic_json(path, value, *, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    encoded = json.dumps(value, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def assigned_ids(specs, worker_id=0, num_workers=1, strategy="task_ordinal_v1"):
    if type(num_workers) is not int or num_workers < 1 or not 0 <= worker_id < num_workers:
        raise ValueError("Invalid worker assignment")
    if strategy != "task_ordinal_v1":
        raise ValueError("Incompatible worker assignment strategy")
    owners = {task_id: ordinal % num_workers
              for ordinal, task_id in enumerate(sorted({s["task_id"] for s in specs}))}
    return {s["global_episode_id"] for s in specs if owners[s["task_id"]] == worker_id}


def worker_assignment(worker_id, num_workers):
    assignment = dict(worker_id=worker_id, num_workers=num_workers)
    if num_workers > 1:
        # Reject old episode-sharded journals even if their completed rows happen to overlap.
        assignment["strategy"] = "task_ordinal_v1"
    return assignment


def validate_records(records, specs, allowed, *, complete=False):
    expected = {s["global_episode_id"]: s for s in specs}
    seen = set()
    if not isinstance(records, list):
        raise ValueError("Episode records must be a list")
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Malformed episode record")
        eid = record.get("global_episode_id")
        if type(eid) is not int or eid not in allowed or eid not in expected or eid in seen:
            raise ValueError("Duplicate, unknown or incorrectly assigned episode")
        if any(type(record.get(k)) is not int for k in ("task_id", "trial_id", "episode_seed")):
            raise ValueError("Invalid episode ID/seed types")
        if not isinstance(record.get("task_description"), str):
            raise ValueError("Invalid task description")
        if any(record.get(k) != v for k, v in expected[eid].items()):
            raise ValueError("Episode identity/state/seed mismatch")
        if type(record.get("success")) is not bool:
            raise ValueError("Invalid success")
        for key in ("policy_calls", "action_steps"):
            if type(record.get(key)) is not int or record[key] < 1:
                raise ValueError(f"Invalid {key}")
        if record["policy_calls"] != (record["action_steps"] + 7) // 8:
            raise ValueError("Invalid H10/execute8 episode counters")
        if record["action_steps"] > 220:
            raise ValueError("Episode exceeds Spatial max steps")
        elapsed = record.get("elapsed_sec")
        if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Invalid elapsed_sec")
        seen.add(eid)
    if complete and seen != allowed:
        raise ValueError("Missing episode results")
    return seen


class EpisodeJournal:
    def __init__(self, path, manifest, specs, *, resume=False, worker_id=0, num_workers=1):
        self.path = Path(path) if path is not None else None
        self.manifest = manifest
        self.specs = specs
        self.assignment = worker_assignment(worker_id, num_workers)
        self.allowed = assigned_ids(specs, worker_id, num_workers)
        self.records = []
        if self.path is not None and self.path.exists():
            if not resume:
                raise FileExistsError(self.path)
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if (payload.get("manifest") != manifest or payload.get("episodes") != specs
                    or payload.get("assignment") != self.assignment):
                raise ValueError("Partial run identity/assignment mismatch")
            self.records = payload["records"]
        self.completed = validate_records(self.records, specs, self.allowed)
        if self.path is not None and not self.path.exists():
            self.publish(overwrite=False)

    def publish(self, *, overwrite=True):
        if self.path is not None:
            atomic_json(self.path, dict(manifest=self.manifest, episodes=self.specs,
                        assignment=self.assignment, records=self.records), overwrite=overwrite)

    def append(self, record):
        records = self.records + [record]
        completed = validate_records(records, self.specs, self.allowed)
        self.records = records
        self.publish()
        self.completed = completed

    def require_complete(self):
        validate_records(self.records, self.specs, self.allowed, complete=True)


def aggregate(records):
    tasks = []
    for task_id in sorted({r["task_id"] for r in records}):
        rows = [r for r in records if r["task_id"] == task_id]
        successes = sum(r["success"] for r in rows)
        tasks.append(dict(task_id=task_id, task_description=rows[0]["task_description"],
            successes=successes, trials=len(rows), success_rate=successes / len(rows),
            total_policy_calls=sum(r["policy_calls"] for r in rows),
            total_action_steps=sum(r["action_steps"] for r in rows)))
    total = len(records)
    successes = sum(r["success"] for r in records)
    return dict(task_results=tasks, total_successes=successes, total_trials=total,
                overall_success_rate=successes / total if total else 0.)


def merge_partials(paths, *, checkpoint):
    payloads = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    if not payloads:
        raise ValueError("No worker partials")
    first = payloads[0]
    records = []
    for index, payload in enumerate(payloads):
        if (payload["manifest"] != first["manifest"] or payload["episodes"] != first["episodes"]
                or payload["assignment"] != worker_assignment(index, len(paths))):
            raise ValueError("Worker identity mismatch")
        validate_records(payload["records"], first["episodes"],
                         assigned_ids(first["episodes"], index, len(paths)), complete=True)
        records.extend(payload["records"])
    validate_records(records, first["episodes"], assigned_ids(first["episodes"]), complete=True)
    manifest = first["manifest"]
    if str(Path(checkpoint).resolve()) != manifest["checkpoint"] or file_hash(checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed during evaluation")
    return dict(checkpoint=str(checkpoint), global_step=manifest["global_step"],
                num_euler_steps=manifest["num_euler_steps"], num_open_loop_steps=8, **aggregate(records))
