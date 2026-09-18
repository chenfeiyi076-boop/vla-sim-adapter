"""Compare complete serial/worker journals; trace differences are reported, not hidden."""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.robot.libero.hybrid_eval_results import assigned_ids, validate_records
from experiments.robot.libero.hybrid_suite_config import suite_from_manifest


def read_records(paths):
    payloads = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    first = payloads[0]
    records = []
    for payload in payloads:
        if payload["manifest"] != first["manifest"] or payload["episodes"] != first["episodes"]:
            raise ValueError("Mixed runs in comparison input")
        assignment = payload["assignment"]
        validate_records(payload["records"], first["episodes"],
            assigned_ids(first["episodes"], **assignment), complete=True, manifest=first["manifest"])
        records.extend(payload["records"])
    validate_records(records, first["episodes"], assigned_ids(first["episodes"]), complete=True, manifest=first["manifest"])
    # Canonicalize additive identity fields for comparison with old Spatial traces only.
    manifest = dict(first["manifest"], **suite_from_manifest(first["manifest"]).identity())
    return manifest, {r["global_episode_id"]: r for r in records}


def compare(baseline, candidate):
    am, a = read_records(baseline)
    bm, b = read_records(candidate)
    ignored = {"rng_mode", "profiling"}
    if {k: v for k, v in am.items() if k not in ignored} != {k: v for k, v in bm.items() if k not in ignored}:
        raise ValueError("Incompatible comparison protocols")
    if set(a) != set(b):
        raise ValueError("Different episode IDs")
    reports = []
    for eid in sorted(a):
        keys = ("task_id", "trial_id", "global_episode_id", "initial_state_hash", "episode_seed",
                "task_description", "success", "policy_calls", "action_steps")
        mismatch = [k for k in keys if a[eid][k] != b[eid][k]]
        report = dict(global_episode_id=eid, differing_fields=mismatch)
        if "debug" in a[eid] and "debug" in b[eid]:
            ad, bd = a[eid]["debug"], b[eid]["debug"]
            ao, bo = ad.get("first_observation", {}), bd.get("first_observation", {})
            report["first_observation_hashes_equal"] = {
                key: ao[key] == bo[key] if key in ao and key in bo else None
                for key in ("agentview_raw_hash", "wrist_raw_hash",
                            "agentview_processed_hash", "wrist_processed_hash")
            }
            report["noise_hashes_equal"] = ad["noise_hashes"] == bd["noise_hashes"]
            report["action_hash_equal"] = ad["action_hash"] == bd["action_hash"]
            aa, ba = np.asarray(ad["actions"], dtype=np.float64), np.asarray(bd["actions"], dtype=np.float64)
            report["trace_shapes_equal"] = aa.shape == ba.shape
            if aa.shape == ba.shape:
                difference = np.abs(aa - ba)
                report.update(max_absolute_action_difference=float(difference.max()),
                              mean_absolute_action_difference=float(difference.mean()))
        reports.append(report)
    return dict(protocol_results_equal=all(not r["differing_fields"] for r in reports), episodes=reports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", nargs="+", required=True)
    parser.add_argument("--candidate", nargs="+", required=True)
    args = parser.parse_args()
    result = compare(args.baseline, args.candidate)
    print(json.dumps(result, indent=2), flush=True)
    if not result["protocol_results_equal"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
