"""Controlled Hybrid evaluation identities; no model/TF/simulator imports."""

from dataclasses import dataclass
from experiments.robot.libero.libero_suite_config import TaskSuite, TASK_MAX_STEPS


DEFAULT_TASK_SUITE = TaskSuite.LIBERO_SPATIAL.value
HYBRID_LIBERO_SUITES = {
    TaskSuite.LIBERO_SPATIAL: "libero_spatial_no_noops",
    TaskSuite.LIBERO_OBJECT: "libero_object_no_noops",
    TaskSuite.LIBERO_GOAL: "libero_goal_no_noops",
    TaskSuite.LIBERO_10: "libero_10_no_noops",
}


@dataclass(frozen=True)
class HybridSuiteConfig:
    task_suite: TaskSuite
    dataset_key: str
    max_steps: int

    def identity(self):
        return dict(task_suite=self.task_suite.value, dataset_key=self.dataset_key)


def get_hybrid_libero_suite_config(name=DEFAULT_TASK_SUITE):
    try:
        suite = TaskSuite(name)
        return HybridSuiteConfig(suite, HYBRID_LIBERO_SUITES[suite], TASK_MAX_STEPS[suite])
    except (ValueError, KeyError, TypeError):
        raise ValueError(f"Unsupported Hybrid LIBERO task suite: {name}") from None


def add_task_suite_argument(parser):
    parser.add_argument("--task-suite", default=DEFAULT_TASK_SUITE,
                        choices=[suite.value for suite in HYBRID_LIBERO_SUITES])


def suite_from_manifest(manifest):
    # Old Spatial journals recorded str(TaskSuite) rather than its value.
    name = manifest.get("task_suite", manifest.get("task_suite_name", DEFAULT_TASK_SUITE))
    if isinstance(name, str) and name.startswith("TaskSuite."):
        name = name.split(".", 1)[1].lower()
    config = get_hybrid_libero_suite_config(name)
    if "task_suite_name" in manifest:
        legacy_name = manifest["task_suite_name"]
        if isinstance(legacy_name, str) and legacy_name.startswith("TaskSuite."):
            legacy_name = legacy_name.split(".", 1)[1].lower()
        if get_hybrid_libero_suite_config(legacy_name).task_suite != config.task_suite:
            raise ValueError("Conflicting manifest task suite identities")
    if "dataset_key" in manifest and manifest["dataset_key"] != config.dataset_key:
        raise ValueError("Manifest suite/dataset identity mismatch")
    if "max_action_steps" in manifest and manifest["max_action_steps"] != config.max_steps:
        raise ValueError("Manifest suite/max_action_steps mismatch")
    return config


def read_suite_statistics(path, task_suite=DEFAULT_TASK_SUITE):
    import json
    from pathlib import Path
    config = get_hybrid_libero_suite_config(task_suite)
    statistics = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.dataset_key not in statistics:
        raise ValueError(f"RLDS statistics must contain {config.dataset_key}; no fallback is allowed")
    return statistics[config.dataset_key]
