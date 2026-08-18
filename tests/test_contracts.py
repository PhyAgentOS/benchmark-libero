from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from adapters.libero_adapter import quat_to_axisangle
from config import derive_episode_seed, load_config, resolve_and_validate_plan
from result import BenchmarkResult, EpisodeRecord


ROOT = Path(__file__).parents[1]


class _Adapter:
    num_tasks = 2

    @staticmethod
    def num_init_states(task_id: int) -> int:
        return 3 if task_id == 0 else 2


def test_dataflow_preserves_dora_input_output_contract() -> None:
    dataflow = yaml.safe_load((ROOT / "dataflow.example.yaml").read_text(encoding="utf-8"))
    benchmark = next(node for node in dataflow["nodes"] if node["id"] == "libero_benchmark")

    assert benchmark["inputs"] == {"action": "policy/action"}
    assert set(benchmark["outputs"]) == {
        "proprio_state",
        "image/main",
        "image/second",
        "policy_command",
        "benchmark_status",
        "benchmark_result",
    }


def test_config_plan_validation_and_seed_are_deterministic(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "suite: libero_spatial\n"
        "task_ids: [0, 1]\n"
        "init_state_ids: [0, 1]\n"
        "num_runs: 2\n",
        encoding="utf-8",
    )
    config = load_config(config_path)
    resolve_and_validate_plan(config, _Adapter())

    assert len(config.episode_plan()) == 8
    assert derive_episode_seed(0, 1, 1, 1) == derive_episode_seed(0, 1, 1, 1)
    assert derive_episode_seed(0, 1, 1, 1) != derive_episode_seed(0, 1, 1, 2)


def test_result_schema_retains_planned_denominator() -> None:
    result = BenchmarkResult(
        suite="libero_spatial",
        control_mode="relative",
        max_steps=300,
        total_episodes=2,
    )
    result.episodes.append(
        EpisodeRecord(
            episode_id="ep-1",
            suite="libero_spatial",
            task_id=0,
            init_state_id=0,
            run_index=0,
            task_description="test",
            success=True,
            termination="success",
            num_steps=10,
            return_value=1.0,
        )
    )

    payload = result.to_dict()
    assert payload["schema_version"] == "benchmark_execution_result_v1"
    assert payload["success_rate"] == 0.5
    assert json.loads(json.dumps(payload))["episodes"][0]["episode_id"] == "ep-1"


def test_quaternion_conversion_identity() -> None:
    np.testing.assert_array_equal(
        quat_to_axisangle(np.array([0.0, 0.0, 0.0, 1.0])),
        np.zeros(3, dtype=np.float32),
    )
