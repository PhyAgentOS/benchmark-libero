"""benchmark.controller/run 的 ToolResult 输出契约（七字段）。

字段集合与 skill 树 gateway.yaml 里的 output_schema 一一对应：
多一个少一个都会让下游 schema 校验失败，改这里必须同时改 skill 树。

outputs.status 是**给 Agent 看的批次结论**，枚举 {succeeded, failed, partial,
stopped}；ToolResult.status 是 forge 的执行结论，只能是 {succeeded, failed,
stopped}。批次被 stop 时两者分叉：completed_episodes>0 → partial，
一集没跑完 → stopped。
"""

from __future__ import annotations

from typing import Any

from _core.result import BenchmarkResult


def stopped_summary(*, completed: int, total: int, reason: str | None) -> str:
    summary = f"stopped after {completed}/{total} episodes"
    return f"{summary}（{reason}）" if reason else summary


def build_outputs(
    *,
    run_id: str,
    batch_status: str,
    completed_episodes: int,
    total_episodes: int,
    success_rate: float,
    result_file: str | None,
    failure_summary: str | None,
) -> dict[str, Any]:
    status = batch_status
    if batch_status == "stopped":
        status = "partial" if completed_episodes > 0 else "stopped"
    return {
        "run_id": run_id,
        "status": status,
        "completed_episodes": completed_episodes,
        "total_episodes": total_episodes,
        "success_rate": success_rate,
        "result_file": result_file,
        "failure_summary": failure_summary,
    }


def outputs_from_result(
    *,
    run_id: str,
    result: BenchmarkResult,
    result_file: str | None,
    failure_summary: str | None,
) -> dict[str, Any]:
    payload = result.to_dict()
    return build_outputs(
        run_id=run_id,
        batch_status=result.status,
        completed_episodes=int(payload.get("completed_episodes") or 0),
        total_episodes=int(payload.get("total_episodes") or 0),
        success_rate=float(payload.get("success_rate") or 0.0),
        result_file=result_file,
        failure_summary=failure_summary,
    )


__all__ = ["build_outputs", "outputs_from_result", "stopped_summary"]
