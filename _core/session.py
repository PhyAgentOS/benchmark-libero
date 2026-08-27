"""服务模式:一次 bench.run invocation 的封装,以及与 bench_endpoint 的线格式。

本模块**只描述一次 invocation 是什么**,不含事件循环——循环在 node.py。

⚠️ **不 import forge_tool。** Forge 协议整层由独立的 bench_endpoint 节点负责,
本节点只发最朴素的 JSON,形态与已有的 `benchmark_status` / `benchmark_result`
通道一致(`pa.array([json.dumps(...)])`)。这样 benchmark 节点的依赖面不变,
也不必跟着 forge_tool 的 Python 版本要求走。

线协议常量在两个包里各有一份(这里 + bench_endpoint/wire.py)。**是刻意重复的**:
两个包跑在不同进程、可能不同解释器,不共享依赖。改动时两边必须一起改,
`WIRE_VERSION` 就是用来在忘记时把错误暴露出来的。
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from typing import Any

import pyarrow as pa

from .config import BaseBenchmarkConfig, resolve_and_validate_plan
from .runner import BenchmarkRunner

WIRE_VERSION = 1

# endpoint → benchmark(通道 bench_cmd)
CMD_RUN = "run"
CMD_CANCEL = "cancel"
CMD_DESCRIBE = "describe"

# benchmark → endpoint(通道 bench_report)
REPORT_CAPABILITIES = "capabilities"
REPORT_ACCEPTED = "accepted"
REPORT_REJECTED = "rejected"
REPORT_PROGRESS = "progress"
REPORT_HEARTBEAT = "heartbeat"
REPORT_RESULT = "result"

# 回传的 episodes 明细上限。Tool Wire 有消息大小限制,整个 suite 的逐集明细会撞上;
# 超了就只给 result_path,明细留在盘上。
MAX_EPISODES_IN_PAYLOAD = 64

# 心跳周期(秒)。endpoint 侧的判活阈值必须**大于本节点一次最长的阻塞调用**
# (episode 启动要重建仿真场景,事件循环会整个停住),不是这个周期的几倍。
HEARTBEAT_PERIOD_S = 1.0


class WireError(ValueError):
    """线上消息不合法。调用方应丢弃并记一行日志,不要让它冲出事件循环。"""


def encode(payload: dict[str, Any]) -> pa.Array:
    body = dict(payload)
    body.setdefault("v", WIRE_VERSION)
    return pa.array([json.dumps(body, ensure_ascii=False)])


def decode(value: Any) -> dict[str, Any]:
    """把 dora 事件的 value 解回 dict。容忍 pyarrow Array / list / 裸字符串。"""
    if isinstance(value, (pa.Array, pa.ChunkedArray)):
        items = value.to_pylist()
    elif isinstance(value, list):
        items = value
    else:
        items = [value]
    if len(items) != 1:
        raise WireError(f"每条消息只应有 1 个元素,收到 {len(items)}")
    text = items[0]
    if isinstance(text, (bytes, bytearray)):
        text = text.decode("utf-8")
    if not isinstance(text, str):
        raise WireError(f"载体必须是字符串,收到 {type(text).__name__}")
    try:
        body = json.loads(text)
    except json.JSONDecodeError as exc:
        raise WireError(f"不是合法 JSON: {exc}") from exc
    if not isinstance(body, dict):
        raise WireError("载体必须是 JSON 对象")
    if body.get("v") != WIRE_VERSION:
        raise WireError(f"线协议版本不匹配:本端 {WIRE_VERSION},收到 {body.get('v')!r}")
    if not isinstance(body.get("type"), str):
        raise WireError("消息缺少 type 字段")
    return body


# ---------------------------------------------------------------- capabilities


def _camera_ids(cfg: BaseBenchmarkConfig, adapter: Any) -> list[str]:
    """观测图像的 dora 通道名。

    不同 benchmark 的来源不一样:有的在 config 里配 `camera_map`(通道→相机名),
    LIBERO 则是 adapter 的 `to_images()` 键写死的。两条都试,都没有就给空表——
    **不要为此改 adapter 协议**,describe 少一个字段不影响调用方构造 bench.run。
    """
    mapping = getattr(cfg, "camera_map", None)
    if isinstance(mapping, dict) and mapping:
        return sorted(mapping)
    hook = getattr(adapter, "camera_ids", None)
    if callable(hook):
        try:
            return sorted(str(x) for x in hook())
        except Exception:  # noqa: BLE001 拿不到相机名不该挡住 describe
            return []
    return []


def capabilities_payload(
    cfg: BaseBenchmarkConfig,
    adapter: Any,
    *,
    limits: dict[str, int] | None = None,
) -> dict[str, Any]:
    """bench.describe 的数据源。

    task 清单只有 adapter 知道,所以由本节点**主动推**给 endpoint 缓存,而不是让
    endpoint 按需来拉:endpoint 的 Tool 分发是同步跑到底的,让它去 await 一条 dora
    消息会死锁,而且完全静默。

    benchmark 特有的口径(LIBERO 的 layout_mode/control_mode 之类)一律走
    `adapter.reproducibility()` 原样透传,_core 不认识它们。
    """
    names = getattr(adapter, "tasks", None)
    tasks: list[dict[str, Any]] = []
    for task_id in range(adapter.num_tasks):
        entry: dict[str, Any] = {
            "task_id": task_id,
            "num_init_states": int(adapter.num_init_states(task_id)),
        }
        if names is not None and task_id < len(names):
            entry["name"] = str(names[task_id])
        try:
            entry["language"] = str(adapter.task_language(task_id))
        except Exception:  # noqa: BLE001 拿不到指令不该挡住 describe
            entry["language"] = ""
        tasks.append(entry)

    repro = getattr(adapter, "reproducibility", None)
    try:
        reproducibility = dict(repro()) if callable(repro) else {}
    except Exception:  # noqa: BLE001
        reproducibility = {}

    return {
        "type": REPORT_CAPABILITIES,
        "benchmark": cfg.benchmark,
        "suite": cfg.suite,
        "policy_id": cfg.policy_id,
        "num_tasks": int(adapter.num_tasks),
        "tasks": tasks,
        "joint_names": list(adapter.state_joint_names()),
        "camera_ids": _camera_ids(cfg, adapter),
        "defaults": {
            "num_runs": cfg.num_runs,
            "seed": cfg.seed,
            "max_steps": cfg.max_steps,
        },
        "limits": dict(limits or {}),
        "reproducibility": reproducibility,
    }


# ------------------------------------------------------------------- 一次执行


class BenchSession:
    """一次 bench.run。

    `BenchmarkRunner` 是**一次性**的——`__init__` 就把 `cfg.episode_plan()` 绑死,
    `finished` 是不可逆的单向开关。所以每次 invocation 新建一个,cfg 用
    `replace()` 克隆,绝不改模板 cfg(改了会污染下一次)。
    """

    def __init__(
        self,
        node: Any,
        cfg_template: BaseBenchmarkConfig,
        adapter: Any,
        plan: dict[str, Any],
        invocation_id: str,
        log=print,
    ) -> None:
        self.invocation_id = invocation_id
        self._log = log
        self.cfg = replace(
            cfg_template,
            task_ids=[int(t) for t in plan["task_ids"]],
            init_state_ids=[int(i) for i in plan["init_state_ids"]],
            num_runs=int(plan["num_runs"]),
            seed=int(plan["seed"]),
            max_steps=int(plan["max_steps"]),
        )
        # 权威兜底校验。endpoint 侧拿缓存的 capabilities 也校验了一遍,但那份可能
        # 陈旧;这里抛出来会被上层转成 rejected 回执,而不是"跑起来之后失败"。
        resolve_and_validate_plan(self.cfg, adapter)

        # **adapter 会回读 cfg,必须让它和 runner 看的是同一个对象。**
        # LiberoAdapter 在构造时存了 `self.cfg`,并在 reset_episode / _ensure_env
        # 里回读 `seed` / `layout_mode` / `control_mode` / `num_steps_wait`。
        # 不指过去的话,bench.run 传的 seed 到不了 adapter —— LIBERO 会一直用
        # 模板里的 seed 播种家具布局,**分数照出、不报任何错**。
        # 判据是「有没有 cfg 这个属性」,不是「cfg 是不是 None」:后者会让 cfg
        # 初始为 None 的 adapter 静默拿不到回写通道。
        if hasattr(adapter, "cfg"):
            adapter.cfg = self.cfg

        self.runner = BenchmarkRunner(
            node,
            self.cfg,
            adapter,
            log=log,
            # 文件名的时间戳只到秒,同一秒内两次 invocation 会写到同一路径
            result_tag=invocation_id,
            # Skill Runtime 的生命周期长于单次 invocation,关了 env 下一次就要重载
            close_adapter_on_finish=False,
        )
        self.started_at = time.time()
        self._last_progress = -1

    # ---- 生命周期 ----

    def start(self) -> None:
        self.runner.policy_command("start")
        self.runner.advance()

    def on_action(self, command: Any) -> None:
        self.runner.on_action(command)

    def cancel(self) -> None:
        self.runner.finish("cancelled")

    def fail(self, code: str, message: str) -> None:
        self.runner.fail_current(code, message)

    @property
    def finished(self) -> bool:
        return self.runner.finished

    @property
    def result_path(self) -> str | None:
        return self.runner.result_path

    # ---- 回执 ----

    def progress_report(self) -> dict[str, Any] | None:
        """episode 边界才发一条,不要每步发。没跨界就返回 None。"""
        index = self.runner.plan_index
        if index == self._last_progress:
            return None
        self._last_progress = index
        return {
            "type": REPORT_PROGRESS,
            "invocation_id": self.invocation_id,
            "episode_index": index,
            "total": len(self.runner.plan),
            "successes": self.runner.result.successes,
        }

    def result_report(self) -> dict[str, Any]:
        result = self.runner.result
        payload = result.to_dict()
        episodes = payload.get("episodes") or []
        truncated = len(episodes) > MAX_EPISODES_IN_PAYLOAD
        return {
            "type": REPORT_RESULT,
            "invocation_id": self.invocation_id,
            "status": result.status,
            "payload": {
                # status 与 success_rate 必须并存:success_rate 的分母是
                # total_episodes 而不是 completed_episodes,批次中途挂掉时两者不等,
                # 只看 success_rate 会把「跑挂了」读成「分数低」。
                "status": result.status,
                "primary_metric": result.primary_metric,
                "success_rate": payload.get("success_rate"),
                "successes": payload.get("successes"),
                "total_episodes": payload.get("total_episodes"),
                "completed_episodes": payload.get("completed_episodes"),
                "valid_episodes": payload.get("valid_episodes"),
                "timed_out_episodes": payload.get("timed_out_episodes"),
                "num_steps": payload.get("num_steps"),
                "mean_policy_latency_ms": payload.get("mean_policy_latency_ms"),
                "elapsed_s": payload.get("elapsed_s"),
                "result_path": self.runner.result_path,
                "episodes": [] if truncated else [_slim(e) for e in episodes],
                "episodes_truncated": truncated,
            },
        }


def _slim(episode: dict[str, Any]) -> dict[str, Any]:
    """只回传判分需要的字段;task_description 之类留在 result.json 里。"""
    return {
        "task_id": episode.get("task_id"),
        "init_state_id": episode.get("init_state_id"),
        "run_index": episode.get("run_index"),
        "seed": episode.get("episode_seed"),
        "success": episode.get("success"),
        "termination": episode.get("termination"),
        "error_code": episode.get("error_code"),
    }


__all__ = [
    "CMD_CANCEL",
    "CMD_DESCRIBE",
    "CMD_RUN",
    "HEARTBEAT_PERIOD_S",
    "MAX_EPISODES_IN_PAYLOAD",
    "REPORT_ACCEPTED",
    "REPORT_CAPABILITIES",
    "REPORT_HEARTBEAT",
    "REPORT_PROGRESS",
    "REPORT_REJECTED",
    "REPORT_RESULT",
    "WIRE_VERSION",
    "BenchSession",
    "WireError",
    "capabilities_payload",
    "decode",
    "encode",
]
