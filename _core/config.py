"""所有 benchmark 共有的配置字段与批次工具。

benchmark 特有的字段（如 LIBERO 的 layout_mode / control_mode）由各自的
config 模块继承 BaseBenchmarkConfig 追加，核心不认识它们。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Iterable

import yaml


def parse_id_list(spec: Any, default: Iterable[int]) -> list[int]:
    """解析 episode 规格：'all' / '0-9' / '0,2-4,8' / [0,1,2] / None。"""
    if spec is None or spec == "all":
        return list(default)
    if isinstance(spec, int):
        return [spec]
    if isinstance(spec, (list, tuple)):
        return [int(x) for x in spec]

    out: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def _parse_name_list(spec: Any) -> list[str]:
    """suite 名单：None / "" → []；"a,b" 或 [a, b] → 去重保序的列表。"""
    if spec is None or spec == "":
        return []
    items = spec if isinstance(spec, (list, tuple)) else str(spec).split(",")
    out: list[str] = []
    for item in items:
        name = str(item).strip()
        if name and name not in out:
            out.append(name)
    return out


def derive_episode_seed(
    base_seed: int, task_id: int, init_state_id: int, run_index: int
) -> int:
    """把 episode 身份映射成稳定种子。

    刻意不用内建 `hash()`：它对 str/bytes 带 PYTHONHASHSEED 随机化，跨进程不一致，
    正好会毁掉我们想要的复现性。这里用固定整数混合，纯算术、跨进程恒定。
    """
    h = (int(base_seed) & 0xFFFF) * 1_000_003
    h ^= (int(task_id) + 1) * 73_856_093
    h ^= (int(init_state_id) + 1) * 19_349_663
    h ^= (int(run_index) + 1) * 83_492_791
    return h & 0x7FFF_FFFF


@dataclass
class BaseBenchmarkConfig:
    """所有 benchmark 共有的字段。"""

    benchmark: str = "unknown"
    suite: str = ""
    task_ids: list[int] = field(default_factory=list)
    init_state_ids: list[int] = field(default_factory=list)
    num_runs: int = 1
    max_steps: int = 300
    seed: int = 0

    policy_id: str = ""
    send_instruction: bool = True

    result_dir: str = "results/benchmark"
    record_dir: str | None = None

    # ---- 服务模式（Forge Tool 接入）----
    # 三个字段默认全关，**不配这几项时行为与改造前逐字一致**：
    # serve=False 时 run_node 走 _run_autorun()，就是原来那段代码。
    # LIBERO / RoboCasa / CALVIN 现有的 config.yaml 都不含这些键，照常加载。
    serve: bool = False
    # 单次 bench.run 的护栏，0 = 不限。没有这道闸，调用方会一口气开满整个
    # suite 然后跑几小时，而且不报错。只在 serve 模式下由 endpoint 侧读取。
    max_total_episodes: int = 0
    max_max_steps: int = 0

    # ---- 多 suite（serve 模式下 run 可按次指定 suite）----
    # `suites` 是本配置（≈ 一组权重）允许跑的 suite 白名单。空 = 只允许 `suite`，
    # 与改造前逐字一致。很多权重只在单个 suite 上训练，跑白名单外的 suite 不报错
    # 但分数是 0，所以白名单外一律拒绝。
    suites: list[str] = field(default_factory=list)
    # 按 suite 覆盖默认 max_steps。查找顺序见 default_max_steps()。
    max_steps_by_suite: dict[str, int] = field(default_factory=dict)
    # 各 benchmark 自己的官方步数口径（子类覆盖）；ClassVar 不进 dataclass 字段。
    BUILTIN_MAX_STEPS: ClassVar[dict[str, int]] = {}

    def allowed_suites(self) -> list[str]:
        return list(self.suites) if self.suites else [self.suite]

    def default_max_steps(self, suite: str | None = None) -> int:
        """某个 suite 的默认 max_steps。

        顺序：配置里的 max_steps_by_suite → 默认 suite 用 `max_steps`（保持旧
        profile 的数值不变）→ benchmark 官方口径 → `max_steps`。
        """
        name = suite or self.suite
        if name in self.max_steps_by_suite:
            return int(self.max_steps_by_suite[name])
        if name == self.suite:
            return int(self.max_steps)
        return int(self.BUILTIN_MAX_STEPS.get(name, self.max_steps))

    @property
    def total_episodes(self) -> int:
        return len(self.task_ids) * len(self.init_state_ids) * self.num_runs

    def episode_plan(self) -> list[tuple[int, int, int]]:
        """批次 = task_ids × init_state_ids × num_runs 的笛卡尔积，串行执行。"""
        return [
            (task_id, init_state_id, run_index)
            for task_id in self.task_ids
            for init_state_id in self.init_state_ids
            for run_index in range(self.num_runs)
        ]


def read_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def fill_base(cfg: BaseBenchmarkConfig, data: dict[str, Any]) -> BaseBenchmarkConfig:
    """把 YAML 里的共有字段填进配置对象，并做合法性检查。"""
    cfg.suite = str(data.get("suite", cfg.suite))
    cfg.num_runs = int(data.get("num_runs", cfg.num_runs))
    cfg.max_steps = int(data.get("max_steps", cfg.max_steps))
    cfg.seed = int(data.get("seed", cfg.seed))
    cfg.policy_id = str(data.get("policy_id", cfg.policy_id))
    cfg.send_instruction = bool(data.get("send_instruction", cfg.send_instruction))
    cfg.result_dir = str(data.get("result_dir", cfg.result_dir))
    cfg.record_dir = data.get("record_dir", cfg.record_dir)
    # 缺省即关闭：现有 config.yaml 不含这三个键，data.get 直接回落到 dataclass 默认值
    cfg.serve = bool(data.get("serve", cfg.serve))
    cfg.max_total_episodes = int(data.get("max_total_episodes", cfg.max_total_episodes))
    cfg.max_max_steps = int(data.get("max_max_steps", cfg.max_max_steps))

    cfg.suites = _parse_name_list(data.get("suites"))
    raw_steps = data.get("max_steps_by_suite") or {}
    if not isinstance(raw_steps, dict):
        raise ValueError(f"max_steps_by_suite 必须是 suite→步数 的映射，收到 {raw_steps!r}")
    cfg.max_steps_by_suite = {str(k): int(v) for k, v in raw_steps.items()}

    if cfg.num_runs < 1:
        raise ValueError(f"num_runs 必须 ≥1，收到 {cfg.num_runs}")
    if cfg.max_steps < 1:
        raise ValueError(f"max_steps 必须 ≥1，收到 {cfg.max_steps}")
    if cfg.suites and cfg.suite not in cfg.suites:
        raise ValueError(f"suites 必须包含默认 suite {cfg.suite!r}，收到 {cfg.suites}")
    bad_steps = {k: v for k, v in cfg.max_steps_by_suite.items() if v < 1}
    if bad_steps:
        raise ValueError(f"max_steps_by_suite 的步数必须 ≥1，收到 {bad_steps}")

    # task/init_state 的默认全集由 adapter 在拿到 suite 后回填（见 resolve_and_validate_plan）
    cfg.task_ids = parse_id_list(data.get("task_ids"), [])
    cfg.init_state_ids = parse_id_list(data.get("init_state_ids"), [])
    return cfg


def resolve_and_validate_plan(cfg: BaseBenchmarkConfig, adapter: Any) -> None:
    """回填 'all'，并**逐 task** 校验 id 范围。

    必须 fail fast：越界 id 如果留到 episode 推进时才炸，异常会在事件循环的 except
    分支里再次抛出，冲出循环导致整批结果不落盘。
    """
    n_tasks = adapter.num_tasks
    if not cfg.task_ids:
        cfg.task_ids = list(range(n_tasks))

    bad = [t for t in cfg.task_ids if not 0 <= t < n_tasks]
    if bad:
        raise ValueError(f"{cfg.suite!r} 只有 {n_tasks} 个 task，越界 task_ids: {bad}")

    # 逐 task 取 init state 数：不同 task 的数量未必相同，不能拿 task_ids[0] 代表全体。
    counts = {t: adapter.num_init_states(t) for t in cfg.task_ids}
    if not cfg.init_state_ids:
        cfg.init_state_ids = list(range(min(counts.values())))

    for t in cfg.task_ids:
        bad = [i for i in cfg.init_state_ids if not 0 <= i < counts[t]]
        if bad:
            raise ValueError(
                f"task {t} 只有 {counts[t]} 个 init state，越界 init_state_ids: {bad}"
            )

    if not cfg.task_ids or not cfg.init_state_ids:
        raise ValueError("批次为空：task_ids 或 init_state_ids 解析后没有任何元素")
