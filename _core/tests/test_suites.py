"""多 suite：配置解析、默认步数口径、capabilities 选 suite、BenchSession 跑指定 suite。

这里盯的是"换错 suite 照样出分"的路径：校验对着上一次 session 的 suite 做、
结果文件写成默认 suite 的名字、adapter 没收到本次 suite —— 跑一遍都不报错。
"""

from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import dataclass
from typing import ClassVar

import pytest

from _core import session as S
from _core.config import BaseBenchmarkConfig, fill_base

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "_core_tests_conftest", os.path.join(_here, "conftest.py")
)
assert _spec is not None and _spec.loader is not None
_conftest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_conftest)
FakeAdapter = _conftest.FakeAdapter
MultiSuiteAdapter = _conftest.MultiSuiteAdapter
drive = _conftest.drive
make_plan = _conftest.make_plan


@pytest.fixture
def multi_cfg(cfg_template) -> BaseBenchmarkConfig:
    cfg_template.suites = ["fake_suite", "big_suite"]
    return cfg_template


# ============================================================ 配置解析


def test_old_config_without_suites_only_allows_default() -> None:
    cfg = fill_base(BaseBenchmarkConfig(), {"suite": "a", "max_steps": 123})
    assert cfg.suites == []
    assert cfg.allowed_suites() == ["a"]
    assert cfg.default_max_steps() == 123


def test_suites_accept_list_or_comma_string_and_dedupe() -> None:
    cfg = fill_base(BaseBenchmarkConfig(), {"suite": "a", "suites": "a, b,a,"})
    assert cfg.allowed_suites() == ["a", "b"]
    cfg = fill_base(BaseBenchmarkConfig(), {"suite": "a", "suites": ["b", "a"]})
    assert cfg.allowed_suites() == ["b", "a"]


def test_suites_must_contain_default_suite() -> None:
    with pytest.raises(ValueError, match="必须包含默认 suite"):
        fill_base(BaseBenchmarkConfig(), {"suite": "a", "suites": ["b", "c"]})


@pytest.mark.parametrize("raw", [[1, 2], "b:10", 5])
def test_max_steps_by_suite_must_be_a_mapping(raw) -> None:
    with pytest.raises(ValueError, match="映射"):
        fill_base(BaseBenchmarkConfig(), {"suite": "a", "max_steps_by_suite": raw})


def test_max_steps_by_suite_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="≥1"):
        fill_base(BaseBenchmarkConfig(), {"suite": "a", "max_steps_by_suite": {"b": 0}})


@dataclass
class _BuiltinCfg(BaseBenchmarkConfig):
    BUILTIN_MAX_STEPS: ClassVar[dict[str, int]] = {"a": 999, "b": 111, "c": 222}


def test_default_max_steps_lookup_order() -> None:
    cfg = fill_base(
        _BuiltinCfg(),
        {"suite": "a", "max_steps": 300, "max_steps_by_suite": {"c": 5}},
    )
    assert cfg.default_max_steps() == 300  # 默认 suite：沿用 max_steps，不被官方口径改掉
    assert cfg.default_max_steps("a") == 300
    assert cfg.default_max_steps("b") == 111  # 官方口径
    assert cfg.default_max_steps("c") == 5  # 配置覆盖优先于官方口径
    assert cfg.default_max_steps("zzz") == 300  # 都没有 → max_steps

    cfg.max_steps_by_suite = {"a": 7}
    assert cfg.default_max_steps("a") == 7  # 覆盖对默认 suite 也生效


# ============================================================ capabilities


def test_capabilities_default_suite_lists_whitelist(multi_cfg) -> None:
    caps = S.capabilities_payload(multi_cfg, MultiSuiteAdapter())
    assert caps["suite"] == "fake_suite"
    assert caps["suites"] == ["fake_suite", "big_suite"]
    assert caps["num_tasks"] == 2


def test_capabilities_for_other_suite_reads_that_suite(multi_cfg) -> None:
    adapter = MultiSuiteAdapter()
    multi_cfg.max_steps_by_suite = {"big_suite": 77}
    caps = S.capabilities_payload(multi_cfg, adapter, suite="big_suite")
    assert caps["suite"] == "big_suite"
    assert caps["num_tasks"] == 3
    assert [t["num_init_states"] for t in caps["tasks"]] == [1, 1, 4]
    assert caps["defaults"]["max_steps"] == 77
    # describe 是只读的：不能为了数 task 去改 adapter 的当前 suite
    assert adapter.cfg is None


def test_capabilities_reject_suite_outside_whitelist(multi_cfg) -> None:
    with pytest.raises(ValueError, match="不在本配置允许的 suites"):
        S.capabilities_payload(multi_cfg, MultiSuiteAdapter(), suite="libero_90")


def test_capabilities_reject_switch_when_adapter_has_no_suite_view(multi_cfg) -> None:
    with pytest.raises(ValueError, match="不支持切换 suite"):
        S.capabilities_payload(multi_cfg, FakeAdapter(), suite="big_suite")


# ============================================================ BenchSession


def test_session_runs_requested_suite(multi_cfg, node) -> None:
    adapter = MultiSuiteAdapter()
    # task 2 / init 3 只在 big_suite 里存在：对默认 suite 校验会直接拒掉
    session = S.BenchSession(
        node, multi_cfg, adapter,
        make_plan(suite="big_suite", task_ids=[2], init_state_ids=[3]), "inv-big",
        log=lambda _m: None,
    )
    assert session.cfg.suite == "big_suite"
    assert multi_cfg.suite == "fake_suite"  # 模板不被改
    assert adapter.cfg is session.cfg

    drive(session)
    assert adapter.resets[-1]["cfg_suite"] == "big_suite"
    assert os.path.basename(session.result_path).startswith("big_suite_")
    with open(session.result_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["suite"] == "big_suite"
    assert payload["episodes"][0]["episode_id"].startswith("big_suite_t2_i3_")


def test_session_validates_against_its_own_suite_not_the_previous_one(multi_cfg, node) -> None:
    """上一个 session 把 adapter.cfg 留在 big_suite；这次回到默认 suite 的
    task 1 / init 1 —— 拿 big_suite 的清单校验会误判越界（big_suite 的 task 1 只有 1 个）。"""
    adapter = MultiSuiteAdapter()
    S.BenchSession(
        node, multi_cfg, adapter, make_plan(suite="big_suite", task_ids=[2]), "inv-1",
        log=lambda _m: None,
    )
    assert adapter.cfg.suite == "big_suite"

    session = S.BenchSession(
        node, multi_cfg, adapter, make_plan(task_ids=[1], init_state_ids=[1]), "inv-2",
        log=lambda _m: None,
    )
    assert session.cfg.suite == "fake_suite"
    assert adapter.cfg.suite == "fake_suite"


def test_session_rejects_wrong_suite_bounds(multi_cfg, node) -> None:
    with pytest.raises(ValueError):
        S.BenchSession(
            node, multi_cfg, MultiSuiteAdapter(),
            make_plan(suite="big_suite", task_ids=[0], init_state_ids=[1]), "inv-x",
            log=lambda _m: None,
        )


def test_session_rejects_suite_outside_whitelist(cfg_template, node) -> None:
    adapter = MultiSuiteAdapter()
    with pytest.raises(ValueError, match="不在本配置允许的 suites"):
        S.BenchSession(
            node, cfg_template, adapter, make_plan(suite="big_suite"), "inv-x",
            log=lambda _m: None,
        )
    assert adapter.cfg is None  # 拒绝前不得回写 adapter
