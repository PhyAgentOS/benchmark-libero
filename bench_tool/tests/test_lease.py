"""注册租约。EndpointLease 逐字移植自 bench_endpoint（endpoint_id 变了），
行为必须与 forge_tool 的租约语义一致：注册 → ttl/3 处重注册 → 注销。"""

from __future__ import annotations

from forge_tool import EndpointRegistryResponse, make_endpoint_registry_response_envelope

from bench_tool.descriptor import BENCH_DESCRIPTOR
from bench_tool.lease import EndpointLease


def test_lease_emits_registration_then_backs_off() -> None:
    lease = EndpointLease(BENCH_DESCRIPTOR, "instance-1")
    first = lease.poll(0.0)
    assert first is not None
    assert first.message_type == "endpoint.register"
    # 回执没到之前不重发
    assert lease.poll(0.1) is None


def test_lease_renews_at_one_third_of_ttl() -> None:
    lease = EndpointLease(BENCH_DESCRIPTOR, "instance-1")
    request = lease.poll(0.0)
    response = make_endpoint_registry_response_envelope(
        EndpointRegistryResponse(
            operation="register",
            status="accepted",
            registry_revision=1,
            lease_ttl_ms=3000,
        ),
        request,
    )
    assert lease.acknowledge(response, 0.0) is True
    # ttl/3 = 1s：所以 tick 周期必须 ≤1000ms，否则续不上
    assert lease.poll(0.5) is None
    assert lease.poll(1.1) is not None


def test_stop_emits_unregister_once() -> None:
    lease = EndpointLease(BENCH_DESCRIPTOR, "instance-1")
    first = lease.stop()
    assert first is not None
    assert first.message_type == "endpoint.unregister"
    assert lease.stop() is None  # 幂等
    assert lease.poll(999.0) is None  # 停了就不再注册
