from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from headroom.dashboard import get_dashboard_html
from headroom.proxy import helpers as proxy_helpers


class _StatsStub:
    def __init__(self, calls: dict[str, int], key: str, payload: dict):
        self._calls = calls
        self._key = key
        self._payload = payload

    def get_stats(self) -> dict:
        self._calls[self._key] += 1
        return dict(self._payload)


class _ToinStub:
    def get_stats(self) -> dict:
        return {"patterns": 0}


@pytest.fixture(autouse=True)
def _reset_rtk_stats_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)
    monkeypatch.setenv("HEADROOM_REQUIRE_RUST_CORE", "false")
    proxy_helpers._rtk_stats_cache.update(
        {"expires_at": 0.0, "has_value": False, "tool": None, "value": None}
    )
    proxy_helpers._rtk_session_baseline.update(
        {"initialized": False, "tool": None, "total_commands": 0, "tokens_saved": 0}
    )


def test_get_rtk_stats_memoizes_subprocess_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)
    now = {"value": 100.0}
    calls = {"run": 0}
    totals = [
        {"total_commands": 7, "total_saved": 1234},
        {"total_commands": 9, "total_saved": 1500},
    ]

    def _fake_run(*args, **kwargs):
        calls["run"] += 1
        summary = totals[min(calls["run"] - 1, len(totals) - 1)]
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"summary": summary}),
        )

    monkeypatch.setattr(proxy_helpers.time, "monotonic", lambda: now["value"])
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/rtk")
    monkeypatch.setattr(subprocess, "run", _fake_run)

    first = proxy_helpers._get_rtk_stats()
    second = proxy_helpers._get_rtk_stats()

    assert first == second
    assert first == {
        "tool": "rtk",
        "label": "RTK",
        "installed": True,
        "total_commands": 0,
        "tokens_saved": 0,
        "avg_savings_pct": 0.0,
    }
    assert calls["run"] == 1

    now["value"] += proxy_helpers.RTK_STATS_CACHE_TTL_SECONDS + 0.1
    third = proxy_helpers._get_rtk_stats()

    assert third == {
        "tool": "rtk",
        "label": "RTK",
        "installed": True,
        "total_commands": 2,
        "tokens_saved": 266,
        "avg_savings_pct": 0.0,
    }
    assert calls["run"] == 2


def test_get_context_tool_stats_reads_lean_ctx_gain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_CONTEXT_TOOL", "lean-ctx")
    now = {"value": 100.0}
    calls = {"run": 0}
    totals = [
        {"total_commands": 3, "tokens_saved": 400, "avg_savings_pct": 12.5},
        {"total_commands": 5, "tokens_saved": 475, "avg_savings_pct": 15.0},
    ]

    def _fake_run(args, **kwargs):
        calls["run"] += 1
        assert args == ["/usr/bin/lean-ctx", "gain", "--json"]
        summary = totals[min(calls["run"] - 1, len(totals) - 1)]
        return SimpleNamespace(returncode=0, stdout=json.dumps({"summary": summary}))

    monkeypatch.setattr(proxy_helpers.time, "monotonic", lambda: now["value"])
    monkeypatch.setattr(
        "headroom.lean_ctx.get_lean_ctx_path",
        lambda: Path("/usr/bin/lean-ctx"),
    )
    monkeypatch.setattr(subprocess, "run", _fake_run)

    first = proxy_helpers._get_context_tool_stats()
    second = proxy_helpers._get_context_tool_stats()

    assert first == second
    assert first == {
        "tool": "lean-ctx",
        "label": "lean-ctx",
        "installed": True,
        "total_commands": 0,
        "tokens_saved": 0,
        "avg_savings_pct": 12.5,
    }
    assert calls["run"] == 1

    now["value"] += proxy_helpers.CONTEXT_TOOL_STATS_CACHE_TTL_SECONDS + 0.1
    third = proxy_helpers._get_context_tool_stats()

    assert third == {
        "tool": "lean-ctx",
        "label": "lean-ctx",
        "installed": True,
        "total_commands": 2,
        "tokens_saved": 75,
        "avg_savings_pct": 15.0,
    }
    assert calls["run"] == 2


def test_stats_cached_query_reuses_short_ttl_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import headroom.proxy.server as server
    from headroom.proxy.server import ProxyConfig, create_app

    calls = {"store": 0, "telemetry": 0, "feedback": 0, "context_tool": 0}
    now = {"value": 100.0}

    monkeypatch.setattr(server.time, "monotonic", lambda: now["value"])
    monkeypatch.setattr(
        server,
        "get_compression_store",
        lambda: _StatsStub(calls, "store", {"entry_count": 1, "max_entries": 100}),
    )
    monkeypatch.setattr(
        server,
        "get_telemetry_collector",
        lambda: _StatsStub(calls, "telemetry", {"enabled": True}),
    )
    monkeypatch.setattr(
        server,
        "get_compression_feedback",
        lambda: _StatsStub(calls, "feedback", {}),
    )

    def _fake_context_tool_stats() -> dict[str, int | bool | float | str]:
        calls["context_tool"] += 1
        return {
            "tool": "rtk",
            "label": "RTK",
            "installed": True,
            "total_commands": 1,
            "tokens_saved": 5,
            "avg_savings_pct": 10.0,
        }

    monkeypatch.setattr(server, "_get_context_tool_stats", _fake_context_tool_stats)
    monkeypatch.setattr(server, "get_toin", lambda: _ToinStub())

    app = create_app(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
            ccr_inject_tool=False,
            ccr_handle_responses=False,
            ccr_context_tracking=False,
        )
    )

    with TestClient(app) as client:
        first = client.get("/stats?cached=1")
        second = client.get("/stats?cached=1")
        now["value"] += 5.1
        third = client.get("/stats?cached=1")
        uncached = client.get("/stats")

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 200
    assert uncached.status_code == 200

    assert calls == {"store": 3, "telemetry": 3, "feedback": 3, "context_tool": 3}
    assert first.json()["context_tool"]["configured"] == "rtk"
    assert first.json()["context_tool"]["label"] == "RTK"
    assert first.json()["cli_filtering"]["tokens_saved"] == 5
    assert first.json()["tokens"]["saved"] == 5
    assert first.json()["tokens"]["proxy_compression_saved"] == 0
    assert first.json()["tokens"]["cli_filtering_saved"] == 5
    assert first.json()["tokens"]["rtk_saved"] == 5
    assert first.json()["tokens"]["lean_ctx_saved"] == 0
    assert first.json()["tokens"]["all_layers_saved"] == 5
    assert (
        first.json()["tokens"]["savings_percent"]
        == first.json()["tokens"]["all_layers_savings_percent"]
    )
    assert first.json()["savings"]["by_layer"]["compression"]["tokens"] == 0
    assert first.json()["savings"]["by_layer"]["compression"]["cli_filtering_tokens"] == 5
    assert first.json()["savings"]["by_layer"]["compression"]["rtk_tokens"] == 5
    assert first.json()["savings"]["by_layer"]["compression"]["lean_ctx_tokens"] == 0
    assert first.json()["savings"]["by_layer"]["compression"]["all_layers_tokens"] == 5


def test_stats_reports_lean_ctx_as_selected_cli_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import headroom.proxy.server as server
    from headroom.proxy.server import ProxyConfig, create_app

    monkeypatch.setattr(
        server,
        "get_compression_store",
        lambda: _StatsStub({"store": 0}, "store", {}),
    )
    monkeypatch.setattr(
        server,
        "get_telemetry_collector",
        lambda: _StatsStub({"telemetry": 0}, "telemetry", {}),
    )
    monkeypatch.setattr(
        server,
        "get_compression_feedback",
        lambda: _StatsStub({"feedback": 0}, "feedback", {}),
    )
    monkeypatch.setattr(
        server,
        "_get_context_tool_stats",
        lambda: {
            "tool": "lean-ctx",
            "label": "lean-ctx",
            "installed": True,
            "total_commands": 1,
            "tokens_saved": 9,
            "avg_savings_pct": 11.0,
        },
    )
    monkeypatch.setattr(server, "get_toin", lambda: _ToinStub())

    app = create_app(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
            ccr_inject_tool=False,
            ccr_handle_responses=False,
            ccr_context_tracking=False,
        )
    )

    with TestClient(app) as client:
        response = client.get("/stats")

    payload = response.json()
    assert response.status_code == 200
    assert payload["context_tool"]["configured"] == "lean-ctx"
    assert payload["savings"]["by_layer"]["cli_filtering"]["label"] == "lean-ctx"
    assert payload["tokens"]["cli_filtering_saved"] == 9
    assert payload["tokens"]["rtk_saved"] == 0
    assert payload["tokens"]["lean_ctx_saved"] == 9
    assert payload["savings"]["by_layer"]["compression"]["rtk_tokens"] == 0
    assert payload["savings"]["by_layer"]["compression"]["lean_ctx_tokens"] == 9


def test_cost_merge_uses_generic_cli_filtering_name() -> None:
    from headroom.proxy.cost import merge_cost_stats

    payload = merge_cost_stats(
        {"savings_usd": 1.23456, "other": "kept"},
        {"totals": {"net_savings_usd": 0.25}},
        cli_tokens_avoided=12,
    )

    assert payload is not None
    assert payload["compression_savings_usd"] == 1.2346
    assert payload["cache_savings_usd"] == 0.25
    assert payload["cli_tokens_avoided"] == 12
    assert payload["cli_filtering_tokens_avoided"] == 12
    assert payload["cli_filtering_tokens_included_in_compression"] is True
    assert payload["cli_tokens_included_in_compression"] is True


def test_session_summary_uses_generic_cli_filtering_keys() -> None:
    from headroom.proxy.cost import build_session_summary

    proxy = SimpleNamespace(
        config=SimpleNamespace(mode="token"),
        logger=SimpleNamespace(_logs=[]),
        cost_tracker=SimpleNamespace(
            stats=lambda: {
                "cost_with_headroom_usd": 2.0,
                "savings_usd": 0.5,
            }
        ),
    )
    metrics = SimpleNamespace(
        requests_by_model={"gpt-test": 1},
        tokens_saved_total=20,
    )

    payload = build_session_summary(
        proxy,
        metrics,
        {"totals": {"net_savings_usd": 0.2}},
        cli_tokens_avoided=7,
        total_tokens_before=100,
    )

    assert payload["compression"]["cli_filtering_tokens_avoided"] == 7
    assert payload["compression"]["total_tokens_saved_with_cli_filtering"] == 27
    assert payload["compression"]["total_tokens_before_with_cli_filtering"] == 100
    assert payload["compression"]["rtk_tokens_avoided"] == 7
    assert payload["cost"]["breakdown"]["cli_filtering_savings_usd"] is None
    assert payload["cost"]["breakdown"]["rtk_savings_usd"] is None


def test_stats_reset_clears_runtime_proxy_counters(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import headroom.proxy.server as server
    from headroom.proxy.loopback_guard import require_loopback
    from headroom.proxy.server import ProxyConfig, create_app

    monkeypatch.setattr(
        server,
        "get_compression_store",
        lambda: _StatsStub({"store": 0}, "store", {}),
    )
    monkeypatch.setattr(
        server,
        "get_telemetry_collector",
        lambda: _StatsStub({"telemetry": 0}, "telemetry", {}),
    )
    monkeypatch.setattr(
        server,
        "get_compression_feedback",
        lambda: _StatsStub({"feedback": 0}, "feedback", {}),
    )
    monkeypatch.setattr(server, "_get_context_tool_stats", lambda: None)
    monkeypatch.setattr(server, "get_toin", lambda: _ToinStub())

    app = create_app(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
            ccr_inject_tool=False,
            ccr_handle_responses=False,
            ccr_context_tracking=False,
        )
    )
    app.dependency_overrides[require_loopback] = lambda: None

    with TestClient(app) as client:
        proxy = client.app.state.proxy
        proxy.metrics.tokens_saved_total = 123
        proxy.metrics.tokens_input_total = 456
        proxy.metrics.requests_total = 2

        before = client.get("/stats").json()
        reset = client.post("/stats/reset")
        after = client.get("/stats").json()

    assert before["tokens"]["proxy_compression_saved"] == 123
    assert reset.status_code == 200
    assert after["tokens"]["proxy_compression_saved"] == 0
    assert after["tokens"]["input"] == 0
    assert after["requests"]["total"] == 0


def test_dashboard_uses_cached_stats_and_lazy_history_feed_polling() -> None:
    html = get_dashboard_html()

    assert "fetch('/stats?cached=1')" in html
    assert "@click=\"setViewMode('history')\"" in html
    assert '@click="toggleFeed()"' in html
    assert "this.viewMode === 'history'" in html
    assert "this.feedOpen" in html
    assert "CLI Filtering (rtk)" not in html
    assert "RTK Filtered" not in html
    assert "|| 'RTK'" not in html
    assert "rtkShareOfTotal" not in html
    assert "Lean-ctx" in html
    assert "Context Tool" in html
    assert "cliFilteringLabel + ' Filtered'" in html
