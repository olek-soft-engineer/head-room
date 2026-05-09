from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from headroom.cli import wrap as wrap_cli
from headroom.cli.main import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_remove_claude_rtk_hooks_preserves_unrelated_hooks(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "model": "opus",
                "hooks": {
                    "PreToolUse": [
                        {
                            "matcher": "Bash",
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "/Users/test/.claude/hooks/rtk-rewrite.sh",
                                },
                                {"type": "command", "command": "echo keep"},
                            ],
                        }
                    ],
                    "SessionStart": [
                        {"matcher": "startup", "hooks": [{"type": "command", "command": "keep"}]}
                    ],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert wrap_cli._remove_claude_rtk_hooks(settings) is True

    payload = json.loads(settings.read_text(encoding="utf-8"))
    pre_tool_hooks = payload["hooks"]["PreToolUse"][0]["hooks"]
    assert pre_tool_hooks == [{"type": "command", "command": "echo keep"}]
    assert payload["hooks"]["SessionStart"][0]["hooks"][0]["command"] == "keep"


def test_unwrap_claude_removes_mcp_rtk_and_stops_proxy(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = str(tmp_path)
    monkeypatch.setenv("HOME", home)
    monkeypatch.setenv("USERPROFILE", home)
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    settings = claude_dir / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "hooks": {
                    "PreToolUse": [
                        {
                            "matcher": "Bash",
                            "hooks": [
                                {"type": "command", "command": str(claude_dir / "rtk-rewrite.sh")}
                            ],
                        }
                    ]
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    stopped: list[int] = []
    unregistered: list[str] = []

    class Registrar:
        def detect(self) -> bool:
            return True

        def unregister_server(self, server_name: str) -> bool:
            unregistered.append(server_name)
            return True

    with (
        patch("headroom.mcp_registry.ClaudeRegistrar", return_value=Registrar()),
        patch(
            "headroom.cli.wrap._stop_local_proxy_for_unwrap",
            side_effect=lambda port: stopped.append(port) or "stopped",
        ),
    ):
        result = runner.invoke(main, ["unwrap", "claude", "--port", "9999"])

    assert result.exit_code == 0, result.output
    assert unregistered == ["headroom", "codebase-memory-mcp"]
    assert stopped == [9999]
    assert "Stopped local Headroom proxy on port 9999" in result.output
    assert "hooks" not in json.loads(settings.read_text(encoding="utf-8"))


def test_unwrap_claude_keep_flags_skip_cleanup(
    runner: CliRunner,
) -> None:
    with (
        patch("headroom.mcp_registry.ClaudeRegistrar") as registrar,
        patch("headroom.cli.wrap._remove_claude_rtk_hooks") as remove_rtk,
        patch("headroom.cli.wrap._stop_local_proxy_for_unwrap") as stop_proxy,
    ):
        result = runner.invoke(
            main,
            ["unwrap", "claude", "--keep-mcp", "--keep-rtk", "--no-stop-proxy"],
        )

    assert result.exit_code == 0, result.output
    registrar.assert_not_called()
    remove_rtk.assert_not_called()
    stop_proxy.assert_not_called()
