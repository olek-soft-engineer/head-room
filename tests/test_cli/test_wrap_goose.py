"""Tests for `headroom wrap goose` command (PR-G1, Phase G)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from headroom.cli import wrap as wrap_mod
from headroom.cli.main import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_wrap_goose_sets_provider_envs(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OPENAI_BASE_URL, OPENAI_API_BASE, ANTHROPIC_BASE_URL are set on launch."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)

    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with patch.object(wrap_mod.shutil, "which", return_value="goose"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=fake_launch_tool):
            with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=Path("/tmp/rtk")):
                result = runner.invoke(main, ["wrap", "goose", "--port", "9000", "--", "session"])

    assert result.exit_code == 0, result.output
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:9000/v1"
    assert env["OPENAI_API_BASE"] == "http://127.0.0.1:9000/v1"
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:9000"
    assert captured["tool_label"] == "GOOSE"
    assert captured["agent_type"] == "goose"
    assert captured["args"] == ("session",)


def test_wrap_goose_injects_rtk_into_goosehints(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RTK block must be written to .goosehints at the project root."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)

    with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=Path("/tmp/rtk")):
        result = runner.invoke(main, ["wrap", "goose", "--prepare-only"])

    assert result.exit_code == 0, result.output
    goosehints = tmp_path / ".goosehints"
    assert goosehints.exists()
    content = goosehints.read_text()
    assert wrap_mod._RTK_MARKER in content
    assert "RTK (Rust Token Killer)" in content


def test_wrap_goose_idempotent_no_duplicate_block(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-running prepare-only must not duplicate the .goosehints RTK block."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)

    with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=Path("/tmp/rtk")):
        runner.invoke(main, ["wrap", "goose", "--prepare-only"])
        runner.invoke(main, ["wrap", "goose", "--prepare-only"])

    content = (tmp_path / ".goosehints").read_text()
    assert content.count(wrap_mod._RTK_MARKER) == 1


def test_wrap_goose_missing_binary_errors_clearly(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the goose binary is missing the command must fail with a clear error."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)

    with patch.object(wrap_mod.shutil, "which", return_value=None):
        with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=Path("/tmp/rtk")):
            result = runner.invoke(main, ["wrap", "goose"])

    assert result.exit_code == 1
    assert "'goose' not found in PATH" in result.output


def test_wrap_goose_no_context_tool_skips_goosehints(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--no-context-tool must not create .goosehints."""
    monkeypatch.chdir(tmp_path)

    with patch.object(wrap_mod, "_ensure_rtk_binary") as ensure:
        result = runner.invoke(main, ["wrap", "goose", "--prepare-only", "--no-context-tool"])

    assert result.exit_code == 0, result.output
    assert not (tmp_path / ".goosehints").exists()
    ensure.assert_not_called()
