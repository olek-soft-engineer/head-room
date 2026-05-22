"""Tests for `headroom wrap cline` command (PR-G1, Phase G)."""

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


def test_wrap_cline_prepare_only_injects_rtk_into_clinerules(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`wrap cline --prepare-only` writes RTK guidance to .clinerules at cwd."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)

    with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=Path("/tmp/rtk")):
        result = runner.invoke(main, ["wrap", "cline", "--prepare-only"])

    assert result.exit_code == 0, result.output
    clinerules = tmp_path / ".clinerules"
    assert clinerules.exists(), ".clinerules should be created"
    content = clinerules.read_text()
    assert wrap_mod._RTK_MARKER in content
    assert "RTK (Rust Token Killer)" in content


def test_wrap_cline_prepare_only_idempotent_no_duplicate_block(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Running wrap cline twice must not duplicate the RTK block."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)

    with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=Path("/tmp/rtk")):
        runner.invoke(main, ["wrap", "cline", "--prepare-only"])
        runner.invoke(main, ["wrap", "cline", "--prepare-only"])

    clinerules = tmp_path / ".clinerules"
    content = clinerules.read_text()
    assert content.count(wrap_mod._RTK_MARKER) == 1


def test_wrap_cline_no_context_tool_does_not_create_clinerules(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--no-context-tool must not create .clinerules."""
    monkeypatch.chdir(tmp_path)

    # Patch RTK to fail if accidentally called.
    with patch.object(wrap_mod, "_ensure_rtk_binary") as ensure:
        result = runner.invoke(main, ["wrap", "cline", "--prepare-only", "--no-context-tool"])

    assert result.exit_code == 0, result.output
    assert not (tmp_path / ".clinerules").exists()
    ensure.assert_not_called()


def test_wrap_cline_preserves_existing_clinerules_content(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-existing .clinerules content must be preserved when RTK is appended."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)
    clinerules = tmp_path / ".clinerules"
    original = "# Project conventions\n\nAlways use Python 3.12.\n"
    clinerules.write_text(original)

    with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=Path("/tmp/rtk")):
        result = runner.invoke(main, ["wrap", "cline", "--prepare-only"])

    assert result.exit_code == 0, result.output
    content = clinerules.read_text()
    assert "Always use Python 3.12." in content
    assert wrap_mod._RTK_MARKER in content
