"""Tests for `headroom wrap continue` command (PR-G1, Phase G)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from headroom.cli import wrap as wrap_mod
from headroom.cli.main import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_inject_continue_rtk_systemmessage_new_file(tmp_path: Path) -> None:
    """Writing into a non-existent config.json creates parents + sets systemMessage."""
    config_file = tmp_path / ".continue" / "config.json"
    assert not config_file.exists()

    assert wrap_mod._inject_continue_rtk_systemmessage(config_file) is True

    data = json.loads(config_file.read_text())
    assert wrap_mod._RTK_MARKER in data["systemMessage"]


def test_inject_continue_rtk_systemmessage_preserves_existing_keys(tmp_path: Path) -> None:
    """Pre-existing keys like ``models`` are not touched."""
    config_file = tmp_path / ".continue" / "config.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text(json.dumps({"models": [{"title": "GPT-4o", "provider": "openai"}]}))

    wrap_mod._inject_continue_rtk_systemmessage(config_file)

    data = json.loads(config_file.read_text())
    assert data["models"] == [{"title": "GPT-4o", "provider": "openai"}]
    assert wrap_mod._RTK_MARKER in data["systemMessage"]


def test_inject_continue_rtk_systemmessage_appends_to_existing_message(
    tmp_path: Path,
) -> None:
    """Pre-existing systemMessage content is preserved; rtk block is appended."""
    config_file = tmp_path / ".continue" / "config.json"
    config_file.parent.mkdir(parents=True)
    existing_msg = "You are a helpful assistant."
    config_file.write_text(json.dumps({"systemMessage": existing_msg}))

    wrap_mod._inject_continue_rtk_systemmessage(config_file)

    data = json.loads(config_file.read_text())
    assert data["systemMessage"].startswith(existing_msg)
    assert wrap_mod._RTK_MARKER in data["systemMessage"]


def test_inject_continue_rtk_systemmessage_idempotent(tmp_path: Path) -> None:
    """Re-injection must not duplicate the marker."""
    config_file = tmp_path / ".continue" / "config.json"

    wrap_mod._inject_continue_rtk_systemmessage(config_file)
    wrap_mod._inject_continue_rtk_systemmessage(config_file)

    data = json.loads(config_file.read_text())
    assert data["systemMessage"].count(wrap_mod._RTK_MARKER) == 1


def test_inject_continue_rtk_systemmessage_refuses_invalid_json(
    tmp_path: Path,
) -> None:
    """Malformed JSON must be left untouched and the helper must return False."""
    config_file = tmp_path / ".continue" / "config.json"
    config_file.parent.mkdir(parents=True)
    malformed = '{ "models": [ this is not valid json'
    config_file.write_text(malformed)

    result = wrap_mod._inject_continue_rtk_systemmessage(config_file)

    assert result is False
    assert config_file.read_text() == malformed


def test_inject_continue_rtk_systemmessage_refuses_non_object_root(
    tmp_path: Path,
) -> None:
    """A JSON array at the root is not a valid Continue config; leave untouched."""
    config_file = tmp_path / ".continue" / "config.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("[]")

    result = wrap_mod._inject_continue_rtk_systemmessage(config_file)

    assert result is False
    assert config_file.read_text() == "[]"


def test_wrap_continue_prepare_only_injects_systemmessage(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`wrap continue --prepare-only` injects into ./.continue/config.json by default."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)

    with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=Path("/tmp/rtk")):
        result = runner.invoke(main, ["wrap", "continue", "--prepare-only"])

    assert result.exit_code == 0, result.output
    config_file = tmp_path / ".continue" / "config.json"
    assert config_file.exists()
    data = json.loads(config_file.read_text())
    assert wrap_mod._RTK_MARKER in data["systemMessage"]


def test_wrap_continue_respects_custom_config_path(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--config writes to the user-specified path, not the cwd default."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)
    custom_config = tmp_path / "custom" / "my-continue.json"

    with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=Path("/tmp/rtk")):
        result = runner.invoke(
            main,
            ["wrap", "continue", "--prepare-only", "--config", str(custom_config)],
        )

    assert result.exit_code == 0, result.output
    assert custom_config.exists()
    assert not (tmp_path / ".continue" / "config.json").exists()
    data = json.loads(custom_config.read_text())
    assert wrap_mod._RTK_MARKER in data["systemMessage"]
