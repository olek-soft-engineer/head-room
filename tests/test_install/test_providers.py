from __future__ import annotations

import json
import os
from pathlib import Path

import click
import pytest

from headroom.install.models import DeploymentManifest, ManagedMutation
from headroom.install.providers import _apply_windows_env_scope, _remove_windows_env_scope
from headroom.providers.claude.install import apply_provider_scope as apply_claude_provider_scope
from headroom.providers.claude.install import build_install_env as build_claude_install_env
from headroom.providers.claude.install import revert_provider_scope as revert_claude_provider_scope
from headroom.providers.codex.install import apply_provider_scope as apply_codex_provider_scope
from headroom.providers.codex.install import build_install_env as build_codex_install_env
from headroom.providers.codex.install import revert_provider_scope as revert_codex_provider_scope
from headroom.providers.copilot.install import build_install_env as build_copilot_install_env


def _manifest(tmp_path: Path) -> DeploymentManifest:
    return DeploymentManifest(
        profile="default",
        preset="persistent-service",
        runtime_kind="python",
        supervisor_kind="service",
        scope="provider",
        provider_mode="manual",
        targets=["claude", "codex"],
        port=8787,
        host="127.0.0.1",
        backend="anthropic",
        memory_db_path=str(tmp_path / "memory.db"),
        tool_envs={
            "claude": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8787"},
            "codex": {"OPENAI_BASE_URL": "http://127.0.0.1:8787/v1"},
        },
    )


def test_apply_and_revert_claude_provider_scope(monkeypatch, tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps({"env": {"ANTHROPIC_API_KEY": "keep", "ANTHROPIC_BASE_URL": "https://old"}})
    )
    monkeypatch.setattr(
        "headroom.providers.claude.install.claude_settings_path", lambda: settings_path
    )
    manifest = _manifest(tmp_path)

    mutation = apply_claude_provider_scope(manifest)
    payload = json.loads(settings_path.read_text())
    assert payload["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8787"
    assert payload["env"]["ANTHROPIC_API_KEY"] == "keep"

    assert mutation is not None
    revert_claude_provider_scope(mutation, manifest)
    reverted = json.loads(settings_path.read_text())
    assert reverted["env"]["ANTHROPIC_BASE_URL"] == "https://old"
    assert reverted["env"]["ANTHROPIC_API_KEY"] == "keep"


def test_apply_and_revert_codex_provider_scope(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text('model = "gpt-4o"\n')
    monkeypatch.setattr("headroom.providers.codex.install.codex_config_path", lambda: config_path)
    manifest = _manifest(tmp_path)

    mutation = apply_codex_provider_scope(manifest)
    content = config_path.read_text()
    assert 'model_provider = "headroom"' in content
    assert 'base_url = "http://127.0.0.1:8787/v1"' in content
    assert 'env_key = "OPENAI_API_KEY"' in content
    assert "requires_openai_auth" not in content

    assert mutation is not None
    revert_codex_provider_scope(mutation, manifest)
    reverted = config_path.read_text()
    assert 'model_provider = "headroom"' not in reverted
    assert reverted.strip() == 'model = "gpt-4o"'


def test_codex_build_install_env_returns_proxy_base_url() -> None:
    env = build_codex_install_env(port=5566, backend="ignored")

    assert env == {"OPENAI_BASE_URL": "http://127.0.0.1:5566/v1"}


def test_apply_codex_provider_scope_skips_non_provider_scope(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    monkeypatch.setattr("headroom.providers.codex.install.codex_config_path", lambda: config_path)
    manifest = _manifest(tmp_path)
    manifest.scope = "user"

    mutation = apply_codex_provider_scope(manifest)

    assert mutation is None
    assert not config_path.exists()


def test_apply_codex_provider_scope_replaces_existing_managed_block(
    monkeypatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        'model = "gpt-4o"\n\n'
        "# --- Headroom persistent provider ---\n"
        'model_provider = "headroom"\n\n'
        "[model_providers.headroom]\n"
        'name = "Headroom persistent proxy"\n'
        'base_url = "http://127.0.0.1:1111/v1"\n'
        "requires_openai_auth = true\n"
        "supports_websockets = true\n"
        "# --- end Headroom persistent provider ---\n"
    )
    monkeypatch.setattr("headroom.providers.codex.install.codex_config_path", lambda: config_path)
    manifest = _manifest(tmp_path)
    manifest.port = 9999

    apply_codex_provider_scope(manifest)

    content = config_path.read_text()
    assert content.count("# --- Headroom persistent provider ---") == 1
    assert 'base_url = "http://127.0.0.1:9999/v1"' in content
    assert 'base_url = "http://127.0.0.1:1111/v1"' not in content
    # Bug 3 (#406): the replacement block must NOT carry requires_openai_auth.
    assert "requires_openai_auth" not in content


def test_apply_codex_provider_scope_creates_new_config_when_missing(
    monkeypatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "nested" / "config.toml"
    monkeypatch.setattr("headroom.providers.codex.install.codex_config_path", lambda: config_path)
    manifest = _manifest(tmp_path)

    mutation = apply_codex_provider_scope(manifest)

    assert mutation is not None
    assert 'base_url = "http://127.0.0.1:8787/v1"' in config_path.read_text()


def test_revert_codex_provider_scope_ignores_missing_path_and_file(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)

    revert_codex_provider_scope(
        ManagedMutation(target="codex", kind="toml-block"),
        manifest,
    )
    revert_codex_provider_scope(
        ManagedMutation(
            target="codex",
            kind="toml-block",
            path=str(tmp_path / "missing.toml"),
        ),
        manifest,
    )


def test_revert_codex_provider_scope_ignores_files_without_managed_block(
    monkeypatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text('model = "gpt-4o"\n')
    monkeypatch.setattr("headroom.providers.codex.install.codex_config_path", lambda: config_path)
    manifest = _manifest(tmp_path)
    mutation = ManagedMutation(target="codex", kind="toml-block", path=str(config_path))

    revert_codex_provider_scope(mutation, manifest)

    assert config_path.read_text() == 'model = "gpt-4o"\n'


def test_apply_openclaw_provider_scope_uses_manifest_port(monkeypatch, tmp_path: Path) -> None:
    recorded: list[list[str]] = []
    monkeypatch.setattr("headroom.providers.openclaw.install.shutil_which", lambda name: "openclaw")
    monkeypatch.setattr(
        "headroom.providers.openclaw.install.resolve_headroom_command",
        lambda: ["headroom"],
    )
    monkeypatch.setattr(
        "headroom.providers.openclaw.install._invoke_openclaw",
        lambda command: recorded.append(command),
    )
    monkeypatch.setattr(
        "headroom.providers.openclaw.install.openclaw_config_path",
        lambda: tmp_path / "openclaw.json",
    )
    manifest = _manifest(tmp_path)
    manifest.port = 9999

    from headroom.providers.openclaw.install import (
        apply_provider_scope as apply_openclaw_provider_scope,
    )

    apply_openclaw_provider_scope(manifest)

    assert recorded == [["headroom", "wrap", "openclaw", "--no-auto-start", "--proxy-port", "9999"]]


def test_openclaw_apply_provider_scope_requires_installed_binary(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr("headroom.providers.openclaw.install.shutil_which", lambda name: None)

    with pytest.raises(click.ClickException, match="openclaw not found"):
        from headroom.providers.openclaw.install import (
            apply_provider_scope as apply_openclaw_provider_scope,
        )

        apply_openclaw_provider_scope(_manifest(tmp_path))


def test_openclaw_helper_wrappers_delegate_to_stdlib(monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda name: f"/fake/{name}")
    recorded: list[tuple[list[str], bool]] = []

    def fake_run(command: list[str], check: bool) -> None:
        recorded.append((command, check))

    monkeypatch.setattr("subprocess.run", fake_run)

    from headroom.providers.openclaw.install import _invoke_openclaw, shutil_which

    assert shutil_which("openclaw") == "/fake/openclaw"
    _invoke_openclaw(["headroom", "wrap", "openclaw"])

    assert recorded == [(["headroom", "wrap", "openclaw"], True)]


def test_openclaw_revert_provider_scope_skips_without_binary(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("headroom.providers.openclaw.install.shutil_which", lambda name: None)
    called = False

    def fail_if_called(command: list[str]) -> None:
        nonlocal called
    // ... 309 lines omitted
    from headroom.providers.openclaw.install import (
    // ... 308 lines omitted
def test_openclaw_revert_provider_scope_invokes_unwrap(monkeypatch, tmp_path: Path) -> None:
    // ... 307 lines omitted
    from headroom.providers.openclaw.install import (
    // ... 306 lines omitted
def test_windows_env_scope_restores_previous_values(monkeypatch, tmp_path: Path) -> None:
    // ... 305 lines omitted
    }
    // ... 304 lines omitted
    class Result:
        def __init__(self, stdout: str = "") -> None:
    // ... 302 lines omitted
    def fake_run(command: list[str], **kwargs):
    // ... 301 lines omitted
def test_remove_windows_env_scope_requires_name_and_scope() -> None:
    // ... 300 lines omitted
def test_apply_mutations_runs_openclaw_for_user_scope(monkeypatch, tmp_path: Path) -> None:
    // ... 299 lines omitted
    from headroom.install.providers import apply_mutations
    // ... 298 lines omitted
def test_claude_build_install_env_returns_proxy_base_url() -> None:
    // ... 297 lines omitted
def test_copilot_build_install_env_uses_provider_type_specific_proxy_urls() -> None:
    // ... 296 lines omitted
    }
    // ... 295 lines omitted
    }
    // ... 294 lines omitted
def test_apply_claude_provider_scope_skips_non_provider_scope(monkeypatch, tmp_path: Path) -> None:
    // ... 293 lines omitted
def test_revert_claude_provider_scope_removes_new_values_from_non_mapping_env(
    // ... 292 lines omitted
def test_apply_claude_provider_scope_creates_settings_when_missing(
    // ... 291 lines omitted
    }
    // ... 290 lines omitted
def test_revert_claude_provider_scope_ignores_missing_mutation_path(tmp_path: Path) -> None:
    // ... 289 lines omitted
def test_revert_claude_provider_scope_ignores_missing_settings_file(tmp_path: Path) -> None:
    // ... 288 lines omitted
def test_headroom_provider_block_never_sets_requires_openai_auth(
    // ... 287 lines omitted
def test_inject_codex_provider_config_does_not_write_openai_base_url(
    // ... 286 lines omitted
    from headroom.cli import wrap as wrap_mod
// ... 285 more lines (total: 544)