from __future__ import annotations

from pathlib import Path

from headroom.install.models import DeploymentManifest, SupervisorKind
from headroom.install.supervisors import (
    _linux_service_unit,
    _linux_task_spec,
    _macos_launchd_plist,
    _render_windows_runner,
)


def _manifest(
    *, profile: str = "default", scope: str = "user", supervisor: str = "service"
) -> DeploymentManifest:
    return DeploymentManifest(
        profile=profile,
        preset="persistent-service",
        runtime_kind="python",
        supervisor_kind=supervisor,
        scope=scope,
        provider_mode="manual",
        targets=[],
        port=8787,
        host="127.0.0.1",
        backend="anthropic",
        service_name=f"headroom-{profile}",
    )


def test_linux_service_unit_uses_user_systemd_path(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    manifest = _manifest()

    unit_path, content = _linux_service_unit(manifest, tmp_path / "run-headroom.sh")

    assert unit_path == tmp_path / ".config" / "systemd" / "user" / "headroom-default.service"
    assert "ExecStart=" + str(tmp_path / "run-headroom.sh") in content
    assert "Restart=on-failure" in content


def test_linux_task_spec_for_user_scope_includes_crontab_markers(tmp_path: Path) -> None:
    manifest = _manifest(profile="smoke", supervisor=SupervisorKind.TASK.value)

    cron_path, content = _linux_task_spec(manifest, tmp_path / "ensure-headroom.sh")

    assert cron_path is None
    assert "# >>> headroom smoke >>>" in content
    assert "# <<< headroom smoke <<<" in content
    assert "@reboot" in content
    assert "*/5 * * * *" in content


def test_macos_launchd_plist_switches_between_keepalive_and_interval(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    service_manifest = _manifest(supervisor=SupervisorKind.SERVICE.value)
    service_path, service_content = _macos_launchd_plist(
        service_manifest, tmp_path / "run-headroom.sh"
    )
    assert service_path == tmp_path / "Library" / "LaunchAgents" / "com.headroom.default.plist"
    assert "<key>KeepAlive</key>" in service_content
    assert "<key>StartInterval</key>" not in service_content

    task_manifest = _manifest(profile="tasky", supervisor=SupervisorKind.TASK.value)
    task_path, task_content = _macos_launchd_plist(
        task_manifest, tmp_path / "ensure-headroom.sh", interval=300
    )
    assert task_path == tmp_path / "Library" / "LaunchAgents" / "com.headroom.tasky.plist"
    assert "<key>StartInterval</key>" in task_content
    assert "<integer>300</integer>" in task_content


def test_render_windows_runner_writes_ps1_and_cmd_wrappers(tmp_path: Path) -> None:
    ps1_path = tmp_path / "run-headroom.ps1"
    cmd_path = tmp_path / "run-headroom.cmd"

    records = _render_windows_runner(
        ps1_path,
        cmd_path,
        ["C:\\Program Files\\Python\\python.exe", "headroom", "install", "agent", "run"],
    )

    assert [record.path for record in records] == [str(ps1_path), str(cmd_path)]
    ps1_content = ps1_path.read_text(encoding="utf-8")
    cmd_content = cmd_path.read_text(encoding="utf-8")
    assert '& "C:\\Program Files\\Python\\python.exe" headroom install agent run' in ps1_content
    assert (
        'powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run-headroom.ps1" %*'
        in cmd_content
    )
