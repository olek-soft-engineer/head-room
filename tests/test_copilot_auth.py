from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from headroom import copilot_auth


def test_read_cached_oauth_token_prefers_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_TOKEN", "gho-env")
    assert copilot_auth.read_cached_oauth_token() == "gho-env"


def test_read_cached_oauth_token_falls_back_to_gh_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_COPILOT_GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_COPILOT_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(copilot_auth, "_read_windows_copilot_cli_oauth_token", lambda: None)
    monkeypatch.setattr(copilot_auth, "_read_gh_cli_oauth_token", lambda: "gho-gh-cli")

    assert copilot_auth.read_cached_oauth_token() == "gho-gh-cli"


def test_read_cached_oauth_token_prefers_copilot_cli_windows_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_COPILOT_GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_COPILOT_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(
        copilot_auth, "_read_windows_copilot_cli_oauth_token", lambda: "gho-copilot"
    )
    monkeypatch.setattr(copilot_auth, "_read_gh_cli_oauth_token", lambda: "gho-gh-cli")

    assert copilot_auth.read_cached_oauth_token() == "gho-copilot"


def test_read_cached_oauth_token_reads_hosts_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hosts = tmp_path / "hosts.json"
    hosts.write_text(
        json.dumps(
            {
                "github.com": {
                    "oauth_token": "gho-file",
                    "expires_at": "2999-01-01T00:00:00Z",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("GITHUB_COPILOT_TOKEN", raising=False)
    monkeypatch.setenv("GITHUB_COPILOT_TOKEN_FILE", str(hosts))
    monkeypatch.setattr(copilot_auth, "_read_windows_copilot_cli_oauth_token", lambda: None)
    monkeypatch.setattr(copilot_auth, "_read_gh_cli_oauth_token", lambda: None)

    assert copilot_auth.read_cached_oauth_token() == "gho-file"


def test_read_cached_oauth_token_skips_expired_entries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hosts = tmp_path / "hosts.json"
    hosts.write_text(
        json.dumps({"github.com": {"oauthToken": "gho-old", "expiresAt": 1}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("GITHUB_COPILOT_TOKEN_FILE", str(hosts))
    monkeypatch.setattr(copilot_auth, "_read_windows_copilot_cli_oauth_token", lambda: None)
    monkeypatch.setattr(copilot_auth, "_read_gh_cli_oauth_token", lambda: None)

    assert copilot_auth.read_cached_oauth_token() is None


def test_read_gh_cli_oauth_token_uses_hostname(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    class CompletedProcess:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = "gho-gh-cli\n"

    def fake_run(*args: object, **kwargs: object) -> CompletedProcess:
        calls.append(list(args[0]))
        assert kwargs["capture_output"] is True
        assert kwargs["check"] is False
        return CompletedProcess()

    monkeypatch.setenv("GITHUB_COPILOT_HOST", "example.ghe.com")
    monkeypatch.setattr(copilot_auth.subprocess, "run", fake_run)

    assert copilot_auth._read_gh_cli_oauth_token() == "gho-gh-cli"
    assert calls == [["gh", "auth", "token", "--hostname", "example.ghe.com"]]


def test_resolve_client_bearer_token_prefers_api_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_API_TOKEN", "copilot-api")
    monkeypatch.setenv("GITHUB_COPILOT_TOKEN", "gho-oauth")

    assert copilot_auth.resolve_client_bearer_token() == "copilot-api"


def test_is_copilot_api_url_matches_expected_hosts() -> None:
    assert copilot_auth.is_copilot_api_url("https://api.githubcopilot.com/v1/chat/completions")
    assert copilot_auth.is_copilot_api_url("wss://api.githubcopilot.com/v1/responses")
    assert not copilot_auth.is_copilot_api_url("https://api.openai.com/v1/chat/completions")


def test_build_copilot_upstream_url_strips_v1_only_for_copilot_hosts() -> None:
    assert (
        copilot_auth.build_copilot_upstream_url(
            "https://api.githubcopilot.com",
            "/v1/chat/completions",
        )
        == "https://api.githubcopilot.com/chat/completions"
    )
    assert (
        copilot_auth.build_copilot_upstream_url(
            "https://api.openai.com",
            "/v1/chat/completions",
        )
        == "https://api.openai.com/v1/chat/completions"
    )


def test_apply_copilot_api_auth_replaces_authorization(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_get_api_token() -> copilot_auth.CopilotAPIToken:
        return copilot_auth.CopilotAPIToken(
            token="copilot-session",
            expires_at=time.time() + 3600,
            api_url=copilot_auth.DEFAULT_API_URL,
        )

    monkeypatch.setattr(
        copilot_auth.get_copilot_token_provider(),
        "get_api_token",
        fake_get_api_token,
    )

    headers = asyncio.run(
        copilot_auth.apply_copilot_api_auth(
            {"authorization": "Bearer downstream-token"},
            url="https://api.githubcopilot.com/v1/chat/completions",
        )
    )

    assert headers["Authorization"] == "Bearer copilot-session"
    assert "authorization" not in headers


def test_token_provider_reuses_oauth_token_without_exchange(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_TOKEN", "gho-oauth")

    provider = copilot_auth.CopilotTokenProvider()
    calls = {"count": 0}

    def fake_exchange(headers: dict[str, str]) -> dict[str, object]:
        calls["count"] += 1
        return {
            "token": "copilot-api",
            "expires_at": int(time.time()) + 3600,
            "refresh_in": 1200,
            "endpoints": {"api": "https://api.githubcopilot.com"},
            "sku": "copilot_individual",
        }

    monkeypatch.setattr(provider, "_exchange_token_sync", staticmethod(fake_exchange))

    first = asyncio.run(provider.get_api_token())
    second = asyncio.run(provider.get_api_token())

    assert first.token == "gho-oauth"
    assert second.token == "gho-oauth"
    assert calls["count"] == 0


def test_token_provider_can_exchange_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_TOKEN", "gho-oauth")
    monkeypatch.setenv("GITHUB_COPILOT_USE_TOKEN_EXCHANGE", "true")

    provider = copilot_auth.CopilotTokenProvider()
    calls = {"count": 0}

    def fake_exchange(headers: dict[str, str]) -> dict[str, object]:
        calls["count"] += 1
        return {
            "token": "copilot-api",
            "expires_at": int(time.time()) + 3600,
            "refresh_in": 1200,
            "endpoints": {"api": "https://api.githubcopilot.com"},
            "sku": "copilot_individual",
        }

    monkeypatch.setattr(provider, "_exchange_token_sync", staticmethod(fake_exchange))

    first = asyncio.run(provider.get_api_token())
    second = asyncio.run(provider.get_api_token())

    assert first.token == "copilot-api"
    assert second.token == "copilot-api"
    assert calls["count"] == 1
