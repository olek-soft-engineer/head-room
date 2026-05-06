from __future__ import annotations

from headroom.providers.codex.install import build_provider_section


def test_codex_provider_section_preserves_openai_oauth() -> None:
    section = build_provider_section(port=8787, name="OpenAI via Headroom proxy")

    assert 'name = "OpenAI via Headroom proxy"' in section
    assert 'base_url = "http://127.0.0.1:8787/v1"' in section
    assert "requires_openai_auth = true" in section
    assert "supports_websockets = true" in section
    assert 'env_key = "OPENAI_API_KEY"' not in section


def test_codex_provider_section_supports_custom_markers() -> None:
    section = build_provider_section(
        port=9100,
        name="Headroom init proxy",
        marker_start="# --- start ---",
        marker_end="# --- end ---",
    )

    assert section.startswith("# --- start ---\n")
    assert section.endswith("# --- end ---\n")
    assert 'base_url = "http://127.0.0.1:9100/v1"' in section
    assert 'env_key = "OPENAI_API_KEY"' not in section
