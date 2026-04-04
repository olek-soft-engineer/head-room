"""Tests for proxy token/cache mode normalization."""

from headroom.proxy.modes import (
    PROXY_MODE_CACHE,
    PROXY_MODE_TOKEN,
    is_cache_mode,
    is_token_mode,
    normalize_proxy_mode,
)


def test_proxy_mode_normalizes_canonical_values() -> None:
    assert normalize_proxy_mode("token") == PROXY_MODE_TOKEN
    assert normalize_proxy_mode("cache") == PROXY_MODE_CACHE


def test_proxy_mode_normalizes_legacy_aliases() -> None:
    assert normalize_proxy_mode("token_headroom") == PROXY_MODE_TOKEN
    assert normalize_proxy_mode("token_savings") == PROXY_MODE_TOKEN
    assert normalize_proxy_mode("cost_savings") == PROXY_MODE_CACHE
    assert normalize_proxy_mode("cache_mode") == PROXY_MODE_CACHE


def test_proxy_mode_invalid_falls_back_to_default() -> None:
    assert normalize_proxy_mode("wat", default=PROXY_MODE_CACHE) == PROXY_MODE_CACHE


def test_proxy_mode_predicates() -> None:
    assert is_token_mode("token_headroom") is True
    assert is_cache_mode("cost_savings") is True
