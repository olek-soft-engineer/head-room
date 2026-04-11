"""Headroom cache optimization exports."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    # Base types
    "BaseCacheOptimizer",
    "CacheBreakpoint",
    "CacheConfig",
    "CacheMetrics",
    "CacheOptimizer",
    "CacheResult",
    "CacheStrategy",
    "OptimizationContext",
    # Dynamic content detection
    "DetectorConfig",
    "DynamicCategory",
    "DynamicContentDetector",
    "DynamicSpan",
    "detect_dynamic_content",
    # Registry
    "CacheOptimizerRegistry",
    # Provider implementations
    "AnthropicCacheOptimizer",
    "OpenAICacheOptimizer",
    "GoogleCacheOptimizer",
    # Semantic caching
    "SemanticCacheLayer",
    "SemanticCache",
    # Compression cache (token headroom mode)
    "CompressionCache",
    # Prefix cache tracking
    "PrefixCacheTracker",
    "PrefixFreezeConfig",
    "FreezeStats",
    "SessionTrackerStore",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # Base types
    "BaseCacheOptimizer": ("headroom.cache.base", "BaseCacheOptimizer"),
    "CacheBreakpoint": ("headroom.cache.base", "CacheBreakpoint"),
    "CacheConfig": ("headroom.cache.base", "CacheConfig"),
    "CacheMetrics": ("headroom.cache.base", "CacheMetrics"),
    "CacheOptimizer": ("headroom.cache.base", "CacheOptimizer"),
    "CacheResult": ("headroom.cache.base", "CacheResult"),
    "CacheStrategy": ("headroom.cache.base", "CacheStrategy"),
    "OptimizationContext": ("headroom.cache.base", "OptimizationContext"),
    # Dynamic content detection
    "DetectorConfig": ("headroom.cache.dynamic_detector", "DetectorConfig"),
    "DynamicCategory": ("headroom.cache.dynamic_detector", "DynamicCategory"),
    "DynamicContentDetector": ("headroom.cache.dynamic_detector", "DynamicContentDetector"),
    "DynamicSpan": ("headroom.cache.dynamic_detector", "DynamicSpan"),
    "detect_dynamic_content": ("headroom.cache.dynamic_detector", "detect_dynamic_content"),
    # Registry
    "CacheOptimizerRegistry": ("headroom.cache.registry", "CacheOptimizerRegistry"),
    # Provider implementations
    "AnthropicCacheOptimizer": ("headroom.cache.anthropic", "AnthropicCacheOptimizer"),
    "OpenAICacheOptimizer": ("headroom.cache.openai", "OpenAICacheOptimizer"),
    "GoogleCacheOptimizer": ("headroom.cache.google", "GoogleCacheOptimizer"),
    # Semantic caching
    "SemanticCacheLayer": ("headroom.cache.semantic", "SemanticCacheLayer"),
    "SemanticCache": ("headroom.cache.semantic", "SemanticCache"),
    # Compression cache
    "CompressionCache": ("headroom.cache.compression_cache", "CompressionCache"),
    # Prefix cache tracking
    "PrefixCacheTracker": ("headroom.cache.prefix_tracker", "PrefixCacheTracker"),
    "PrefixFreezeConfig": ("headroom.cache.prefix_tracker", "PrefixFreezeConfig"),
    "FreezeStats": ("headroom.cache.prefix_tracker", "FreezeStats"),
    "SessionTrackerStore": ("headroom.cache.prefix_tracker", "SessionTrackerStore"),
}


def __getattr__(name: str) -> object:
    if name == "__path__":
        raise AttributeError(name)

    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
