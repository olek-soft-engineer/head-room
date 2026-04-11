"""Model registry and shared ML model helpers."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    # LLM Registry
    "ModelRegistry",
    "ModelInfo",
    "get_model_info",
    "list_models",
    "register_model",
    # ML Model Registry
    "MLModelRegistry",
    "get_sentence_transformer",
    "get_siglip",
    "get_spacy",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # LLM registry
    "ModelRegistry": ("headroom.models.registry", "ModelRegistry"),
    "ModelInfo": ("headroom.models.registry", "ModelInfo"),
    "get_model_info": ("headroom.models.registry", "get_model_info"),
    "list_models": ("headroom.models.registry", "list_models"),
    "register_model": ("headroom.models.registry", "register_model"),
    # ML model registry
    "MLModelRegistry": ("headroom.models.ml_models", "MLModelRegistry"),
    "get_sentence_transformer": ("headroom.models.ml_models", "get_sentence_transformer"),
    "get_siglip": ("headroom.models.ml_models", "get_siglip"),
    "get_spacy": ("headroom.models.ml_models", "get_spacy"),
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
