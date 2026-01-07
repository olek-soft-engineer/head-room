"""Model registry and capabilities database.

Provides a centralized registry of LLM models with their capabilities,
context limits, pricing, and provider information.

Usage:
    from headroom.models import ModelRegistry, get_model_info

    # Get info about a model
    info = get_model_info("gpt-4o")
    print(f"Context: {info.context_window}")
    print(f"Provider: {info.provider}")

    # List all models from a provider
    models = ModelRegistry.list_models(provider="openai")

    # Register a custom model
    ModelRegistry.register(
        "my-custom-model",
        provider="custom",
        context_window=32000,
    )
"""

from .registry import (
    ModelInfo,
    ModelRegistry,
    get_model_info,
    list_models,
    register_model,
)

__all__ = [
    "ModelRegistry",
    "ModelInfo",
    "get_model_info",
    "list_models",
    "register_model",
]
