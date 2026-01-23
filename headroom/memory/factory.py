"""Factory module for creating memory system components.

Provides a unified factory function that creates all memory system components
from a single configuration object, ensuring consistent initialization
and proper wiring between components.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from headroom.memory.config import (
    EmbedderBackend,
    MemoryConfig,
    StoreBackend,
    TextBackend,
    VectorBackend,
)

if TYPE_CHECKING:
    from headroom.memory.ports import Embedder, MemoryCache, MemoryStore, TextIndex, VectorIndex


async def create_memory_system(
    config: MemoryConfig | None = None,
) -> tuple[MemoryStore, VectorIndex, TextIndex, Embedder, MemoryCache | None]:
    """Create a complete memory system from configuration.

    This factory function creates and initializes all memory system components
    based on the provided configuration. Components are created in dependency
    order to ensure proper initialization.

    Args:
        config: Memory system configuration. If None, uses default configuration.

    Returns:
        A tuple of (store, vector_index, text_index, embedder, cache) where:
        - store: The memory persistence backend
        - vector_index: The vector similarity search index
        - text_index: The full-text search index
        - embedder: The text embedding generator
        - cache: The memory cache (or None if caching is disabled)

    Raises:
        ValueError: If an unknown backend type is specified in the config.

    Example:
        config = MemoryConfig(
            embedder_backend=EmbedderBackend.LOCAL,
            cache_max_size=2000,
        )
        store, vector, text, embedder, cache = await create_memory_system(config)
    """
    config = config or MemoryConfig()

    # Create store
    store = _create_store(config)

    # Create embedder (needed by vector index for text queries)
    embedder = _create_embedder(config)

    # Create vector index
    vector_index = _create_vector_index(config)

    # Create text index
    text_index = _create_text_index(config)

    # Create cache (optional)
    cache = _create_cache(config) if config.cache_enabled else None

    return store, vector_index, text_index, embedder, cache


def _create_store(config: MemoryConfig) -> MemoryStore:
    """Create a memory store backend.

    Args:
        config: Memory system configuration.

    Returns:
        A MemoryStore implementation based on config.store_backend.

    Raises:
        ValueError: If the store backend is not supported.
    """
    if config.store_backend == StoreBackend.SQLITE:
        from headroom.memory.adapters.sqlite import SQLiteMemoryStore

        return SQLiteMemoryStore(config.db_path)

    raise ValueError(f"Unknown store backend: {config.store_backend}")


def _create_embedder(config: MemoryConfig) -> Embedder:
    """Create an embedder backend.

    Args:
        config: Memory system configuration.

    Returns:
        An Embedder implementation based on config.embedder_backend.

    Raises:
        ValueError: If the embedder backend is not supported.
    """
    if config.embedder_backend == EmbedderBackend.LOCAL:
        from headroom.memory.adapters.embedders import LocalEmbedder

        return LocalEmbedder(model_name=config.embedder_model)

    if config.embedder_backend == EmbedderBackend.OPENAI:
        from headroom.memory.adapters.embedders import OpenAIEmbedder

        if not config.openai_api_key:
            raise ValueError("openai_api_key is required for OpenAI embedder")
        return OpenAIEmbedder(
            api_key=config.openai_api_key,
            model_name=config.embedder_model,
        )

    if config.embedder_backend == EmbedderBackend.OLLAMA:
        from headroom.memory.adapters.embedders import OllamaEmbedder

        return OllamaEmbedder(
            base_url=config.ollama_base_url,
            model_name=config.embedder_model,
        )

    raise ValueError(f"Unknown embedder backend: {config.embedder_backend}")


def _create_vector_index(config: MemoryConfig) -> VectorIndex:
    """Create a vector index backend.

    Args:
        config: Memory system configuration.

    Returns:
        A VectorIndex implementation based on config.vector_backend.

    Raises:
        ValueError: If the vector backend is not supported.
    """
    if config.vector_backend == VectorBackend.HNSW:
        from headroom.memory.adapters.hnsw import HNSWVectorIndex

        return HNSWVectorIndex(
            dimension=config.vector_dimension,
            ef_construction=config.hnsw_ef_construction,
            m=config.hnsw_m,
            ef_search=config.hnsw_ef_search,
        )

    raise ValueError(f"Unknown vector backend: {config.vector_backend}")


def _create_text_index(config: MemoryConfig) -> TextIndex:
    """Create a text index backend.

    Args:
        config: Memory system configuration.

    Returns:
        A TextIndex implementation based on config.text_backend.

    Raises:
        ValueError: If the text backend is not supported.
    """
    if config.text_backend == TextBackend.FTS5:
        from headroom.memory.adapters.fts5 import FTS5TextIndex

        # FTS5TextIndex has a compatible interface but different method signatures
        return FTS5TextIndex(db_path=config.db_path)  # type: ignore[return-value]

    raise ValueError(f"Unknown text backend: {config.text_backend}")


def _create_cache(config: MemoryConfig) -> MemoryCache:
    """Create a memory cache.

    Args:
        config: Memory system configuration.

    Returns:
        A MemoryCache implementation.
    """
    from headroom.memory.adapters.cache import LRUMemoryCache

    # LRUMemoryCache implements MemoryCache protocol
    return LRUMemoryCache(max_size=config.cache_max_size)  # type: ignore[return-value]
