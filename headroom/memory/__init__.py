"""Headroom Memory - Hierarchical memory system for AI applications.

This module provides a sophisticated memory system with:
- Hierarchical scoping (user -> session -> agent -> turn)
- Temporal versioning with supersession
- Vector and text search capabilities
- Pluggable storage backends via Protocol interfaces
- Zero-latency inline memory extraction (Letta-style)

Quick Start (One-liner with any LLM client):
    from openai import OpenAI
    from headroom.memory import with_memory

    client = with_memory(OpenAI(), user_id="alice")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "I prefer Python"}]
    )
    # Memory automatically extracted and stored!

Advanced Usage (Direct API):
    from headroom.memory import HierarchicalMemory, MemoryConfig, MemoryCategory

    memory = await HierarchicalMemory.create()
    await memory.add(
        content="User prefers Python over JavaScript",
        user_id="alice",
        category=MemoryCategory.PREFERENCE,
    )
    results = await memory.search("programming preferences", user_id="alice")

Configuration:
    from headroom.memory import MemoryConfig, EmbedderBackend

    config = MemoryConfig(
        embedder_backend=EmbedderBackend.OPENAI,
        openai_api_key="sk-...",
    )
    memory = await HierarchicalMemory.create(config)
"""

# Configuration
from headroom.memory.config import (
    EmbedderBackend,
    MemoryConfig,
    StoreBackend,
    TextBackend,
    VectorBackend,
)

# Core orchestrator
from headroom.memory.core import HierarchicalMemory

# Factory
from headroom.memory.factory import create_memory_system

# Data models
from headroom.memory.models import Memory, MemoryCategory, ScopeLevel

# Protocol interfaces (ports)
from headroom.memory.ports import (
    Embedder,
    MemoryCache,
    MemoryFilter,
    MemoryStore,
    TextFilter,
    TextIndex,
    TextSearchResult,
    VectorFilter,
    VectorIndex,
    VectorSearchResult,
)

# Wrapper for LLM clients (main user-facing API)
from headroom.memory.wrapper import MemoryWrapper, with_memory

__all__ = [
    # Main user-facing API
    "with_memory",
    "MemoryWrapper",
    # Core orchestrator
    "HierarchicalMemory",
    # Data models
    "Memory",
    "MemoryCategory",
    "ScopeLevel",
    # Protocol interfaces (ports)
    "MemoryStore",
    "VectorIndex",
    "TextIndex",
    "Embedder",
    "MemoryCache",
    # Filter dataclasses
    "MemoryFilter",
    "VectorFilter",
    "TextFilter",
    # Search result dataclasses
    "VectorSearchResult",
    "TextSearchResult",
    # Configuration
    "MemoryConfig",
    "StoreBackend",
    "VectorBackend",
    "TextBackend",
    "EmbedderBackend",
    # Factory
    "create_memory_system",
]
