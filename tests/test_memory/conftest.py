"""Test fixtures for Headroom Memory."""

from __future__ import annotations

# CRITICAL: Must be set before ANY imports that could trigger sentence_transformers
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
