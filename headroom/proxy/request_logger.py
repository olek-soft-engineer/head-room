"""Request logger for the Headroom proxy.

Logs requests to an in-memory deque and optionally to a JSONL file.

Extracted from server.py for maintainability.

Phase G PR-G3 (P4-45): base64-encoded image payloads in the
``request_messages`` / ``response_content`` are redacted before
write to keep request logs small. Multi-MB base64 strings would
otherwise saturate the JSONL log and the in-memory deque.
"""

from __future__ import annotations

import json
import logging
import sys
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..memory.tracker import ComponentStats

from headroom.proxy.models import RequestLog

logger = logging.getLogger(__name__)

# Phase G PR-G3 — base64 redaction threshold (P4-45).
#
# Anthropic image blocks carry base64-encoded JPEGs/PNGs in
# ``source.data``; OpenAI's vision shape carries them in
# ``image_url.url`` as a ``data:image/...;base64,<payload>`` URL.
# The threshold gates "real image payload" against short base64
# strings (which can appear in arguments, signatures, etc.).
IMAGE_BASE64_REDACT_THRESHOLD_BYTES = 1024

# Phase G PR-G3 — replacement-marker format. Operators can grep the
# JSONL for ``<image:base64-redacted`` to count the redactions; the
# byte count keeps cost attribution honest even after redaction.
IMAGE_BASE64_REPLACEMENT_TEMPLATE = "<image:base64-redacted bytes={n}>"

# Constants for log redaction counter export (Prometheus). The Rust
# proxy owns the canonical metric; the Python side increments a
# best-effort module-level counter so tests and the in-process
# ``/stats`` endpoint can read back the redaction rate.
_redactions_total: int = 0


def redactions_total() -> int:
    """Return the running count of base64 redactions performed.

    Exposed for unit tests + the legacy Python ``/stats`` endpoint.
    The canonical observability surface is the Rust proxy's
    ``proxy_image_generation_call_log_redacted_total`` metric.
    """
    return _redactions_total


def _looks_like_base64_image(value: str) -> bool:
    """Heuristic: does ``value`` look like a base64-encoded image?

    Two patterns we recognise:

    * Raw base64 over the threshold (Anthropic ``source.data`` shape).
    * ``data:image/<subtype>;base64,<payload>`` data URLs (OpenAI
      vision shape). The ``;base64,`` substring is the load-bearing
      signal — any data URL with that segment over the threshold gets
      redacted, even if the MIME type isn't ``image/...`` (because
      the cost-of-logging is paid the same way).

    Returns ``False`` for short strings (under the threshold) and for
    any non-string value. Per realignment build-constraint "no
    regexes", we use prefix/substring checks instead of pattern
    matching.
    """
    if not isinstance(value, str):
        return False
    if len(value) < IMAGE_BASE64_REDACT_THRESHOLD_BYTES:
        return False
    if value.startswith("data:") and ";base64," in value[:64]:
        return True
    # Bare base64 payload: heuristic — over-threshold string with no
    # whitespace and a high alpha-num+/+= density. We sample the first
    # 256 bytes for speed (the full string can be megabytes).
    head = value[:256]
    base64_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=")
    matches = sum(1 for ch in head if ch in base64_chars)
    return matches / len(head) > 0.95


def _redact_value(value: Any) -> Any:
    """Recursively redact base64-image payloads in a JSON-ish value.

    Returns a new structure with any over-threshold base64 string
    replaced by the placeholder. Non-string, non-container values
    pass through unchanged.
    """
    global _redactions_total
    if isinstance(value, str):
        if _looks_like_base64_image(value):
            _redactions_total += 1
            return IMAGE_BASE64_REPLACEMENT_TEMPLATE.format(n=len(value))
        return value
    if isinstance(value, Mapping):
        return {k: _redact_value(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_redact_value(item) for item in value]
    return value


def redact_image_base64(payload: Any) -> Any:
    """Public entry point for base64-image redaction.

    Walks ``payload`` (a dict, list, or string) and replaces any
    over-threshold base64 string with a size-only placeholder.
    Idempotent — applying twice yields the same structure.
    """
    return _redact_value(payload)


class RequestLogger:
    """Log requests to JSONL file.

    Uses a deque with max 10,000 entries to prevent unbounded memory growth.
    Gracefully degrades to in-memory-only if the log file cannot be written
    (read-only filesystem, permissions error, etc.).
    """

    MAX_LOG_ENTRIES = 10_000

    def __init__(self, log_file: str | None = None, log_full_messages: bool = False):
        self.log_file = Path(log_file) if log_file else None
        self.log_full_messages = log_full_messages
        # Use deque with maxlen for automatic FIFO eviction
        self._logs: deque[RequestLog] = deque(maxlen=self.MAX_LOG_ENTRIES)

        if self.log_file:
            try:
                self.log_file.parent.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.warning(
                    "Cannot create log directory %s: %s — logging to memory only",
                    self.log_file.parent,
                    e,
                )
                self.log_file = None

    def log(self, entry: RequestLog):
        """Log a request. Oldest entries are automatically removed when limit reached.

        Phase G PR-G3 (P4-45): base64-encoded image payloads in
        ``request_messages`` / ``response_content`` are redacted
        before write. Redaction also applies to the in-memory deque
        so the ``/stats/recent_requests`` endpoint never serves a
        multi-MB image either.
        """
        # Redact image payloads in-place on the deque entry so memory
        # use stays bounded. We mutate the dataclass fields rather
        # than wrapping the entry to keep ``get_recent`` /
        # ``get_recent_with_messages`` unchanged.
        if entry.request_messages is not None:
            entry.request_messages = redact_image_base64(entry.request_messages)
        if entry.response_content is not None:
            entry.response_content = redact_image_base64(entry.response_content)

        self._logs.append(entry)

        if self.log_file:
            try:
                with open(self.log_file, "a") as f:
                    log_dict = asdict(entry)
                    if not self.log_full_messages:
                        log_dict.pop("request_messages", None)
                        log_dict.pop("response_content", None)
                    f.write(json.dumps(log_dict) + "\n")
            except OSError:
                pass  # Graceful degradation: memory-only logging continues

    def get_recent(self, n: int = 100) -> list[dict]:
        """Get recent log entries (without request_messages and response_content)."""
        # Convert deque to list for slicing (deque doesn't support slicing)
        entries = list(self._logs)[-n:]
        return [
            {
                k: v
                for k, v in asdict(e).items()
                if k not in ("request_messages", "response_content")
            }
            for e in entries
        ]

    def get_recent_with_messages(self, n: int = 20) -> list[dict]:
        """Get recent log entries including full request/response messages."""
        entries = list(self._logs)[-n:]
        return [asdict(e) for e in entries]

    def stats(self) -> dict:
        """Get logging statistics."""
        return {
            "total_logged": len(self._logs),
            "log_file": str(self.log_file) if self.log_file else None,
        }

    def get_memory_stats(self) -> ComponentStats:
        """Get memory statistics for the MemoryTracker.

        Returns:
            ComponentStats with current memory usage.
        """
        from ..memory.tracker import ComponentStats

        # Calculate size
        size_bytes = sys.getsizeof(self._logs)

        for log_entry in self._logs:
            size_bytes += sys.getsizeof(log_entry)
            # Add string fields
            if log_entry.request_id:
                size_bytes += len(log_entry.request_id)
            if log_entry.provider:
                size_bytes += len(log_entry.provider)
            if log_entry.model:
                size_bytes += len(log_entry.model)
            if log_entry.error:
                size_bytes += len(log_entry.error)
            # Messages and response can be large
            if log_entry.request_messages:
                size_bytes += sys.getsizeof(log_entry.request_messages)
            if log_entry.response_content:
                size_bytes += len(log_entry.response_content)

        return ComponentStats(
            name="request_logger",
            entry_count=len(self._logs),
            size_bytes=size_bytes,
            budget_bytes=None,
            hits=0,
            misses=0,
            evictions=0,
        )
