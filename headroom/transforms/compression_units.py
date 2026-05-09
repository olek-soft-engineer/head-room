"""Provider-neutral compression units.

Provider adapters own request-envelope details and cache/live-zone decisions.
They should extract only safe, mutable text ranges into ``CompressionUnit``
objects, ask ContentRouter to compress each unit, then splice accepted
replacements back into their native request shape.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import Protocol

from .content_router import CompressionStrategy, ContentRouter, RouterCompressionResult


class TokenCounterLike(Protocol):
    def count_text(self, text: str) -> int: ...


@dataclass(frozen=True)
class CompressionUnit:
    """One provider-extracted, cache-safe text slot."""

    text: str
    provider: str
    endpoint: str
    role: str
    item_type: str
    cache_zone: str = "live"
    mutable: bool = True
    context: str = ""
    question: str | None = None
    bias: float = 1.0
    min_bytes: int = 512
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class UnitCompressionResult:
    original: str
    compressed: str
    modified: bool
    tokens_before: int
    tokens_after: int
    tokens_saved: int
    transforms_applied: list[str]
    strategy: str
    reason: str | None = None
    router_result: RouterCompressionResult | None = None


@dataclass(frozen=True)
class RoutedCompressionUnit:
    """A unit paired with its provider-owned slot reference."""

    unit: CompressionUnit
    slot: object


def find_content_router(transforms: object) -> ContentRouter | None:
    """Return the first ContentRouter in a pipeline or iterable."""

    candidates = getattr(transforms, "transforms", transforms)
    if not isinstance(candidates, Iterable):
        return None
    for transform in candidates:
        if isinstance(transform, ContentRouter):
            return transform
    return None


def compress_unit_with_router(
    unit: CompressionUnit,
    *,
    router: ContentRouter,
    tokenizer: TokenCounterLike,
) -> UnitCompressionResult:
    """Compress one safe text unit through ContentRouter.

    The final accept/reject gate uses the provider/model tokenizer, not the
    router's internal word-count estimates.
    """

    tokens_before = tokenizer.count_text(unit.text)
    base = UnitCompressionResult(
        original=unit.text,
        compressed=unit.text,
        modified=False,
        tokens_before=tokens_before,
        tokens_after=tokens_before,
        tokens_saved=0,
        transforms_applied=[],
        strategy=CompressionStrategy.PASSTHROUGH.value,
        router_result=None,
    )

    if not unit.mutable:
        return replace(base, reason="immutable")
    if unit.role == "user":
        return replace(base, reason="protected_user_message")
    if unit.role in {"system", "developer"}:
        return replace(base, reason="protected_system_message")
    if unit.role == "assistant" and unit.metadata.get("compress_assistant") != "true":
        return replace(base, reason="protected_assistant_message")
    if unit.cache_zone != "live":
        return replace(base, reason=f"cache_zone_{unit.cache_zone}")
    if len(unit.text) < unit.min_bytes:
        return replace(base, reason="below_unit_floor")
    if "Retrieve more: hash=" in unit.text or "Retrieve original: hash=" in unit.text:
        return replace(base, reason="already_compressed")

    router_result = router.compress(
        unit.text,
        context=unit.context,
        question=unit.question,
        bias=unit.bias,
    )
    replacement = router_result.compressed
    strategy = router_result.strategy_used.value
    if replacement == unit.text:
        return replace(
            base,
            strategy=strategy,
            router_result=router_result,
            reason="router_no_change",
        )

    tokens_after = tokenizer.count_text(replacement)
    if tokens_after >= tokens_before:
        return replace(
            base,
            compressed=replacement,
            tokens_after=tokens_after,
            strategy=strategy,
            router_result=router_result,
            reason="rejected_not_smaller",
        )

    return UnitCompressionResult(
        original=unit.text,
        compressed=replacement,
        modified=True,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        tokens_saved=tokens_before - tokens_after,
        transforms_applied=[
            f"router:{unit.provider}:{unit.endpoint}:{unit.item_type}:{strategy}",
            strategy,
        ],
        strategy=strategy,
        reason=None,
        router_result=router_result,
    )


def compress_units_with_router(
    units: Iterable[RoutedCompressionUnit],
    *,
    router: ContentRouter,
    tokenizer: TokenCounterLike,
) -> list[tuple[object, UnitCompressionResult]]:
    """Compress provider-extracted units and preserve provider slot refs.

    Provider adapters use this when they have many candidate text slots in one
    request envelope. The slot object is intentionally opaque here; only the
    provider adapter knows how to splice the result back into its native shape.
    """

    return [
        (routed.slot, compress_unit_with_router(routed.unit, router=router, tokenizer=tokenizer))
        for routed in units
    ]
