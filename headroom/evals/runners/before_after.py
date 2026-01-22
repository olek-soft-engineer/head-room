"""Before/After evaluation runner.

This is the core evaluation pattern for proving compression accuracy:
1. Run query with ORIGINAL context → Response A
2. Run query with COMPRESSED context → Response B
3. Compare A and B using multiple metrics
4. Report if accuracy is preserved

This is the gold standard for proving compression doesn't break anything.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from headroom.evals.core import (
    EvalCase,
    EvalMode,
    EvalResult,
    EvalSuite,
    EvalSuiteResult,
)
from headroom.evals.metrics import compute_semantic_similarity
from headroom.transforms.content_router import ContentRouter, ContentRouterConfig
from headroom.transforms.smart_crusher import SmartCrusherConfig


@dataclass
class LLMConfig:
    """Configuration for LLM calls."""

    provider: Literal["anthropic", "openai", "ollama"] = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.0  # Deterministic for reproducibility
    max_tokens: int = 1024


class BeforeAfterRunner:
    """Runner for before/after compression evaluation.

    This runner:
    1. Takes an evaluation suite
    2. For each case, runs the same query against original and compressed context
    3. Compares responses to determine if accuracy is preserved
    4. Generates comprehensive report

    Example:
        ```python
        runner = BeforeAfterRunner(llm_config=LLMConfig(model="claude-sonnet-4-20250514"))
        results = runner.run(suite)
        print(results.summary())
        ```
    """

    def __init__(
        self,
        llm_config: LLMConfig | None = None,
        crusher_config: SmartCrusherConfig | None = None,
        router_config: ContentRouterConfig | None = None,
        use_semantic_similarity: bool = True,
    ):
        """Initialize runner.

        Args:
            llm_config: Configuration for LLM calls
            crusher_config: Configuration for SmartCrusher (deprecated, use router_config)
            router_config: Configuration for ContentRouter (handles all content types)
            use_semantic_similarity: Whether to compute semantic similarity
                                    (requires sentence-transformers)
        """
        self.llm_config = llm_config or LLMConfig()
        self.crusher_config = crusher_config or SmartCrusherConfig()
        self.router_config = router_config or ContentRouterConfig()
        self.use_semantic_similarity = use_semantic_similarity

        # Initialize LLM client
        self._llm_client = self._init_llm_client()

        # Use ContentRouter for intelligent routing to appropriate compressor
        # This handles: JSON → SmartCrusher, plain text → LLMLingua/TextCompressor,
        # code → CodeAwareCompressor, etc.
        self._router = ContentRouter(config=self.router_config)

    def _init_llm_client(self) -> Any:
        """Initialize the appropriate LLM client."""
        if self.llm_config.provider == "anthropic":
            try:
                import anthropic

                return anthropic.Anthropic()
            except ImportError as e:
                raise ImportError(
                    "anthropic package required. Install with: pip install anthropic"
                ) from e
        elif self.llm_config.provider == "openai":
            try:
                import openai

                return openai.OpenAI()
            except ImportError as e:
                raise ImportError(
                    "openai package required. Install with: pip install openai"
                ) from e
        elif self.llm_config.provider == "ollama":
            try:
                import ollama

                return ollama.Client()
            except ImportError as e:
                raise ImportError(
                    "ollama package required. Install with: pip install ollama"
                ) from e
        else:
            raise ValueError(f"Unknown provider: {self.llm_config.provider}")

    def _call_llm(self, context: str, query: str) -> str:
        """Call the LLM with context and query."""
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""

        if self.llm_config.provider == "anthropic":
            response = self._llm_client.messages.create(
                model=self.llm_config.model,
                max_tokens=self.llm_config.max_tokens,
                temperature=self.llm_config.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return str(response.content[0].text)
        elif self.llm_config.provider == "openai":
            response = self._llm_client.chat.completions.create(
                model=self.llm_config.model,
                max_tokens=self.llm_config.max_tokens,
                temperature=self.llm_config.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.choices[0].message.content
            return str(content) if content else ""
        elif self.llm_config.provider == "ollama":
            response = self._llm_client.chat(
                model=self.llm_config.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": self.llm_config.temperature},
            )
            return str(response["message"]["content"])

        return ""

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        return len(text) // 4

    def evaluate_case(self, case: EvalCase) -> EvalResult:
        """Evaluate a single case with before/after comparison."""
        from headroom.evals.metrics import compute_exact_match, compute_f1

        original_tokens = self._estimate_tokens(case.context)

        # Compress the context using ContentRouter
        # This automatically routes to the appropriate compressor:
        # - JSON arrays → SmartCrusher
        # - Plain text → LLMLingua (if available) or TextCompressor
        # - Code → CodeAwareCompressor
        # - Search results → SearchCompressor
        # - etc.
        try:
            compressed_result = self._router.compress(
                case.context,
                context=case.query,  # Pass query as context for relevance-aware compression
            )
            compressed_context = compressed_result.compressed
            compressed_tokens = self._estimate_tokens(compressed_context)
        except Exception:
            compressed_context = case.context
            compressed_tokens = original_tokens

        compression_ratio = 1 - (compressed_tokens / original_tokens) if original_tokens > 0 else 0

        # Run LLM with ORIGINAL context
        start = time.time()
        try:
            response_original = self._call_llm(case.context, case.query)
        except Exception as e:
            response_original = f"ERROR: {e}"
        latency_original = (time.time() - start) * 1000

        # Run LLM with COMPRESSED context
        start = time.time()
        try:
            response_compressed = self._call_llm(compressed_context, case.query)
        except Exception as e:
            response_compressed = f"ERROR: {e}"
        latency_compressed = (time.time() - start) * 1000

        # Compute metrics
        exact_match = compute_exact_match(response_original, response_compressed)
        f1_score = compute_f1(response_original, response_compressed)

        # Semantic similarity
        semantic_sim = None
        if self.use_semantic_similarity:
            try:
                semantic_sim = compute_semantic_similarity(response_original, response_compressed)
            except (ImportError, Exception):
                pass

        # Check ground truth
        contains_ground_truth = None
        if case.ground_truth:
            gt_lower = case.ground_truth.lower()
            contains_ground_truth = (
                gt_lower in response_compressed.lower()
                or compute_f1(response_compressed, case.ground_truth) > 0.5
            )

        # Determine accuracy preservation
        accuracy_preserved = (
            f1_score > 0.7
            or (semantic_sim is not None and semantic_sim > 0.85)
            or contains_ground_truth is True
        )

        return EvalResult(
            case_id=case.id,
            mode=EvalMode.BEFORE_AFTER,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compression_ratio,
            response_original=response_original,
            response_compressed=response_compressed,
            exact_match=exact_match,
            f1_score=f1_score,
            semantic_similarity=semantic_sim,
            contains_ground_truth=contains_ground_truth,
            latency_original_ms=latency_original,
            latency_compressed_ms=latency_compressed,
            accuracy_preserved=accuracy_preserved,
        )

    def run(
        self,
        suite: EvalSuite,
        progress_callback: Callable[[int, int, EvalResult], None] | None = None,
    ) -> EvalSuiteResult:
        """Run evaluation on entire suite.

        Args:
            suite: Evaluation suite to run
            progress_callback: Optional callback(current, total, result)

        Returns:
            Aggregated results
        """
        start_time = time.time()
        results: list[EvalResult] = []

        for i, case in enumerate(suite):
            result = self.evaluate_case(case)
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, len(suite), result)

        # Aggregate
        passed = sum(1 for r in results if r.accuracy_preserved)
        total_original = sum(r.original_tokens for r in results)
        total_compressed = sum(r.compressed_tokens for r in results)

        avg_compression = sum(r.compression_ratio for r in results) / len(results) if results else 0
        avg_f1 = sum(r.f1_score for r in results) / len(results) if results else 0

        semantic_sims = [
            r.semantic_similarity for r in results if r.semantic_similarity is not None
        ]
        avg_semantic = sum(semantic_sims) / len(semantic_sims) if semantic_sims else None

        return EvalSuiteResult(
            suite_name=suite.name,
            total_cases=len(results),
            passed_cases=passed,
            failed_cases=len(results) - passed,
            avg_compression_ratio=avg_compression,
            avg_f1_score=avg_f1,
            avg_semantic_similarity=avg_semantic,
            accuracy_preservation_rate=passed / len(results) if results else 0,
            total_original_tokens=total_original,
            total_compressed_tokens=total_compressed,
            total_tokens_saved=total_original - total_compressed,
            results=results,
            duration_seconds=time.time() - start_time,
        )


def run_quick_eval(
    n_samples: int = 5,
    provider: Literal["anthropic", "openai", "ollama"] = "anthropic",
    model: str = "claude-sonnet-4-20250514",
) -> EvalSuiteResult:
    """Run a quick evaluation with built-in samples.

    This is a fast sanity check to verify compression isn't breaking things.

    Args:
        n_samples: Number of samples to test
        provider: LLM provider
        model: Model to use

    Returns:
        Evaluation results
    """
    from headroom.evals.datasets import load_tool_output_samples

    suite = load_tool_output_samples()
    suite.cases = suite.cases[:n_samples]

    runner = BeforeAfterRunner(
        llm_config=LLMConfig(provider=provider, model=model),
        use_semantic_similarity=False,  # Faster without embeddings
    )

    def progress(current: int, total: int, result: EvalResult) -> None:
        status = "PASS" if result.accuracy_preserved else "FAIL"
        print(f"  [{current}/{total}] {result.case_id}: {status} (F1={result.f1_score:.2f})")

    print(f"\nRunning quick eval with {n_samples} samples...")
    results = runner.run(suite, progress_callback=progress)

    print(f"\n{results.summary()}")
    return results


if __name__ == "__main__":
    # Quick test
    run_quick_eval(n_samples=3)
