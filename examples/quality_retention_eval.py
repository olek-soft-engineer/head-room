#!/usr/bin/env python3
"""
Quality Retention Evaluation for Intelligent Context Management

This eval verifies that when we intelligently drop/compress content,
we RETAIN critical information that the model needs to answer correctly.

Methodology:
1. NEEDLE-IN-HAYSTACK: Embed specific facts in large tool outputs
2. COMPRESS: Apply IntelligentContextManager
3. VERIFY: Ask questions requiring those facts
4. SCORE: Compare answers before/after compression

Key Metrics:
- Retention Rate: % of critical facts preserved
- Answer Accuracy: % of verification questions answered correctly
- Quality-Adjusted Savings: compression_ratio * retention_rate

Usage:
    ANTHROPIC_API_KEY=sk-... python examples/quality_retention_eval.py
"""

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any

# Check for API key early
API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not API_KEY:
    print("ERROR: ANTHROPIC_API_KEY environment variable required")
    sys.exit(1)

from anthropic import Anthropic  # noqa: E402

from headroom import AnthropicProvider, HeadroomClient  # noqa: E402
from headroom.config import HeadroomConfig, IntelligentContextConfig  # noqa: E402
from headroom.tokenizer import Tokenizer  # noqa: E402
from headroom.tokenizers import TiktokenCounter  # noqa: E402

# =============================================================================
# TEST CASE DEFINITIONS
# =============================================================================


@dataclass
class CriticalFact:
    """A fact that MUST be retained after compression."""

    description: str
    value: str
    verification_question: str
    expected_answer_contains: list[str]  # Answer should contain these strings


@dataclass
class EvalTestCase:
    """A test case for quality retention evaluation."""

    name: str
    description: str
    messages: list[dict[str, Any]]
    critical_facts: list[CriticalFact]
    system_prompt: str = "You are a helpful assistant. Answer questions accurately based on the information provided."


@dataclass
class EvalResult:
    """Result of evaluating a single test case."""

    test_name: str
    tokens_before: int
    tokens_after: int
    compression_ratio: float
    facts_tested: int
    facts_retained: int
    retention_rate: float
    quality_adjusted_savings: float
    details: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


# =============================================================================
# TEST CASE GENERATORS
# =============================================================================


def create_search_results_with_needles() -> EvalTestCase:
    """
    Test case: 100 search results with 3 critical "needles" hidden inside.

    The critical facts are specific error reports that should be retained
    even when most results are compressed away.
    """
    results = []

    # Needle 1: Critical memory error at position 42
    # Needle 2: Database failure at position 17
    # Needle 3: Security incident at position 73

    for i in range(100):
        if i == 42:
            result = {
                "id": f"INC-{i:04d}",
                "title": "CRITICAL: OutOfMemory in production worker pool",
                "severity": "P0",
                "timestamp": "2024-01-15T10:42:37Z",
                "server": "prod-worker-7",
                "details": {
                    "error_type": "java.lang.OutOfMemoryError",
                    "heap_used_gb": 7.8,
                    "heap_max_gb": 8.0,
                    "thread_count": 847,
                    "affected_users": 12847,
                },
                "root_cause": "Thread pool executor not releasing completed tasks",
                "resolution": "Restart required, fix deployed in v2.4.1",
            }
        elif i == 17:
            result = {
                "id": f"INC-{i:04d}",
                "title": "Database connection pool exhausted",
                "severity": "P1",
                "timestamp": "2024-01-15T09:17:22Z",
                "server": "db-primary-3",
                "details": {
                    "error_type": "ConnectionPoolExhausted",
                    "active_connections": 500,
                    "max_connections": 500,
                    "waiting_queries": 2341,
                    "longest_wait_ms": 45000,
                },
                "root_cause": "Slow query from analytics job holding connections",
                "resolution": "Killed analytics query, added connection timeout",
            }
        elif i == 73:
            result = {
                "id": f"INC-{i:04d}",
                "title": "SECURITY: Unauthorized API access attempt blocked",
                "severity": "P0",
                "timestamp": "2024-01-15T14:23:55Z",
                "source_ip": "203.0.113.42",
                "details": {
                    "attack_type": "credential_stuffing",
                    "attempts": 15847,
                    "accounts_targeted": 892,
                    "accounts_compromised": 0,
                    "blocked_by": "rate_limiter_v2",
                },
                "root_cause": "Stolen credentials from third-party breach",
                "resolution": "IP blocked, affected users notified to reset passwords",
            }
        else:
            result = {
                "id": f"INC-{i:04d}",
                "title": f"Routine alert #{i}: {'CPU spike' if i % 3 == 0 else 'Latency increase' if i % 3 == 1 else 'Disk usage warning'}",
                "severity": "P3",
                "timestamp": f"2024-01-15T{10 + (i % 8):02d}:{i % 60:02d}:00Z",
                "server": f"app-server-{i % 20}",
                "details": {
                    "metric": "cpu_percent"
                    if i % 3 == 0
                    else "latency_p99"
                    if i % 3 == 1
                    else "disk_usage",
                    "value": 75 + (i % 20),
                    "threshold": 80,
                    "duration_minutes": 5 + (i % 10),
                },
                "root_cause": "Normal traffic variation",
                "resolution": "Auto-resolved",
            }
        results.append(result)

    messages = [
        {"role": "user", "content": "Search for all incidents from today"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_search",
                    "name": "search_incidents",
                    "input": {"date": "2024-01-15"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_search",
                    "content": json.dumps(results, indent=2),
                }
            ],
        },
        {
            "role": "user",
            "content": "What were the critical issues today? I need specific details.",
        },
    ]

    critical_facts = [
        CriticalFact(
            description="Memory error server identification",
            value="prod-worker-7",
            verification_question="Which server experienced the OutOfMemory error?",
            expected_answer_contains=["prod-worker-7"],
        ),
        CriticalFact(
            description="Memory error affected users count",
            value="12847",
            verification_question="How many users were affected by the memory error?",
            expected_answer_contains=["12847", "12,847"],
        ),
        CriticalFact(
            description="Database error connection count",
            value="500 active connections",
            verification_question="How many active database connections were there when the pool was exhausted?",
            expected_answer_contains=["500"],
        ),
        CriticalFact(
            description="Security attack source IP",
            value="203.0.113.42",
            verification_question="What IP address was the source of the credential stuffing attack?",
            expected_answer_contains=["203.0.113.42"],
        ),
        CriticalFact(
            description="Security attack attempt count",
            value="15847 attempts",
            verification_question="How many credential stuffing attempts were made?",
            expected_answer_contains=["15847", "15,847"],
        ),
    ]

    return EvalTestCase(
        name="search_results_needles",
        description="100 search results with 3 critical incidents (needles) at positions 17, 42, 73",
        messages=messages,
        critical_facts=critical_facts,
    )


def create_log_analysis_with_needles() -> EvalTestCase:
    """
    Test case: 200 log entries with critical error buried in the middle.
    """
    logs = []

    for i in range(200):
        if i == 127:
            # The critical needle - a specific error with unique identifiers
            log = {
                "timestamp": "2024-01-15T11:27:33.847Z",
                "level": "ERROR",
                "service": "payment-gateway",
                "trace_id": "abc123def456",
                "message": "Payment processing failed: Card declined",
                "details": {
                    "transaction_id": "TXN-98765432",
                    "amount_cents": 15999,
                    "currency": "USD",
                    "error_code": "CARD_DECLINED_INSUFFICIENT_FUNDS",
                    "customer_id": "CUST-789012",
                    "retry_count": 3,
                    "final_status": "FAILED",
                },
            }
        elif i == 45:
            # Another needle - rate limit hit
            log = {
                "timestamp": "2024-01-15T10:45:12.123Z",
                "level": "WARN",
                "service": "api-gateway",
                "message": "Rate limit exceeded for client",
                "details": {
                    "client_id": "CLIENT-ACME-001",
                    "endpoint": "/api/v2/bulk-upload",
                    "requests_per_minute": 1500,
                    "limit": 1000,
                    "blocked_duration_seconds": 300,
                },
            }
        else:
            log = {
                "timestamp": f"2024-01-15T{10 + (i % 4):02d}:{i % 60:02d}:{i % 60:02d}.{i % 1000:03d}Z",
                "level": "INFO",
                "service": ["api", "auth", "worker", "cache", "db"][i % 5],
                "message": f"Request processed successfully (id={i})",
                "details": {"latency_ms": 50 + (i % 100), "status": 200},
            }
        logs.append(log)

    messages = [
        {"role": "user", "content": "Get the logs from the last hour"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_logs",
                    "name": "get_logs",
                    "input": {"timerange": "1h"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_logs",
                    "content": json.dumps(logs, indent=2),
                }
            ],
        },
        {"role": "user", "content": "Are there any errors or warnings I should know about?"},
    ]

    critical_facts = [
        CriticalFact(
            description="Failed transaction ID",
            value="TXN-98765432",
            verification_question="What was the transaction ID of the failed payment?",
            expected_answer_contains=["TXN-98765432"],
        ),
        CriticalFact(
            description="Payment failure amount",
            value="$159.99 (15999 cents)",
            verification_question="What was the amount of the failed payment in dollars?",
            expected_answer_contains=["159.99", "159"],
        ),
        CriticalFact(
            description="Rate limited client",
            value="CLIENT-ACME-001",
            verification_question="Which client was rate limited?",
            expected_answer_contains=["CLIENT-ACME-001", "ACME"],
        ),
        CriticalFact(
            description="Rate limit endpoint",
            value="/api/v2/bulk-upload",
            verification_question="Which API endpoint triggered the rate limit?",
            expected_answer_contains=["bulk-upload", "/api/v2/bulk-upload"],
        ),
    ]

    return EvalTestCase(
        name="log_analysis_needles",
        description="200 log entries with critical error at position 127 and warning at position 45",
        messages=messages,
        critical_facts=critical_facts,
    )


def create_code_review_with_needles() -> EvalTestCase:
    """
    Test case: Code review results with security vulnerabilities hidden in large output.
    """
    files = []

    for i in range(80):
        if i == 23:
            # Critical security vulnerability
            file_result = {
                "path": "src/auth/login.py",
                "issues": [
                    {
                        "line": 47,
                        "severity": "CRITICAL",
                        "type": "SQL_INJECTION",
                        "message": "User input directly concatenated into SQL query",
                        "code_snippet": "query = f\"SELECT * FROM users WHERE username = '{username}'\"",
                        "fix": "Use parameterized queries: cursor.execute('SELECT * FROM users WHERE username = ?', (username,))",
                    },
                    {
                        "line": 89,
                        "severity": "HIGH",
                        "type": "HARDCODED_SECRET",
                        "message": "API key hardcoded in source code",
                        "code_snippet": 'API_KEY = "sk-prod-a1b2c3d4e5f6g7h8i9j0"',
                        "fix": "Use environment variables or secret management",
                    },
                ],
                "metrics": {"complexity": 34, "coverage": 0.12},
            }
        elif i == 56:
            # Another critical issue
            file_result = {
                "path": "src/api/upload.py",
                "issues": [
                    {
                        "line": 112,
                        "severity": "CRITICAL",
                        "type": "PATH_TRAVERSAL",
                        "message": "File path not sanitized, allows directory traversal",
                        "code_snippet": "file_path = os.path.join(UPLOAD_DIR, user_filename)",
                        "fix": "Use secure_filename() and validate path stays within UPLOAD_DIR",
                    }
                ],
                "metrics": {"complexity": 28, "coverage": 0.45},
            }
        else:
            file_result = {
                "path": f"src/module_{i}/handler.py",
                "issues": [
                    {
                        "line": 10 + (i % 50),
                        "severity": "LOW",
                        "type": "STYLE",
                        "message": "Line too long (> 100 characters)",
                    }
                ]
                if i % 4 == 0
                else [],
                "metrics": {"complexity": 5 + (i % 15), "coverage": 0.7 + (i % 30) / 100},
            }
        files.append(file_result)

    messages = [
        {"role": "user", "content": "Run a security scan on the codebase"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_scan",
                    "name": "security_scan",
                    "input": {"path": "src/"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_scan",
                    "content": json.dumps(files, indent=2),
                }
            ],
        },
        {
            "role": "user",
            "content": "What are the critical security vulnerabilities I need to fix immediately?",
        },
    ]

    critical_facts = [
        CriticalFact(
            description="SQL injection file location",
            value="src/auth/login.py line 47",
            verification_question="In which file and line number is the SQL injection vulnerability?",
            expected_answer_contains=["login.py", "47"],
        ),
        CriticalFact(
            description="Hardcoded API key value",
            value="sk-prod-a1b2c3d4e5f6g7h8i9j0",
            verification_question="What is the hardcoded API key that was found?",
            expected_answer_contains=["sk-prod", "a1b2c3d4"],
        ),
        CriticalFact(
            description="Path traversal file",
            value="src/api/upload.py",
            verification_question="Which file has the path traversal vulnerability?",
            expected_answer_contains=["upload.py"],
        ),
        CriticalFact(
            description="Path traversal line",
            value="line 112",
            verification_question="What line number has the path traversal issue in upload.py?",
            expected_answer_contains=["112"],
        ),
    ]

    return EvalTestCase(
        name="code_review_needles",
        description="80 file scan results with critical vulnerabilities at positions 23 and 56",
        messages=messages,
        critical_facts=critical_facts,
    )


# =============================================================================
# EVALUATION ENGINE
# =============================================================================


class QualityRetentionEvaluator:
    """Evaluates whether compression retains critical information."""

    # Tool definitions required by Anthropic API when messages contain tool results
    TOOL_DEFINITIONS = [
        {
            "name": "search_incidents",
            "description": "Search for incidents by date",
            "input_schema": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date to search (YYYY-MM-DD)"}
                },
                "required": ["date"],
            },
        },
        {
            "name": "get_logs",
            "description": "Get logs for a time range",
            "input_schema": {
                "type": "object",
                "properties": {
                    "timerange": {"type": "string", "description": "Time range (e.g., '1h', '24h')"}
                },
                "required": ["timerange"],
            },
        },
        {
            "name": "security_scan",
            "description": "Run security scan on codebase",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Path to scan"}},
                "required": ["path"],
            },
        },
    ]

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_client = Anthropic(api_key=api_key)
        self.provider = AnthropicProvider()
        self.tokenizer = Tokenizer(TiktokenCounter())

    def _create_client(self, config: HeadroomConfig) -> HeadroomClient:
        """Create a HeadroomClient with the given config."""
        return HeadroomClient(
            original_client=Anthropic(api_key=self.api_key),
            provider=self.provider,
            default_mode="optimize",
            config=config,
        )

    def _ask_question(
        self,
        client: HeadroomClient,
        messages: list[dict],
        question: str,
        system_prompt: str,
    ) -> str:
        """Ask a verification question and get the response."""
        # Add the verification question to the conversation
        eval_messages = messages + [{"role": "user", "content": question}]

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            messages=eval_messages,
            system=system_prompt,
            tools=self.TOOL_DEFINITIONS,  # Required when messages contain tool results
            max_tokens=300,
        )

        if hasattr(response, "content") and response.content:
            if isinstance(response.content, list):
                return response.content[0].text if response.content else ""
            return str(response.content)
        return str(response)

    def _check_answer(self, answer: str, expected_contains: list[str]) -> bool:
        """Check if answer contains expected content."""
        answer_lower = answer.lower()
        for expected in expected_contains:
            if expected.lower() in answer_lower:
                return True
        return False

    def evaluate_test_case(
        self,
        test_case: EvalTestCase,
        config: HeadroomConfig,
    ) -> EvalResult:
        """Evaluate a single test case."""
        print(f"\n{'=' * 60}")
        print(f"EVAL: {test_case.name}")
        print(f"{'=' * 60}")
        print(f"Description: {test_case.description}")

        # Count tokens before compression
        tokens_before = self.tokenizer.count_messages(test_case.messages)
        print(f"Tokens before: {tokens_before:,}")

        # Create client with compression enabled
        client = self._create_client(config)

        # Simulate compression to see what we get
        try:
            sim_result = client.messages.simulate(
                model="claude-sonnet-4-20250514",
                messages=test_case.messages,
                system=test_case.system_prompt,
                tools=self.TOOL_DEFINITIONS,  # Required when messages contain tool results
            )
            tokens_after = sim_result.tokens_after
            compression_ratio = 1 - (tokens_after / tokens_before) if tokens_before > 0 else 0
            print(f"Tokens after:  {tokens_after:,}")
            print(f"Compression:   {compression_ratio * 100:.1f}%")
            print(f"Transforms:    {sim_result.transforms[:3]}...")  # First 3
        except Exception as e:
            return EvalResult(
                test_name=test_case.name,
                tokens_before=tokens_before,
                tokens_after=0,
                compression_ratio=0,
                facts_tested=len(test_case.critical_facts),
                facts_retained=0,
                retention_rate=0,
                quality_adjusted_savings=0,
                error=str(e),
            )

        # Now verify each critical fact
        print(f"\nVerifying {len(test_case.critical_facts)} critical facts...")
        facts_retained = 0
        details = []

        for fact in test_case.critical_facts:
            print(f"\n  Fact: {fact.description}")
            print(f"  Question: {fact.verification_question}")

            try:
                # Ask the verification question (with compression applied)
                answer = self._ask_question(
                    client,
                    test_case.messages,
                    fact.verification_question,
                    test_case.system_prompt,
                )

                # Check if answer contains expected content
                retained = self._check_answer(answer, fact.expected_answer_contains)

                if retained:
                    facts_retained += 1
                    print("  Result: ✅ RETAINED")
                    print(f"  Answer: {answer[:100]}...")
                else:
                    print("  Result: ❌ LOST")
                    print(f"  Expected: {fact.expected_answer_contains}")
                    print(f"  Got: {answer[:150]}...")

                details.append(
                    {
                        "fact": fact.description,
                        "question": fact.verification_question,
                        "expected": fact.expected_answer_contains,
                        "answer": answer[:200],
                        "retained": retained,
                    }
                )

            except Exception as e:
                print(f"  Result: ❌ ERROR: {e}")
                details.append(
                    {
                        "fact": fact.description,
                        "error": str(e),
                        "retained": False,
                    }
                )

        retention_rate = (
            facts_retained / len(test_case.critical_facts) if test_case.critical_facts else 0
        )
        quality_adjusted_savings = compression_ratio * retention_rate

        print(f"\n{'=' * 60}")
        print(
            f"RESULT: {facts_retained}/{len(test_case.critical_facts)} facts retained ({retention_rate * 100:.0f}%)"
        )
        print(f"Quality-Adjusted Savings: {quality_adjusted_savings * 100:.1f}%")
        print(f"{'=' * 60}")

        return EvalResult(
            test_name=test_case.name,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            compression_ratio=compression_ratio,
            facts_tested=len(test_case.critical_facts),
            facts_retained=facts_retained,
            retention_rate=retention_rate,
            quality_adjusted_savings=quality_adjusted_savings,
            details=details,
        )

    def run_full_eval(self, config: HeadroomConfig) -> list[EvalResult]:
        """Run evaluation on all test cases."""
        test_cases = [
            create_search_results_with_needles(),
            create_log_analysis_with_needles(),
            create_code_review_with_needles(),
        ]

        results = []
        for test_case in test_cases:
            result = self.evaluate_test_case(test_case, config)
            results.append(result)

        return results


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("\n" + "=" * 70)
    print("QUALITY RETENTION EVALUATION")
    print("=" * 70)
    print("Verifying that intelligent compression retains critical information")
    print("=" * 70)

    # Create config with intelligent context enabled
    config = HeadroomConfig()
    config.intelligent_context = IntelligentContextConfig(
        enabled=True,
        use_importance_scoring=True,
        compress_threshold=0.10,
        summarize_threshold=0.25,
    )
    config.rolling_window.enabled = False
    config.smart_crusher.enabled = True

    # Run evaluation
    evaluator = QualityRetentionEvaluator(API_KEY)
    results = evaluator.run_full_eval(config)

    # Summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    total_facts = sum(r.facts_tested for r in results)
    total_retained = sum(r.facts_retained for r in results)
    total_tokens_before = sum(r.tokens_before for r in results)
    total_tokens_after = sum(r.tokens_after for r in results)

    print(f"\n{'Test Case':<30} {'Compression':<15} {'Retention':<15} {'Quality-Adj':<15}")
    print("-" * 75)

    for result in results:
        status = (
            "✅" if result.retention_rate >= 0.8 else "⚠️" if result.retention_rate >= 0.5 else "❌"
        )
        print(
            f"{result.test_name:<30} "
            f"{result.compression_ratio * 100:>6.1f}%        "
            f"{result.retention_rate * 100:>6.0f}% ({result.facts_retained}/{result.facts_tested})  "
            f"{result.quality_adjusted_savings * 100:>6.1f}%  {status}"
        )

    print("-" * 75)
    overall_compression = (
        1 - (total_tokens_after / total_tokens_before) if total_tokens_before > 0 else 0
    )
    overall_retention = total_retained / total_facts if total_facts > 0 else 0
    overall_quality_adj = overall_compression * overall_retention

    print(
        f"{'OVERALL':<30} "
        f"{overall_compression * 100:>6.1f}%        "
        f"{overall_retention * 100:>6.0f}% ({total_retained}/{total_facts})  "
        f"{overall_quality_adj * 100:>6.1f}%"
    )

    print("\n" + "=" * 70)
    if overall_retention >= 0.8:
        print("✅ PASS: Critical information retention is good (>=80%)")
    elif overall_retention >= 0.5:
        print("⚠️ WARNING: Some critical information was lost (50-80% retention)")
    else:
        print("❌ FAIL: Significant critical information loss (<50% retention)")
    print(
        f"Tokens saved: {total_tokens_before - total_tokens_after:,} ({overall_compression * 100:.1f}% compression)"
    )
    print("=" * 70)

    return 0 if overall_retention >= 0.8 else 1


if __name__ == "__main__":
    sys.exit(main())
