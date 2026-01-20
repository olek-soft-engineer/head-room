#!/usr/bin/env python3
"""
Real Data Demo: Headroom with Production-Scale Data

This demo uses realistic VOLUME of data to show Headroom's value:
- 100 log entries (one critical error buried inside)
- 50 pods (one unhealthy)
- 200 Prometheus metrics (a few critical ones)
- Real code, config, and service definitions

The scenario: Debug why a Kubernetes deployment is failing.
The agent must find the needle (database connection error) in the haystack.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/real_data_demo.py
"""

import json
import os
import time

from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.tools import tool

if not os.environ.get("ANTHROPIC_API_KEY"):
    raise ValueError("ANTHROPIC_API_KEY environment variable required")

MODEL_ID = "claude-sonnet-4-20250514"


# =============================================================================
# REALISTIC DATA GENERATORS - Production-scale volume
# =============================================================================


def generate_app_logs(count: int = 100, error_at: int = 67) -> str:
    """Generate 100 log lines with critical error at position 67."""
    lines = []
    services = [
        "api-gateway",
        "user-service",
        "order-service",
        "payment-service",
        "inventory-service",
    ]

    for i in range(count):
        ts = f"2024-01-15T14:{i // 60:02d}:{i % 60:02d}.{(i * 123) % 1000:03d}Z"

        if i == error_at:
            # THE CRITICAL ERROR - buried in the middle
            lines.append(
                f"{ts} ERROR [main] c.zaxxer.hikari.pool.HikariPool - HikariPool-1 - Exception during pool initialization"
            )
            lines.append(
                "org.postgresql.util.PSQLException: Connection to db.internal:5432 refused. Check that the hostname and port are correct and that the postmaster is accepting TCP/IP connections."
            )
            lines.append(
                "    at org.postgresql.core.v3.ConnectionFactoryImpl.openConnectionImpl(ConnectionFactoryImpl.java:319)"
            )
            lines.append(
                "    at org.postgresql.core.ConnectionFactory.openConnection(ConnectionFactory.java:49)"
            )
            lines.append("    at org.postgresql.jdbc.PgConnection.<init>(PgConnection.java:223)")
            lines.append(
                "    at com.zaxxer.hikari.util.DriverDataSource.getConnection(DriverDataSource.java:138)"
            )
            lines.append("    at com.zaxxer.hikari.pool.PoolBase.newConnection(PoolBase.java:364)")
            lines.append("Caused by: java.net.ConnectException: Connection refused")
            lines.append("    at java.base/sun.nio.ch.Net.pollConnect(Native Method)")
            lines.append("    ... 15 more")
        else:
            svc = services[i % len(services)]
            # Normal INFO logs - the haystack
            messages = [
                f"Request processed successfully - latency={50 + i % 30}ms",
                f"Cache hit for key user:{i * 7}",
                "Connection pool stats: active=5, idle=10, total=15",
                "Health check passed - all dependencies healthy",
                f"Processed batch of {10 + i % 20} items in {100 + i % 50}ms",
            ]
            lines.append(
                f"{ts} INFO  [{svc}] c.m.a.{svc.replace('-', '.')} - {messages[i % len(messages)]}"
            )

    return "\n".join(lines)


def generate_pod_list(count: int = 50, unhealthy_at: int = 23) -> str:
    """Generate kubectl get pods output with one unhealthy pod."""
    header = (
        "NAME                                      READY   STATUS             RESTARTS      AGE"
    )
    lines = [header]

    deployments = [
        "api-server",
        "user-service",
        "order-service",
        "payment-service",
        "inventory-service",
        "cache-service",
    ]

    for i in range(count):
        deploy = deployments[i % len(deployments)]
        suffix = f"{i:05x}"[:5]
        name = f"{deploy}-{suffix[:5]}-{suffix[2:]}"

        if i == unhealthy_at:
            # THE UNHEALTHY POD
            lines.append(
                "api-server-7f8d9-m4n2p                    0/1     CrashLoopBackOff   5 (30s ago)   10m"
            )
        else:
            # Healthy pods - the haystack
            age = f"{(i * 3) % 24}h" if i % 3 == 0 else f"{(i * 7) % 30}d"
            lines.append(f"{name:<42} 1/1     Running            0             {age}")

    return "\n".join(lines)


def generate_prometheus_metrics(count: int = 200, critical_at: list = None) -> str:
    """Generate Prometheus metrics response with critical metrics mixed in."""
    if critical_at is None:
        critical_at = [47, 123, 156]  # Positions of critical metrics

    results = []

    # Common metric types
    metric_types = [
        ("http_requests_total", ["handler", "method", "status"]),
        ("http_request_duration_seconds", ["handler", "method"]),
        ("process_cpu_seconds_total", ["instance"]),
        ("go_goroutines", ["instance"]),
        ("node_memory_MemAvailable_bytes", ["instance"]),
        ("container_memory_usage_bytes", ["pod", "container"]),
    ]

    handlers = ["/api/users", "/api/orders", "/api/products", "/api/health", "/api/metrics"]
    instances = ["10.244.1.10:8080", "10.244.1.11:8080", "10.244.2.15:8080", "10.244.3.20:8080"]
    pods = [
        "api-server-abc12",
        "user-service-def34",
        "order-service-ghi56",
        "payment-service-jkl78",
    ]

    for i in range(count):
        ts = 1705329825 + i

        if i in critical_at:
            # Critical metrics showing the problem
            if i == critical_at[0]:
                results.append(
                    {
                        "metric": {
                            "__name__": "up",
                            "job": "postgresql",
                            "instance": "db.internal:5432",
                        },
                        "value": [ts, "0"],  # DATABASE IS DOWN
                    }
                )
            elif i == critical_at[1]:
                results.append(
                    {
                        "metric": {"__name__": "pg_up", "datname": "myapp_production"},
                        "value": [ts, "0"],  # DATABASE NOT ACCEPTING CONNECTIONS
                    }
                )
            elif i == critical_at[2]:
                results.append(
                    {
                        "metric": {
                            "__name__": "kube_pod_container_status_restarts_total",
                            "pod": "api-server-7f8d9-m4n2p",
                        },
                        "value": [ts, "5"],  # POD RESTARTING
                    }
                )
        else:
            # Normal healthy metrics - the haystack
            metric_name, label_keys = metric_types[i % len(metric_types)]
            metric = {"__name__": metric_name}

            if "handler" in label_keys:
                metric["handler"] = handlers[i % len(handlers)]
                metric["method"] = "GET" if i % 2 == 0 else "POST"
            if "status" in label_keys:
                metric["status"] = "200"  # All healthy
            if "instance" in label_keys:
                metric["instance"] = instances[i % len(instances)]
            if "pod" in label_keys:
                metric["pod"] = pods[i % len(pods)]
                metric["container"] = "main"

            # Normal values
            value = str(1000 + (i * 17) % 5000)
            if "duration" in metric_name:
                value = f"0.{(50 + i % 200):03d}"
            elif metric_name == "up":
                value = "1"  # Healthy

            results.append({"metric": metric, "value": [ts, value]})

    return json.dumps(
        {"status": "success", "data": {"resultType": "vector", "result": results}}, indent=2
    )


# Generate the production-scale data
APP_LOGS = generate_app_logs(100, error_at=67)
POD_LIST = generate_pod_list(50, unhealthy_at=23)
PROMETHEUS_METRICS = generate_prometheus_metrics(200)

# Static realistic content (not repetitive, but real)
K8S_POD_DESCRIBE = """Name:             api-server-7f8d9-m4n2p
Namespace:        production
Status:           Running
IP:               10.244.2.45
Containers:
  api-server:
    State:          Waiting
      Reason:       CrashLoopBackOff
    Ready:          False
    Restart Count:  5
    Limits:
      cpu:     500m
      memory:  512Mi
    Environment:
      DATABASE_URL:  <set to the key 'url' in secret 'db-credentials'>
Events:
  Type     Reason     Age                Message
  ----     ------     ----               -------
  Warning  Unhealthy  4m (x3 over 4m)    Readiness probe failed: HTTP 503
  Warning  Unhealthy  3m (x9 over 4m)    Liveness probe failed: HTTP 503
  Warning  BackOff    30s (x5 over 2m)   Back-off restarting failed container"""

K8S_SERVICES = """NAME           TYPE        CLUSTER-IP       EXTERNAL-IP   PORT(S)    AGE
api-server     ClusterIP   10.96.45.123     <none>        8080/TCP   30d
db             ClusterIP   10.96.78.234     <none>        5432/TCP   30d
redis-master   ClusterIP   10.96.12.345     <none>        6379/TCP   30d

ENDPOINTS:
NAME           ENDPOINTS                                      AGE
api-server     10.244.2.45:8080,10.244.1.33:8080              30d
db             <none>                                         30d
redis-master   10.244.3.67:6379                               30d"""

HEALTH_CHECK_CODE = '''"""Health check endpoints for Kubernetes probes."""
from fastapi import APIRouter, Response, status
from sqlalchemy import text

router = APIRouter()

@router.get("/health")
async def health_check(response: Response):
    """Liveness probe - checks database connectivity."""
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return {"status": "healthy"}
    except Exception as e:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "unhealthy", "error": str(e)}
'''

DB_CONFIG = """# config/database.yaml
production:
  adapter: postgresql
  host: db.internal
  port: 5432
  database: myapp_production
  pool: 25
  timeout: 5000
  max_connections: 100"""


# =============================================================================
# TOOLS - Return production-scale data
# =============================================================================


@tool(name="kubectl_describe_pod")
def kubectl_describe_pod(pod_name: str) -> str:
    """Describe a Kubernetes pod."""
    return K8S_POD_DESCRIBE


@tool(name="kubectl_get_pods")
def kubectl_get_pods(namespace: str = "production") -> str:
    """List all pods in namespace."""
    return POD_LIST


@tool(name="kubectl_get_services")
def kubectl_get_services(namespace: str = "production") -> str:
    """Get Kubernetes services and endpoints."""
    return K8S_SERVICES


@tool(name="get_application_logs")
def get_application_logs(pod_name: str, lines: int = 100) -> str:
    """Get application logs from a pod."""
    return APP_LOGS


@tool(name="get_source_code")
def get_source_code(file_path: str) -> str:
    """Read source code file."""
    return HEALTH_CHECK_CODE


@tool(name="get_config_file")
def get_config_file(path: str) -> str:
    """Read a configuration file."""
    return DB_CONFIG


@tool(name="query_prometheus")
def query_prometheus(query: str) -> str:
    """Query Prometheus metrics."""
    return PROMETHEUS_METRICS


# =============================================================================
# GROUND TRUTH - What we expect the agent to find
# =============================================================================


def verify_response(response: str) -> dict[str, bool]:
    """Verify the agent found the key information."""
    response_lower = response.lower()

    return {
        "found_db_error": any(
            term in response_lower
            for term in ["connection refused", "db.internal", "5432", "postgresql", "psqlexception"]
        ),
        "found_pod_issue": any(
            term in response_lower
            for term in ["crashloop", "restart", "unhealthy", "503", "probe failed"]
        ),
        "found_endpoint_issue": any(
            term in response_lower for term in ["endpoint", "no endpoints", "<none>", "db service"]
        ),
        "found_root_cause": any(
            term in response_lower for term in ["database", "connection", "postgres"]
        )
        and "down" in response_lower
        or "fail" in response_lower
        or "refused" in response_lower,
    }


# =============================================================================
# MAIN DEMO
# =============================================================================


def main():
    from headroom.integrations.agno import HeadroomAgnoModel
    from headroom.pricing import estimate_cost

    print("\n" + "=" * 70)
    print("  REAL DATA DEMO: Production-Scale Kubernetes Investigation")
    print("=" * 70)

    # Show what data we're using
    data_sources = {
        "Application logs (100 entries)": APP_LOGS,
        "Pod list (50 pods)": POD_LIST,
        "Prometheus metrics (200 series)": PROMETHEUS_METRICS,
        "Pod describe": K8S_POD_DESCRIBE,
        "Services/Endpoints": K8S_SERVICES,
        "Health check code": HEALTH_CHECK_CODE,
        "Database config": DB_CONFIG,
    }

    total_chars = sum(len(v) for v in data_sources.values())

    print("\n  Data sources (production-scale volume):")
    for name, data in data_sources.items():
        print(f"    {name:<35} {len(data):>8,} chars")
    print(f"    {'─' * 50}")
    print(f"    {'TOTAL':<35} {total_chars:>8,} chars")

    print("\n  Content types: Java stack traces, K8s YAML, Python code, Prometheus JSON")
    print("  Challenge: Find critical error at position 67 in 100 log entries")

    # Create agent with Headroom
    base_model = Claude(id=MODEL_ID)
    model = HeadroomAgnoModel(wrapped_model=base_model)

    agent = Agent(
        model=model,
        tools=[
            kubectl_describe_pod,
            kubectl_get_pods,
            kubectl_get_services,
            get_application_logs,
            get_source_code,
            get_config_file,
            query_prometheus,
        ],
        markdown=True,
    )

    question = """Our api-server deployment in production is failing. Pods keep restarting.

Please investigate:
1. List all pods and identify unhealthy ones
2. Describe the problematic pod
3. Check application logs for errors
4. Verify services and endpoints
5. Check Prometheus metrics for anomalies

Find the ROOT CAUSE and explain what's failing."""

    print("\n  Running investigation...")
    start = time.time()

    response = agent.run(question)
    response_text = response.content if hasattr(response, "content") else str(response)

    duration = time.time() - start

    # Get Headroom stats
    stats = model.get_savings_summary()
    tokens_before = stats["total_tokens_before"]
    tokens_after = stats["total_tokens_after"]
    tokens_saved = stats["total_tokens_saved"]
    pct_saved = (tokens_saved / tokens_before * 100) if tokens_before > 0 else 0

    # Calculate costs
    cost_before = estimate_cost(MODEL_ID, input_tokens=tokens_before)
    cost_after = estimate_cost(MODEL_ID, input_tokens=tokens_after)

    # Verify findings
    verification = verify_response(response_text)
    findings_found = sum(verification.values())

    # Results
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    print(f"""
                              Without         With
                              Headroom        Headroom        Savings
    ─────────────────────────────────────────────────────────────────
    Input Tokens              {tokens_before:>8,}        {tokens_after:>8,}        {tokens_saved:,} ({pct_saved:.0f}%)""")

    if cost_before and cost_after:
        cost_saved = cost_before - cost_after
        print(
            f"    Input Cost                ${cost_before:.4f}         ${cost_after:.4f}         ${cost_saved:.4f}"
        )

    print(f"""
    Duration                                    {duration:.1f}s
    Tool calls                                  {stats["total_requests"]}

    Ground Truth Verification ({findings_found}/4 findings):""")

    for name, found in verification.items():
        print(f"      {'✓' if found else '✗'} {name.replace('_', ' ').title()}")

    print("\n" + "=" * 70)
    print("  AGENT RESPONSE (excerpt)")
    print("=" * 70)
    # Show first 2000 chars of response
    excerpt = response_text[:2000] + "..." if len(response_text) > 2000 else response_text
    print(excerpt)

    print("\n" + "=" * 70)
    if findings_found >= 3 and pct_saved > 30:
        print(f"  SUCCESS: {pct_saved:.0f}% compression with {findings_found}/4 findings preserved")
    elif findings_found >= 3:
        print(f"  ACCURACY OK: {findings_found}/4 findings, but only {pct_saved:.0f}% compression")
    else:
        print(f"  WARNING: Only {findings_found}/4 findings - compression may be too aggressive")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
