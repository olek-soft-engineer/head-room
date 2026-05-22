# Observability — proxy metrics

The Headroom Rust proxy exposes Prometheus-format metrics on the
`/metrics` endpoint of every running proxy instance. The metric
catalogue below covers Phase D (Bedrock route instrumentation) and
Phase G PR-G3 (per-invocation RTK + proxy-wide observability).

All metric names + label keys are constants in
`crates/headroom-proxy/src/observability/metric_names.rs`, so any
rename catches one file in code review.

## Metric catalogue

### Bedrock route (Phase D PR-D3)

| Name | Type | Labels | Purpose |
|------|------|--------|---------|
| `bedrock_invoke_count_total` | Counter | `model`, `region`, `auth_mode` | One increment per Bedrock `/invoke` or `/converse` request. |
| `bedrock_invoke_latency_seconds` | Histogram | `model`, `region` | Latency from proxy entry to upstream completion. Buckets target 50ms–60s. |
| `bedrock_eventstream_message_count_total` | Counter | `model`, `region`, `event_type` | One increment per parsed binary EventStream message. |

### Proxy-wide (Phase G PR-G3)

#### Cache + compression

| Name | Type | Labels | Purpose |
|------|------|--------|---------|
| `proxy_cache_hit_rate_per_session` | Histogram | `provider` | Per-session cache hit rate. **Phase H canary gate.** |
| `proxy_compression_ratio_by_strategy` | Histogram | `strategy`, `content_type` | `compressed_tokens / original_tokens` per shrunk block. |
| `proxy_compression_rejected_by_token_check_total` | Counter | `strategy` | Compressor ran but failed the shrink check. |

#### Cache-safety alarm

| Name | Type | Labels | Purpose |
|------|------|--------|---------|
| `proxy_passthrough_bytes_modified_total` | Counter | `path` | Bytes mutated on a passthrough path. **Must stay 0 outside the compression hot path** — any non-zero rate fires the cache-safety alarm. |

#### Upstream rate limits

| Name | Type | Labels | Purpose |
|------|------|--------|---------|
| `proxy_rate_limit_remaining_requests` | Gauge | `provider` | Last-seen remaining requests in the current window. |
| `proxy_rate_limit_remaining_tokens` | Gauge | `provider` | Last-seen remaining tokens in the current window. |
| `proxy_rate_limit_remaining_input_tokens` | Gauge | `provider` | Anthropic-only input-token bucket. |
| `proxy_rate_limit_remaining_output_tokens` | Gauge | `provider` | Anthropic-only output-token bucket. |

#### OpenAI Responses telemetry

| Name | Type | Labels | Purpose |
|------|------|--------|---------|
| `proxy_service_tier_count_total` | Counter | `tier` | Service-tier distribution observed at the proxy. |
| `proxy_response_status_count_total` | Counter | `status` | Terminal status distribution (`completed`, `incomplete`, `failed`, `cancelled`, `in_progress`). |

#### Wrap CLI / RTK

| Name | Type | Labels | Purpose |
|------|------|--------|---------|
| `wrap_rtk_invocations_total` | Counter | `tool` | RTK invocations observed via `rtk gain --format json` polling. Driven from the wrap-CLI tail. |

#### Image log redaction

| Name | Type | Labels | Purpose |
|------|------|--------|---------|
| `proxy_image_generation_call_log_redacted_total` | Counter | _none_ | Base64-encoded image payloads redacted from request logs. |

## How to query

The proxy renders Prometheus text-format on `GET /metrics`:

```bash
curl -s http://127.0.0.1:8787/metrics
```

Common PromQL queries:

```promql
# Phase H canary gate — cache hit rate parity vs Python proxy.
histogram_quantile(0.50, sum by (le) (rate(proxy_cache_hit_rate_per_session_bucket[5m])))

# Cache-safety alarm. Should always be 0.
sum(rate(proxy_passthrough_bytes_modified_total[5m]))

# Aggregate compression value by strategy.
histogram_quantile(0.50, sum by (strategy, le) (rate(proxy_compression_ratio_by_strategy_bucket[1h])))

# Upstream rate-limit headroom (smaller = closer to throttle).
proxy_rate_limit_remaining_tokens{provider="anthropic"}
```

## Wiring

Every metric registration is `OnceLock`-backed and lazy: the first
call to a `*_counter()` / `*_gauge()` / `*_histogram()` helper
registers the family with the shared registry. `handle_metrics`
force-touches every Phase G PR-G3 family before scraping so the
descriptors are reachable from `/metrics` on a fresh boot — note that
the `prometheus` crate v0.13 skips empty MetricVecs from `gather()`,
so HELP/TYPE only surface once a family has been incremented at
least once.

## Cardinality discipline

Every label vocabulary is bounded by code, not customer input:

- `model` / `region`: read from path params + `Config::bedrock_region`.
- `auth_mode`: 3-variant enum (`payg`, `oauth`, `subscription`).
- `provider`: 3 values (`anthropic`, `openai_chat`, `openai_responses`).
- `strategy`: `&'static str` from the compressor's `BlockAction::Compressed`.
- `content_type`: `&'static str` from `headroom_core::transforms::ContentType`.
- `tier`: bounded by OpenAI Responses spec.
- `status`: 5-variant enum.

There is no code path where a malicious client can drive label
cardinality unbounded.

## See also

- `docs/rtk-architecture.md` — why RTK lives wrap-side, not proxy-side.
- `crates/headroom-proxy/src/observability/` — implementation.
- `REALIGNMENT/09-phase-G-rtk-observability.md` — spec.
- `REALIGNMENT/10-phase-H-python-retirement.md` — H1 acceptance gate.
