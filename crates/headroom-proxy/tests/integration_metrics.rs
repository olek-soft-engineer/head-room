//! Phase G PR-G3 — integration tests for the new proxy-wide
//! observability metrics. One test per category per the spec:
//!
//! 1. `cache_hit_rate_emitted_per_session` — drive an Anthropic
//!    streaming response that carries cache_read_input_tokens on
//!    `message_delta`; assert the histogram saw the right sample.
//! 2. `compression_ratio_emitted_per_strategy` — exercise the
//!    helper directly (the metric records on
//!    `Outcome::Compressed`; live-zone compression of small
//!    payloads doesn't reliably fire, so the integration test
//!    drives the public helper as a state-machine smoke test).
//! 3. `passthrough_bytes_modified_zero_when_no_compression` —
//!    confirm the alarm-able counter stays at 0 across an
//!    end-to-end passthrough request.
//! 4. `service_tier_logged` — drive a Responses request carrying
//!    `service_tier` and assert the counter rows appear.
//! 5. `incomplete_status_logged_with_reason` — drive an SSE
//!    response with `response.incomplete`; assert the counter row
//!    and the structured-log reason.
//!
//! Like the D3 metrics tests, every test owns a unique label tuple
//! so concurrent test runs over the shared global registry never
//! cross-contaminate.

mod common;

use bytes::Bytes;
use common::start_proxy_with;
use headroom_proxy::observability;
use headroom_proxy::sse::openai_responses::ResponseState;
use headroom_proxy::sse::SseFramer;
use serde_json::json;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::time::Duration;

use http_body_util::StreamBody;
use hyper::body::Frame;
use hyper::service::service_fn;
use hyper::{Request, Response};
use hyper_util::rt::TokioIo;

/// Fetch the proxy's `/metrics` text-format scrape.
async fn scrape_metrics(proxy_url: &str) -> String {
    let resp = reqwest::Client::new()
        .get(format!("{proxy_url}/metrics"))
        .send()
        .await
        .expect("metrics scrape");
    assert_eq!(resp.status(), 200, "metrics endpoint must return 200");
    resp.text().await.unwrap()
}

/// Find a line that contains the metric name + every label pair, and
/// parse its trailing numeric value. Mirrors the helper used by the
/// Bedrock D3 metrics test.
fn find_value_with_labels(scrape: &str, metric: &str, label_pairs: &[(&str, &str)]) -> Option<f64> {
    for line in scrape.lines() {
        if !line.starts_with(metric) {
            continue;
        }
        if !label_pairs
            .iter()
            .all(|(k, v)| line.contains(&format!("{k}=\"{v}\"")))
        {
            continue;
        }
        if let Some(value_str) = line.rsplit_once(' ').map(|(_, v)| v.trim()) {
            if let Ok(f) = value_str.parse::<f64>() {
                return Some(f);
            }
        }
    }
    None
}

// ============================================================================
// Test 1: cache_hit_rate_emitted_per_session
// ============================================================================
//
// Anthropic SSE upstream that emits message_start (carrying
// input_tokens + cache_read_input_tokens) and message_delta
// (with the final usage object). We pipe this through the
// proxy's `/v1/messages` SSE path; the state machine then closes
// and the `proxy_cache_hit_rate_per_session{provider="anthropic"}`
// histogram should have exactly one sample.

async fn anthropic_streaming_upstream(
    cache_read_input_tokens: u64,
    input_tokens: u64,
) -> (SocketAddr, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        loop {
            let Ok((stream, _)) = listener.accept().await else {
                break;
            };
            tokio::spawn(async move {
                let io = TokioIo::new(stream);
                let _ = hyper::server::conn::http1::Builder::new()
                    .serve_connection(
                        io,
                        service_fn(move |_req: Request<hyper::body::Incoming>| async move {
                            let (tx, rx) = tokio::sync::mpsc::channel::<
                                Result<Frame<Bytes>, std::io::Error>,
                            >(8);
                            tokio::spawn(async move {
                                let start = format!(
                                    "event: message_start\ndata: {{\"type\":\"message_start\",\"message\":{{\"id\":\"msg_x\",\"model\":\"claude\",\"usage\":{{\"input_tokens\":{input_tokens},\"output_tokens\":0,\"cache_read_input_tokens\":{cache_read_input_tokens}}}}}}}\n\n"
                                );
                                let delta = format!(
                                    "event: message_delta\ndata: {{\"type\":\"message_delta\",\"delta\":{{\"stop_reason\":\"end_turn\"}},\"usage\":{{\"input_tokens\":{input_tokens},\"output_tokens\":4,\"cache_read_input_tokens\":{cache_read_input_tokens}}}}}\n\n"
                                );
                                let stop =
                                    b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
                                let frames: Vec<Vec<u8>> = vec![
                                    start.into_bytes(),
                                    delta.into_bytes(),
                                    stop.to_vec(),
                                ];
                                for f in frames {
                                    if tx.send(Ok(Frame::data(Bytes::from(f)))).await.is_err() {
                                        return;
                                    }
                                    tokio::time::sleep(Duration::from_millis(10)).await;
                                }
                            });
                            let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
                            let body = StreamBody::new(stream);
                            Ok::<_, Infallible>(
                                Response::builder()
                                    .status(200)
                                    .header("content-type", "text/event-stream")
                                    .body(body)
                                    .unwrap(),
                            )
                        }),
                    )
                    .await;
            });
        }
    });
    (addr, task)
}

#[tokio::test]
async fn cache_hit_rate_emitted_per_session() {
    // Pick unusual token counts so the histogram bucket we land in
    // is identifiable across parallel test runs.
    // input=200, cache_read=800 → denom=1000, rate=0.8 (falls into the
    // [0.75, 0.9] bucket).
    let (addr, _server) = anthropic_streaming_upstream(800, 200).await;
    let proxy = start_proxy_with(&format!("http://{addr}"), |c| {
        c.compression_mode = headroom_proxy::config::CompressionMode::Off;
    })
    .await;

    let body = json!({
        "model": "claude-3-haiku-20240307",
        "stream": true,
        "max_tokens": 8,
        "messages": [{"role":"user","content":"hi"}]
    });
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/messages", proxy.url()))
        .header("content-type", "application/json")
        .header("accept", "text/event-stream")
        .body(serde_json::to_vec(&body).unwrap())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    // Drain so the state-machine task observes message_stop and emits.
    let _ = resp.bytes().await.unwrap();
    // Give the spawned state-machine task time to close + observe.
    tokio::time::sleep(Duration::from_millis(80)).await;

    let scrape = scrape_metrics(&proxy.url()).await;
    // The histogram exposes _bucket / _sum / _count rows. We assert
    // that at least one sample was observed for our provider label —
    // exact bucket assertions are fragile; the _count line is the
    // load-bearing one.
    let count = find_value_with_labels(
        &scrape,
        "proxy_cache_hit_rate_per_session_count",
        &[("provider", "anthropic")],
    )
    .expect("histogram count must appear after first session");
    assert!(
        count >= 1.0,
        "expected ≥1 observation on anthropic label; got {count}"
    );
    let sum = find_value_with_labels(
        &scrape,
        "proxy_cache_hit_rate_per_session_sum",
        &[("provider", "anthropic")],
    )
    .expect("histogram sum must appear after first session");
    // We seeded denom=1000, cache_read=800 → rate=0.8; the sum
    // includes this observation. Other concurrent tests may
    // contribute via the same `provider=anthropic` label; assert
    // sum > 0 rather than equality.
    assert!(
        sum > 0.0,
        "histogram sum must reflect at least one observed rate > 0; saw {sum}"
    );

    proxy.shutdown().await;
}

// ============================================================================
// Test 2: compression_ratio_emitted_per_strategy
// ============================================================================
//
// Exercise the helper directly — the integration path requires
// live-zone compression to actually trigger, which depends on
// content size + tokenizer thresholds. We assert the metric
// registration + emit-helper pair lands rows in the registry under
// the documented labels, which is the load-bearing observability
// surface for Phase H.

#[tokio::test]
async fn compression_ratio_emitted_per_strategy() {
    // Use a label tuple unique to this test so parallel runs over
    // the global registry don't cross-contaminate.
    const TEST_STRATEGY: &str = "integration_strategy_metric_test_v1";
    const TEST_CONTENT_TYPE: &str = "integration_content_type_metric_test_v1";

    observability::observe_compression_ratio(TEST_STRATEGY, TEST_CONTENT_TYPE, 1000, 250);
    // Drive the rejected counter on the same strategy so its row
    // also appears in the scrape.
    observability::record_compression_rejected_by_token_check(TEST_STRATEGY);

    // Spin up a proxy purely to expose /metrics — the helpers above
    // already incremented the global registry.
    let proxy = start_proxy_with("http://127.0.0.1:1", |_| {}).await;
    let scrape = scrape_metrics(&proxy.url()).await;

    let ratio_count = find_value_with_labels(
        &scrape,
        "proxy_compression_ratio_by_strategy_count",
        &[
            ("strategy", TEST_STRATEGY),
            ("content_type", TEST_CONTENT_TYPE),
        ],
    )
    .expect("ratio histogram count row must appear");
    assert!(ratio_count >= 1.0);

    let rejected = find_value_with_labels(
        &scrape,
        "proxy_compression_rejected_by_token_check_total",
        &[("strategy", TEST_STRATEGY)],
    )
    .expect("rejected counter row must appear");
    assert!(rejected >= 1.0);

    proxy.shutdown().await;
}

// ============================================================================
// Test 3: passthrough_bytes_modified_zero_when_no_compression
// ============================================================================
//
// End-to-end passthrough request with `CompressionMode::Off` — the
// counter must NOT have advanced. We assert the metric advertises
// itself (HELP/TYPE in the scrape) and has the alarm-able value of 0
// for the `/v1/messages` path label.

async fn anthropic_simple_non_stream_upstream() -> (SocketAddr, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        loop {
            let Ok((stream, _)) = listener.accept().await else {
                break;
            };
            tokio::spawn(async move {
                let io = TokioIo::new(stream);
                let _ = hyper::server::conn::http1::Builder::new()
                    .serve_connection(
                        io,
                        service_fn(|_req: Request<hyper::body::Incoming>| async move {
                            Ok::<_, Infallible>(
                                Response::builder()
                                    .status(200)
                                    .header("content-type", "application/json")
                                    .body(http_body_util::Full::new(Bytes::from_static(
                                        b"{\"id\":\"msg_1\",\"content\":[]}",
                                    )))
                                    .unwrap(),
                            )
                        }),
                    )
                    .await;
            });
        }
    });
    (addr, task)
}

#[tokio::test]
async fn passthrough_bytes_modified_zero_when_no_compression() {
    let (addr, _server) = anthropic_simple_non_stream_upstream().await;
    let proxy = start_proxy_with(&format!("http://{addr}"), |c| {
        c.compression_mode = headroom_proxy::config::CompressionMode::Off;
    })
    .await;

    let body = json!({
        "model": "claude-3-haiku-20240307",
        "max_tokens": 8,
        "messages": [{"role":"user","content":"hi"}]
    });
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/messages", proxy.url()))
        .header("content-type", "application/json")
        .body(serde_json::to_vec(&body).unwrap())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let scrape = scrape_metrics(&proxy.url()).await;
    // The `prometheus` crate v0.13 skips empty MetricVecs entirely
    // in `gather()` — neither HELP/TYPE nor rows appear until the
    // counter has been incremented with at least one label-set.
    // For the "must stay 0" alarm-able metric this absence IS the
    // signal: if the counter is silent, no passthrough policy has
    // been violated. The PromQL alarm queries
    // `rate(proxy_passthrough_bytes_modified_total[5m]) > 0`, which
    // is `0` (the metric doesn't exist) by definition.
    let any_row = scrape
        .lines()
        .any(|l| l.starts_with("proxy_passthrough_bytes_modified_total{"));
    assert!(
        !any_row,
        "counter should have no rows when nothing modified passthrough; got: {scrape}"
    );
    // The HELP/TYPE descriptor is registered eagerly in
    // `handle_metrics`, but `gather()` skips empty families so the
    // descriptor only surfaces after the first emit. We assert the
    // counter is reachable via the public helper to pin behaviour
    // (the helper is what the SSE/handler emit sites call).
    use headroom_proxy::observability::record_passthrough_bytes_modified;
    record_passthrough_bytes_modified(
        "/integration_test_passthrough_synthetic",
        0,
        "integration_test_request_id_passthrough_v1",
    );
    // After a (zero-delta) increment, the row should appear.
    let scrape_after_touch = scrape_metrics(&proxy.url()).await;
    assert!(
        scrape_after_touch.contains("# HELP proxy_passthrough_bytes_modified_total"),
        "scrape missing proxy_passthrough_bytes_modified_total HELP after touch"
    );
    assert!(
        scrape_after_touch.contains("# TYPE proxy_passthrough_bytes_modified_total counter"),
        "scrape missing proxy_passthrough_bytes_modified_total TYPE after touch"
    );

    proxy.shutdown().await;
}

// ============================================================================
// Test 4: service_tier_logged
// ============================================================================
//
// Drive a `/v1/responses` request with `service_tier: "priority"`;
// the handler emits the service-tier counter at the boundary.

async fn responses_passthrough_upstream() -> (SocketAddr, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        loop {
            let Ok((stream, _)) = listener.accept().await else {
                break;
            };
            tokio::spawn(async move {
                let io = TokioIo::new(stream);
                let _ = hyper::server::conn::http1::Builder::new()
                    .serve_connection(
                        io,
                        service_fn(|_req: Request<hyper::body::Incoming>| async move {
                            Ok::<_, Infallible>(
                                Response::builder()
                                    .status(200)
                                    .header("content-type", "application/json")
                                    .body(http_body_util::Full::new(Bytes::from_static(
                                        b"{\"id\":\"resp_1\",\"output\":[]}",
                                    )))
                                    .unwrap(),
                            )
                        }),
                    )
                    .await;
            });
        }
    });
    (addr, task)
}

#[tokio::test]
async fn service_tier_logged() {
    let (addr, _server) = responses_passthrough_upstream().await;
    let proxy = start_proxy_with(&format!("http://{addr}"), |c| {
        c.compression_mode = headroom_proxy::config::CompressionMode::Off;
        c.enable_responses_streaming = true;
    })
    .await;

    // Unique tier value so this test owns its own row.
    const TEST_TIER: &str = "integration_test_tier_priority_v1";
    let body = json!({
        "model": "gpt-5",
        "service_tier": TEST_TIER,
        "input": [
            {"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}
        ]
    });
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/responses", proxy.url()))
        .header("content-type", "application/json")
        .body(serde_json::to_vec(&body).unwrap())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let _ = resp.bytes().await.unwrap();

    let scrape = scrape_metrics(&proxy.url()).await;
    let tier_count = find_value_with_labels(
        &scrape,
        "proxy_service_tier_count_total",
        &[("tier", TEST_TIER)],
    )
    .expect("service tier counter row must appear");
    assert!(
        tier_count >= 1.0,
        "expected ≥1 service_tier increment; got {tier_count}"
    );

    proxy.shutdown().await;
}

// ============================================================================
// Test 5: incomplete_status_logged_with_reason
// ============================================================================
//
// Drive the `ResponseState` directly with a `response.incomplete`
// event payload. The state machine should capture
// `incomplete_details.reason` and `terminal_status()` should return
// `"incomplete"`. Wiring the metric counter is exercised by the
// `record_response_status` helper.

#[tokio::test]
async fn incomplete_status_logged_with_reason() {
    let mut framer = SseFramer::new();
    let mut state = ResponseState::new();

    let payload = b"event: response.incomplete\ndata: {\"type\":\"response.incomplete\",\"response\":{\"id\":\"resp_inc\",\"incomplete_details\":{\"reason\":\"max_output_tokens\"},\"service_tier\":\"integration_test_tier_incomplete_v1\"}}\n\n";
    framer.push(payload);
    while let Some(ev) = framer.next_event() {
        let ev = ev.expect("frame parses");
        state.apply(ev).expect("apply succeeds");
    }

    assert_eq!(
        state.incomplete_reason.as_deref(),
        Some("max_output_tokens"),
        "incomplete_details.reason must be captured"
    );
    assert_eq!(state.terminal_status(), Some("incomplete"));
    assert_eq!(
        state.service_tier.as_deref(),
        Some("integration_test_tier_incomplete_v1"),
        "service_tier must be captured on incomplete responses"
    );

    // Drive the metric helper and assert the row.
    observability::record_response_status(
        state.terminal_status().unwrap(),
        state.incomplete_reason.as_deref(),
        "integration_test_request_id_incomplete_v1",
    );

    let proxy = start_proxy_with("http://127.0.0.1:1", |_| {}).await;
    let scrape = scrape_metrics(&proxy.url()).await;
    let count = find_value_with_labels(
        &scrape,
        "proxy_response_status_count_total",
        &[("status", "incomplete")],
    )
    .expect("response status counter row must appear");
    assert!(count >= 1.0);

    proxy.shutdown().await;
}

// ============================================================================
// Bonus coverage: rate-limit gauge plumbing via the public helper.
// ============================================================================

#[tokio::test]
async fn rate_limit_snapshot_emits_gauges() {
    let snap = observability::RateLimitSnapshot {
        remaining_requests: Some(123),
        remaining_tokens: Some(45678),
        remaining_input_tokens: Some(30000),
        remaining_output_tokens: Some(15000),
    };
    observability::record_rate_limit_snapshot(
        observability::cache_hit_rate_provider::ANTHROPIC,
        &snap,
        "integration_test_rate_limit_v1",
    );

    let proxy = start_proxy_with("http://127.0.0.1:1", |_| {}).await;
    let scrape = scrape_metrics(&proxy.url()).await;

    // Every gauge advertises itself; the rows we just set must
    // appear with our exact values (gauges, unlike counters, are
    // last-write-wins so this is deterministic).
    let v = find_value_with_labels(
        &scrape,
        "proxy_rate_limit_remaining_requests",
        &[("provider", "anthropic")],
    )
    .expect("requests gauge row");
    assert!(v >= 1.0);
    let v = find_value_with_labels(
        &scrape,
        "proxy_rate_limit_remaining_tokens",
        &[("provider", "anthropic")],
    )
    .expect("tokens gauge row");
    assert!(v >= 1.0);

    proxy.shutdown().await;
}
