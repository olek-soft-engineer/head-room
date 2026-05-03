//! POST `/v1/responses` handler — Phase C PR-C3.
//!
//! # Why an explicit handler?
//!
//! The Python proxy currently flattens Responses-shape items into
//! Chat-Completions-shape via
//! `headroom/proxy/responses_converter.py` — a fragile shim that
//! silently breaks every time OpenAI lands a new item type. C3 ports
//! this path to Rust with first-class per-item-type handling.
//!
//! The handler buffers the request body (so the live-zone dispatcher
//! can inspect it) and re-injects it into [`crate::proxy::forward_http`].
//! `forward_http`'s compression gate dispatches on the path
//! classification (`CompressibleEndpoint::OpenAiResponses`) added by
//! this PR.
//!
//! # Streaming
//!
//! When the request carries `Accept: text/event-stream`, this handler
//! defers to PR-C4's streaming wiring. For now we forward
//! byte-for-byte and emit
//! `event = responses_streaming_passthrough_until_c4` so we can
//! measure the volume in production. C4 wires the
//! [`crate::sse::openai_responses::ResponseState`] machine PR-C1
//! shipped.
//!
//! # Per-item-type behaviour
//!
//! See [`crate::responses_items`] for the typed enum. Briefly:
//!
//! - `function_call_output` / `local_shell_call_output` /
//!   `apply_patch_call_output` — output strings are eligible for
//!   live-zone compression when the latest of each kind, above the
//!   2 KiB output-item floor.
//! - `message` (user role) — text content is eligible.
//! - `reasoning.encrypted_content`, `compaction.*`, MCP / computer /
//!   web-search / file-search / code-interpreter / image-generation /
//!   tool-search / custom-tool calls — passthrough byte-equal.
//! - `function_call.arguments` is a STRING the model emitted; never
//!   parsed by the proxy.
//! - `local_shell_call.action.command` is an argv array; never
//!   joined into a string.
//! - `apply_patch_call.operation.diff` is a V4A diff payload; never
//!   re-serialized.
//! - Unknown `type` values log
//!   `event = responses_unknown_item_type` at warn level and pass
//!   through verbatim.

use axum::body::Body;
use axum::extract::{ConnectInfo, State};
use axum::http::{HeaderMap, Method, Request, Uri};
use axum::response::Response;
use bytes::Bytes;
use std::net::SocketAddr;

use crate::proxy::{forward_http, AppState};

/// Axum POST handler for `/v1/responses`. Buffers the body, stitches
/// a fresh `Request<Body>` together, and forwards via
/// [`forward_http`]. Compression dispatch + SSE telemetry is handled
/// inside `forward_http`'s shared gate (PR-C1 + PR-C2 + PR-C3).
pub async fn handle_responses(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    // Streaming detection: when the client asks for SSE, log the
    // volume so we can plan the C4 cut-over. The body still flows
    // through `forward_http` — compression is gated by content-type
    // (application/json) and SSE responses are streamed back via the
    // existing tee in `forward_http` (already wired for the
    // `OpenAIResponsesStreamState` parser by PR-C1).
    if accepts_sse(&headers) {
        tracing::warn!(
            event = "responses_streaming_passthrough_until_c4",
            method = %method,
            path = %uri.path(),
            "/v1/responses called with Accept: text/event-stream — \
             passthrough until PR-C4 wires the streaming state machine"
        );
    }

    // Reconstruct the Request<Body> shape forward_http expects.
    let mut builder = Request::builder().method(method).uri(uri);
    if let Some(hs) = builder.headers_mut() {
        *hs = headers;
    }
    let req = match builder.body(Body::from(body)) {
        Ok(r) => r,
        Err(e) => {
            tracing::error!(
                event = "handler_error",
                handler = "responses",
                error = %e,
                "failed to reconstruct request from buffered body"
            );
            return Response::builder()
                .status(http::StatusCode::INTERNAL_SERVER_ERROR)
                .body(Body::from("internal handler error"))
                .expect("static response");
        }
    };

    forward_http(state, client_addr, req)
        .await
        .unwrap_or_else(|e| {
            use axum::response::IntoResponse;
            e.into_response()
        })
}

/// Cheap check: is this request asking for an SSE response? Compares
/// `Accept` against `text/event-stream` (case-insensitive on the
/// media-type token, RFC 7231 §3.1.1.1). Multiple media types in
/// `Accept` are split on `,`; any match wins.
fn accepts_sse(headers: &HeaderMap) -> bool {
    let Some(v) = headers.get(http::header::ACCEPT) else {
        return false;
    };
    let Ok(s) = v.to_str() else {
        return false;
    };
    s.split(',').any(|piece| {
        let mt = piece.split(';').next().unwrap_or("").trim();
        mt.eq_ignore_ascii_case("text/event-stream")
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use http::HeaderValue;

    #[test]
    fn accepts_sse_explicit() {
        let mut h = HeaderMap::new();
        h.insert(
            http::header::ACCEPT,
            HeaderValue::from_static("text/event-stream"),
        );
        assert!(accepts_sse(&h));
    }

    #[test]
    fn accepts_sse_case_insensitive() {
        let mut h = HeaderMap::new();
        h.insert(
            http::header::ACCEPT,
            HeaderValue::from_static("Text/Event-Stream"),
        );
        assert!(accepts_sse(&h));
    }

    #[test]
    fn accepts_sse_among_others() {
        let mut h = HeaderMap::new();
        h.insert(
            http::header::ACCEPT,
            HeaderValue::from_static("application/json, text/event-stream;q=0.9"),
        );
        assert!(accepts_sse(&h));
    }

    #[test]
    fn accepts_json_only_returns_false() {
        let mut h = HeaderMap::new();
        h.insert(
            http::header::ACCEPT,
            HeaderValue::from_static("application/json"),
        );
        assert!(!accepts_sse(&h));
    }

    #[test]
    fn no_accept_header_returns_false() {
        let h = HeaderMap::new();
        assert!(!accepts_sse(&h));
    }
}
