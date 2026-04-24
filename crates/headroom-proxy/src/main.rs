//! headroom-proxy: Rust proxy binary (Phase 0 scaffolding).
//!
//! Currently exposes only `/healthz`. Provider routes land in Phase 2.

use axum::{routing::get, Json, Router};
use serde_json::{json, Value};
use std::net::SocketAddr;

async fn healthz() -> Json<Value> {
    Json(json!({ "ok": true }))
}

fn app() -> Router {
    Router::new().route("/healthz", get(healthz))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber_init();
    let addr: SocketAddr = "127.0.0.1:8787".parse()?;
    tracing::info!(%addr, crate_hello = headroom_core::hello(), "starting headroom-proxy");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app()).await?;
    Ok(())
}

fn tracing_subscriber_init() {
    // Minimal no-op initializer so we don't pull tracing-subscriber in Phase 0.
    // Replace with tracing-subscriber in Phase 2 when richer logging is needed.
}

#[cfg(test)]
mod tests {
    use super::app;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    #[tokio::test]
    async fn healthz_returns_ok() {
        let response = app()
            .oneshot(
                Request::builder()
                    .uri("/healthz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), 1024)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value, serde_json::json!({"ok": true}));
    }
}
