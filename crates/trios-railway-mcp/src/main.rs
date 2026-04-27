//! `trios-railway-mcp` - public Streamable-HTTP MCP server.
//!
//! Anchor: `phi^2 + phi^-2 = 3`.
//!
//! Implements MCP (Model Context Protocol) over Streamable HTTP using
//! the official `rmcp` Rust SDK. Exposes a typed control surface over
//! the Railway-side helpers from `trios-railway-core` so external
//! agents (Claude Desktop, Cursor, custom clients) can drive deploys
//! against the IGLA project at `e4fe33bb-...`.
//!
//! Standing rules:
//!   R1 - Rust-only, no Python, no TypeScript.
//!   R5 - Honest passthrough of upstream errors via `CallToolResult`.
//!   R7 - Every mutation tool emits a `RailwayHash::seal` triplet to the
//!        local `.trinity/experience/<YYYYMMDD>.trinity` log.
//!   R9 - Destructive tools require explicit `confirm = true` argument.
//!
//! Transport: Streamable HTTP over `axum` at `/mcp`, listens on
//! `0.0.0.0:$PORT` (Railway convention).

mod tools;

use std::net::SocketAddr;

use anyhow::{Context, Result};
use rmcp::transport::streamable_http_server::{
    session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
};
use tracing_subscriber::EnvFilter;

use crate::tools::TriosRailwayMcp;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .compact()
        .init();

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);
    let addr: SocketAddr = SocketAddr::from(([0, 0, 0, 0], port));

    tracing::info!(%addr, "trios-railway-mcp starting");

    let mcp = StreamableHttpService::new(
        || Ok(TriosRailwayMcp::new()),
        LocalSessionManager::default().into(),
        StreamableHttpServerConfig::default(),
    );

    let router = axum::Router::new()
        .route("/", axum::routing::get(root_handler))
        .route("/healthz", axum::routing::get(health_handler))
        .nest_service("/mcp", mcp);

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("bind {addr}"))?;

    tracing::info!("listening on http://{addr}/mcp (Streamable HTTP)");

    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
        .context("axum::serve")?;

    Ok(())
}

async fn root_handler() -> &'static str {
    "trios-railway-mcp: public MCP server for the IGLA project. POST JSON-RPC to /mcp\n"
}

async fn health_handler() -> &'static str {
    "ok"
}
