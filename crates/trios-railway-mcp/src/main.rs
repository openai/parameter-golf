//! `trios-railway-mcp` - public MCP server with dual transport.
//!
//! Anchor: `phi^2 + phi^-2 = 3`.
//!
//! Implements MCP (Model Context Protocol) over **both** Streamable HTTP
//! and legacy SSE transports using the official `rmcp` Rust SDK.
//! Exposes a typed control surface over the Railway-side helpers from
//! `trios-railway-core` so external agents (Claude Desktop, Cursor,
//! custom clients) can drive deploys against the IGLA project.
//!
//! Routes:
//!   GET  /sse      → SSE event stream (legacy transport)
//!   POST /message  → SSE client messages (legacy transport)
//!   POST /mcp      → Streamable HTTP (modern transport)
//!   GET  /healthz  → liveness probe
//!
//! Standing rules:
//!   R1 - Rust-only, no Python, no TypeScript.
//!   R5 - Honest passthrough of upstream errors via `CallToolResult`.
//!   R7 - Every mutation tool emits a `RailwayHash::seal` triplet to the
//!        local `.trinity/experience/<YYYYMMDD>.trinity` log.
//!   R9 - Destructive tools require explicit `confirm = true` argument.

mod tools;

use std::net::{Ipv6Addr, SocketAddr};
use std::time::Duration;

use anyhow::{Context, Result};
use rmcp::transport::streamable_http_server::{
    session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
};
use rmcp::transport::sse_server::SseServerConfig;
use tracing_subscriber::EnvFilter;

use crate::tools::TriosRailwayMcp;

#[tokio::main]
async fn main() -> Result<()> {
    // Pre-tracing diagnostic line so Railway captures *something* even if
    // the env filter swallows tracing output. R5 honesty: never silent.
    println!(
        "[trios-railway-mcp] boot: pid={}, exe ok",
        std::process::id()
    );

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_writer(std::io::stdout)
        .compact()
        .init();

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);
    // Bind on IPv6 unspecified (`[::]`) so the Railway private-network proxy
    // (IPv6-only) can reach us. Linux dual-stack also accepts IPv4 here.
    let addr: SocketAddr = SocketAddr::from((Ipv6Addr::UNSPECIFIED, port));

    tracing::info!(%addr, "trios-railway-mcp starting");
    println!("[trios-railway-mcp] binding to {addr}");

    // --- Streamable HTTP transport (modern) ---
    let streamable = StreamableHttpService::new(
        || Ok(TriosRailwayMcp::new()),
        LocalSessionManager::default().into(),
        StreamableHttpServerConfig::default(),
    );

    // --- SSE transport (legacy, for older clients) ---
    let sse_config = SseServerConfig {
        bind: addr,
        sse_path: "/sse".to_string(),
        post_path: "/message".to_string(),
        ct: tokio_util::sync::CancellationToken::new(),
        sse_keep_alive: Some(Duration::from_secs(15)),
    };
    let (sse_server, sse_router) =
        rmcp::transport::sse_server::SseServer::new(sse_config);
    let _sse_ct = sse_server.with_service(TriosRailwayMcp::new);

    // --- Combined router ---
    let router = axum::Router::new()
        .route("/", axum::routing::get(root_handler))
        .route("/healthz", axum::routing::get(health_handler))
        .nest_service("/mcp", streamable)
        .merge(sse_router);

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("bind {addr}"))?;

    tracing::info!("listening on http://{addr}/mcp (Streamable HTTP) + /sse (legacy SSE)");

    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
        .context("axum::serve")?;

    Ok(())
}

async fn root_handler() -> &'static str {
    "trios-railway-mcp: public MCP server for the IGLA project.\n\
     POST JSON-RPC to /mcp (Streamable HTTP) or connect to /sse (legacy SSE)\n"
}

async fn health_handler() -> &'static str {
    "ok"
}
