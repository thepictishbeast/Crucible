//! `crucible-serve` — runnable binary wrapping the
//! crucible-server library.
//!
//! Listens on $CRUCIBLE_LISTEN (default 127.0.0.1:8090) and
//! mounts the crucible router under $CRUCIBLE_BASE_PATH
//! (default `/crucible`). Configured via env vars only — keeps
//! the binary embeddable behind any reverse proxy that sets
//! standard envs (systemd unit, docker -e, etc).
//!
//! State is the default math-bank + curated-for-everyone
//! attribution. Hosts wanting per-tenant policy or non-math
//! banks should construct AppState directly from the library
//! API rather than running this bin.
//!
//! Captured tuples accumulate in process memory; flush them
//! to disk on shutdown OR via the periodic flush task
//! (CRUCIBLE_FLUSH_DIR set → write_corpus_dir every
//! CRUCIBLE_FLUSH_SECS, default 300).

use std::env;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use crucible_server::{router, AppState};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr: SocketAddr = env::var("CRUCIBLE_LISTEN")
        .unwrap_or_else(|_| "127.0.0.1:8090".to_owned())
        .parse()?;
    let base_path = env::var("CRUCIBLE_BASE_PATH").unwrap_or_else(|_| "/crucible".to_owned());

    let state = AppState::with_math_bank();

    // Optional periodic corpus flush.
    if let Ok(flush_dir) = env::var("CRUCIBLE_FLUSH_DIR") {
        let flush_secs: u64 = env::var("CRUCIBLE_FLUSH_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(300);
        spawn_flusher(state.clone(), PathBuf::from(flush_dir), flush_secs);
    }

    let app = Router::new().nest(&base_path, router(state.clone()));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    eprintln!(
        "crucible-serve listening on http://{addr}{base_path} \
         (POST /challenge, POST /solve)"
    );
    axum::serve(listener, app).await?;
    Ok(())
}

/// Periodic background task: every `flush_secs`, drain captured
/// tuples + write them to a fresh subdirectory of `flush_dir`.
/// Errors are logged to stderr and do NOT terminate the server
/// — the corpus pipeline is best-effort.
fn spawn_flusher(state: Arc<AppState>, dir: PathBuf, flush_secs: u64) {
    tokio::spawn(async move {
        let interval = Duration::from_secs(flush_secs);
        loop {
            tokio::time::sleep(interval).await;
            let captured = state.drain_captured().await;
            if captured.is_empty() {
                continue;
            }
            let patterns: Vec<crucible_corpus::CorpusPattern> = captured
                .iter()
                .filter_map(|t| crucible_corpus::to_pattern(t).ok())
                .collect();
            if patterns.is_empty() {
                continue;
            }
            // Each flush gets its own timestamped subdir so multiple
            // flush cycles never overwrite earlier ones.
            let ts = time::OffsetDateTime::now_utc()
                .format(&time::format_description::well_known::Rfc3339)
                .unwrap_or_else(|_| "unknown".into())
                .replace(':', "");
            let target = dir.join(format!("flush-{ts}"));
            match crucible_corpus::write_corpus_dir(&patterns, &target) {
                Ok(manifest) => {
                    eprintln!(
                        "crucible-serve: flushed {} patterns to {}",
                        manifest.patterns.len(),
                        target.display()
                    );
                }
                Err(e) => {
                    eprintln!(
                        "crucible-serve: flush failed ({}): {}",
                        target.display(),
                        e
                    );
                }
            }
        }
    });
}
