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
use crucible_server::{
    router, AppState, JsonCuratedBank, MultiBank, StaticMathBank,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr: SocketAddr = env::var("CRUCIBLE_LISTEN")
        .unwrap_or_else(|_| "127.0.0.1:8090".to_owned())
        .parse()?;
    let base_path = env::var("CRUCIBLE_BASE_PATH").unwrap_or_else(|_| "/crucible".to_owned());

    let state = build_state()?;

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

/// Wire AppState with the default math bank + any curated
/// banks discovered under $CRUCIBLE_BANKS_DIR.
///
/// $CRUCIBLE_BANKS_DIR is an optional directory; if set, every
/// `*.json` file inside is loaded as a JsonCuratedBank and
/// registered for its declared `kind`. The discovered bank
/// OVERRIDES any default for that kind — so a curated math
/// bank in the dir would replace the StaticMathBank.
///
/// Walks one level deep only (non-recursive). Files that fail
/// to parse log a warning to stderr and are skipped; the
/// server continues with whichever banks DID load.
fn build_state() -> Result<std::sync::Arc<AppState>, Box<dyn std::error::Error>> {
    use crucible_core::AttributionPolicy;
    use std::sync::Arc;

    let mut multi = MultiBank::new().register(
        crucible_core::ChallengeKind::MathArithmetic,
        Box::new(StaticMathBank::default()),
    );

    if let Ok(banks_dir) = env::var("CRUCIBLE_BANKS_DIR") {
        let dir = PathBuf::from(&banks_dir);
        match std::fs::read_dir(&dir) {
            Ok(entries) => {
                let mut paths: Vec<PathBuf> = entries
                    .filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("json"))
                    .collect();
                paths.sort();
                for p in paths {
                    match JsonCuratedBank::from_path(&p) {
                        Ok(bank) => {
                            // Each curated bank carries its own kind; query it
                            // by peeking at the parsed JSON (no public getter on
                            // JsonCuratedBank yet, so the registration key comes
                            // from re-parsing the kind field).
                            let raw = std::fs::read_to_string(&p)?;
                            let v: serde_json::Value = serde_json::from_str(&raw)?;
                            let kind_str = v
                                .get("kind")
                                .and_then(|k| k.as_str())
                                .unwrap_or("");
                            let kind: crucible_core::ChallengeKind =
                                serde_json::from_value(serde_json::json!(kind_str))?;
                            multi = multi.register(kind, Box::new(bank));
                            eprintln!(
                                "crucible-serve: loaded curated bank for {kind_str:?} from {}",
                                p.display()
                            );
                        }
                        Err(e) => {
                            eprintln!(
                                "crucible-serve: skipped {} ({}): {}",
                                p.display(),
                                "bad bank",
                                e
                            );
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!(
                    "crucible-serve: CRUCIBLE_BANKS_DIR={banks_dir} unreadable: {e} \
                     (continuing with math-only)"
                );
            }
        }
    }

    Ok(Arc::new(AppState {
        pending: tokio::sync::RwLock::new(std::collections::HashMap::new()),
        captured: tokio::sync::RwLock::new(Vec::new()),
        registry: crucible_challenges::registry(),
        bank: Arc::new(multi),
        attribution: Arc::new(|_| AttributionPolicy::Curated),
    }))
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
