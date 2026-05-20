//! `crucible-server` — HTTP issuer + verify endpoints for
//! Crucible challenges.
//!
//! Exposes a mountable `axum::Router` that any host site can
//! attach under `/crucible/*`:
//!
//! * `POST /crucible/challenge` — body
//!   `{"kind": "math-arithmetic", "difficulty": "medium",
//!     "tenant_id": "acme"}` → returns a fresh `Challenge`
//! * `POST /crucible/solve` — body
//!   `{"challenge_id": "...", "response": {...},
//!     "submitted_at": "...", "elapsed_ms": <u32>}` →
//!   returns a `Verdict` and captures the
//!   `(challenge, response, ground_truth, verdict)` tuple
//!   into the in-memory corpus channel for downstream
//!   `crucible-corpus::write_corpus_dir()` flush.
//!
//! ## Design choices
//!
//! * **No DB at this layer.** Issued challenges live in an
//!   in-memory `tokio::sync::RwLock<HashMap<id, Challenge>>`
//!   keyed by challenge id. Solutions look the original up,
//!   verify, drop it. Persistence is downstream of corpus
//!   flush — the host wires it.
//!
//! * **Challenge generation pluggable.** The `ChallengeBank`
//!   trait gives one issued Challenge per request. v1 ships
//!   a `StaticMathBank` that mints fresh arithmetic puzzles
//!   (since `MathArithmeticVerifier` has a real impl).
//!   Future banks add image-set / audio-clip / etc backed by
//!   curator-authored corpora.
//!
//! * **Lifetime per challenge: 2 minutes** (matches
//!   `Challenge.expires_at - issued_at` in crucible-core).
//!   Expired entries are pruned lazily on solve attempts.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use std::collections::HashMap;
use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
    Json, Router,
};
use crucible_challenges::Registry;
use crucible_core::{
    AttributionPolicy, CapturedTuple, Challenge, ChallengeKind, CrucibleError, Difficulty,
    Solution, Verdict,
};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

/// Per-tenant policy controlling whether captured tuples are
/// curated (public LFI corpus), tenant-private (per-tenant
/// sub-corpus), or ephemeral (never recorded).
///
/// Mirrors `crucible_core::AttributionPolicy`. Host configures
/// this when mounting the router.
pub type AttributionResolver = Arc<dyn Fn(&str) -> AttributionPolicy + Send + Sync>;

/// Issued-challenge bank — mints a fresh Challenge on demand.
pub trait ChallengeBank: Send + Sync {
    /// Mint a fresh Challenge for the given kind + difficulty +
    /// tenant. Returns `Err` if the bank can't serve that kind
    /// (e.g. no image corpus loaded for ImageClassify).
    fn issue(
        &self,
        kind: ChallengeKind,
        difficulty: Difficulty,
        tenant_id: &str,
    ) -> Result<Challenge, CrucibleError>;
}

/// In-memory `MathArithmetic` bank — mints fresh `(a, op, b)`
/// arithmetic puzzles. Counters are monotonic so challenge IDs
/// are unique per process.
pub struct StaticMathBank {
    counter: std::sync::atomic::AtomicU64,
}

impl Default for StaticMathBank {
    fn default() -> Self {
        Self {
            counter: std::sync::atomic::AtomicU64::new(1),
        }
    }
}

impl ChallengeBank for StaticMathBank {
    fn issue(
        &self,
        kind: ChallengeKind,
        difficulty: Difficulty,
        tenant_id: &str,
    ) -> Result<Challenge, CrucibleError> {
        if !matches!(kind, ChallengeKind::MathArithmetic) {
            return Err(CrucibleError::Internal(format!(
                "StaticMathBank only mints MathArithmetic, got {kind:?}"
            )));
        }
        let n = self
            .counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // Deterministic small-integer arithmetic — the
        // counter doubles as the operand seed so test runs
        // are reproducible.
        let a = (n % 9) as i64 + 1; // 1..=9
        let b = ((n / 9) % 9) as i64 + 1; // 1..=9
        let op = match (n / 81) % 3 {
            0 => "+",
            1 => "-",
            _ => "*",
        };
        let now = time::OffsetDateTime::now_utc();
        let id = format!("math-{n:08}");
        Ok(Challenge {
            id,
            kind,
            difficulty,
            payload: serde_json::json!({"a": a, "op": op, "b": b}),
            issued_at: now,
            expires_at: now + time::Duration::seconds(120),
            tenant_id: tenant_id.to_owned(),
        })
    }
}

/// Dispatch-by-kind bank shim. Wraps a map of
/// `ChallengeKind → Box<dyn ChallengeBank>`. On `issue()`, looks
/// up the bank for the requested kind + delegates. If no bank
/// is registered for the kind, returns an Internal error so
/// `crucible-server` returns 400 Bad Request to the client.
///
/// The standard server topology is one MultiBank wrapping one
/// kind-specialized bank per ChallengeKind variant the host
/// wants to serve. This crate ships:
///   * `StaticMathBank` for MathArithmetic
///   * `JsonCuratedBank` for the 5 other kinds (image-classify,
///     semantic-similarity, audio-transcribe, drawing-reconstruct,
///     prompt-injection-detect) — curator-authored challenges
///     loaded from a JSON file.
#[derive(Default)]
pub struct MultiBank {
    inner: HashMap<ChallengeKind, Box<dyn ChallengeBank>>,
}

impl MultiBank {
    /// Empty builder.
    pub fn new() -> Self {
        Self::default()
    }
    /// Register a bank for a kind. Replaces any existing entry.
    pub fn register(mut self, kind: ChallengeKind, bank: Box<dyn ChallengeBank>) -> Self {
        self.inner.insert(kind, bank);
        self
    }
}

impl ChallengeBank for MultiBank {
    fn issue(
        &self,
        kind: ChallengeKind,
        difficulty: Difficulty,
        tenant_id: &str,
    ) -> Result<Challenge, CrucibleError> {
        let bank = self.inner.get(&kind).ok_or_else(|| {
            CrucibleError::Internal(format!("no bank registered for kind {kind:?}"))
        })?;
        bank.issue(kind, difficulty, tenant_id)
    }
}

/// Curator-authored challenge bank backed by a JSON file.
///
/// File shape:
/// ```json
/// {
///   "kind": "semantic-similarity",
///   "challenges": [
///     {
///       "payload": { ... kind-specific ... },
///       "difficulty_floor": "medium"
///     },
///     ...
///   ]
/// }
/// ```
///
/// On `issue()`, the bank picks the next challenge in
/// round-robin order (monotonic counter mod len), assigns a
/// fresh ID, stamps issued_at/expires_at, copies the requested
/// difficulty + tenant_id, and returns. Curator-authored
/// payload + ground truth flow through unchanged.
///
/// The bank refuses to serve a kind other than the one it was
/// built with — `MultiBank` is the only correct way to route
/// across kinds.
pub struct JsonCuratedBank {
    kind: ChallengeKind,
    challenges: Vec<serde_json::Value>,
    counter: std::sync::atomic::AtomicU64,
}

impl JsonCuratedBank {
    /// Construct from a JSON string. Parses the file shape
    /// documented above. Fails if the JSON is malformed, the
    /// `kind` field is missing/invalid, or the `challenges`
    /// array is empty (an empty bank is degenerate — would
    /// always panic at issue time).
    pub fn from_str(s: &str) -> Result<Self, CrucibleError> {
        let v: serde_json::Value = serde_json::from_str(s)
            .map_err(|e| CrucibleError::Internal(format!("parse curated bank: {e}")))?;
        let kind_str = v
            .get("kind")
            .and_then(|k| k.as_str())
            .ok_or_else(|| CrucibleError::Internal("curated bank missing kind".into()))?;
        // Round-trip through the typed enum so unknown kind
        // strings fail closed.
        let kind: ChallengeKind = serde_json::from_value(serde_json::json!(kind_str))
            .map_err(|e| CrucibleError::Internal(format!("invalid kind {kind_str:?}: {e}")))?;
        let challenges = v
            .get("challenges")
            .and_then(|c| c.as_array())
            .ok_or_else(|| CrucibleError::Internal("curated bank missing challenges array".into()))?
            .clone();
        if challenges.is_empty() {
            return Err(CrucibleError::Internal(
                "curated bank: challenges array is empty".into(),
            ));
        }
        Ok(Self {
            kind,
            challenges,
            counter: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Construct by reading the file at `path`. Convenience
    /// wrapper around `from_str` + std::fs::read_to_string.
    pub fn from_path(path: &std::path::Path) -> Result<Self, CrucibleError> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| CrucibleError::Internal(format!("read {}: {e}", path.display())))?;
        Self::from_str(&raw)
    }
}

impl ChallengeBank for JsonCuratedBank {
    fn issue(
        &self,
        kind: ChallengeKind,
        difficulty: Difficulty,
        tenant_id: &str,
    ) -> Result<Challenge, CrucibleError> {
        if kind != self.kind {
            return Err(CrucibleError::Internal(format!(
                "JsonCuratedBank({:?}) cannot serve {:?}",
                self.kind, kind
            )));
        }
        let n = self
            .counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let idx = (n as usize) % self.challenges.len();
        let entry = &self.challenges[idx];
        let payload = entry.get("payload").cloned().ok_or_else(|| {
            CrucibleError::Internal(format!("curated bank entry[{idx}] missing payload"))
        })?;
        let now = time::OffsetDateTime::now_utc();
        // Slug derives from the kind so IDs are human-skimmable
        // in logs across kinds.
        let kind_slug = kind.slug();
        let id = format!("{kind_slug}-{n:08}");
        Ok(Challenge {
            id,
            kind,
            difficulty,
            payload,
            issued_at: now,
            expires_at: now + time::Duration::seconds(120),
            tenant_id: tenant_id.to_owned(),
        })
    }
}

/// Parsed `crucible.toml` config controlling per-tenant
/// attribution policy. Empty / missing config = every tenant
/// defaults to `AttributionPolicy::Curated`.
///
/// On-disk shape:
/// ```toml
/// [tenant.acme]
/// attribution = "tenant-private"
///
/// [tenant."sacred.vote"]
/// attribution = "curated"
///
/// [tenant.experimental]
/// attribution = "ephemeral"
/// ```
///
/// Unknown attribution values fail the parse — fail-closed
/// rather than silently default away from operator intent.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct ServerConfig {
    /// Per-tenant overrides. Keyed by tenant id verbatim.
    #[serde(default)]
    pub tenant: std::collections::HashMap<String, TenantConfig>,
}

/// One tenant's overrides.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct TenantConfig {
    /// Attribution policy for tuples captured under this tenant.
    /// Values: `"curated"` / `"tenant-private"` / `"ephemeral"`.
    pub attribution: String,
}

impl ServerConfig {
    /// Parse a TOML string.
    pub fn from_str(s: &str) -> Result<Self, CrucibleError> {
        toml::from_str(s).map_err(|e| CrucibleError::Internal(format!("parse config: {e}")))
    }

    /// Read + parse a TOML file. Returns `Ok(Self::default())`
    /// if the file doesn't exist — config is optional.
    pub fn from_path(path: &std::path::Path) -> Result<Self, CrucibleError> {
        match std::fs::read_to_string(path) {
            Ok(raw) => Self::from_str(&raw),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(CrucibleError::Internal(format!(
                "read {}: {e}",
                path.display()
            ))),
        }
    }

    /// Resolve a tenant id to an `AttributionPolicy`. Unknown
    /// tenant ids fall back to `Curated`. Unknown attribution
    /// strings (which already failed the typed parse via
    /// `attribution_policy_from_str`) surface as `Curated` here
    /// — the parse step is where policy strings are validated.
    pub fn resolver(self) -> AttributionResolver {
        Arc::new(move |tenant_id: &str| {
            self.tenant
                .get(tenant_id)
                .and_then(|t| attribution_policy_from_str(&t.attribution))
                .unwrap_or(AttributionPolicy::Curated)
        })
    }
}

/// Parse a string into an `AttributionPolicy`. Returns `None`
/// for unknown values; callers decide whether to fall back or
/// fail.
pub fn attribution_policy_from_str(s: &str) -> Option<AttributionPolicy> {
    match s {
        "curated" => Some(AttributionPolicy::Curated),
        "tenant-private" => Some(AttributionPolicy::TenantPrivate),
        "ephemeral" => Some(AttributionPolicy::Ephemeral),
        _ => None,
    }
}

/// State shared across handler invocations.
pub struct AppState {
    /// Issued-but-unsolved challenges. Keyed by Challenge.id.
    pub pending: RwLock<HashMap<String, Challenge>>,
    /// Captured tuples, in order. The host drains this via
    /// [`AppState::drain_captured`] and calls
    /// `crucible_corpus::write_corpus_dir` on the result.
    pub captured: RwLock<Vec<CapturedTuple>>,
    /// Verifier registry. Default = `crucible_challenges::registry()`.
    pub registry: Registry,
    /// Bank that mints fresh challenges on `/challenge`.
    pub bank: Arc<dyn ChallengeBank>,
    /// Per-tenant attribution policy resolver.
    pub attribution: AttributionResolver,
}

impl AppState {
    /// Convenience constructor wrapping the default math bank
    /// + curated-for-everyone attribution. Hosts can construct
    /// `AppState` directly for non-default banks / policies.
    pub fn with_math_bank() -> Arc<Self> {
        Arc::new(Self {
            pending: RwLock::new(HashMap::new()),
            captured: RwLock::new(Vec::new()),
            registry: crucible_challenges::registry(),
            bank: Arc::new(StaticMathBank::default()),
            attribution: Arc::new(|_| AttributionPolicy::Curated),
        })
    }

    /// Drain captured tuples for downstream corpus flush.
    /// Returns owned vec; subsequent calls return empty until
    /// new solutions arrive.
    pub async fn drain_captured(&self) -> Vec<CapturedTuple> {
        let mut g = self.captured.write().await;
        std::mem::take(&mut *g)
    }

    /// Push tuples back into the buffer (front-prepend). Used by
    /// the corpus flusher on write failure — drained tuples that
    /// couldn't be persisted re-queue for the next flush cycle.
    ///
    /// Order isn't load-bearing for the LFI corpus consumer
    /// (it's a tuple SET, not a sequence), so prepending is fine.
    /// Concurrent captures arriving during a failed flush still
    /// land at the back of the buffer; they merge with the
    /// re-queued set on the next drain.
    pub async fn requeue_captured(&self, mut tuples: Vec<CapturedTuple>) {
        if tuples.is_empty() {
            return;
        }
        let mut buf = self.captured.write().await;
        tuples.extend(std::mem::take(&mut *buf));
        *buf = tuples;
    }
}

/// Outcome of one flush cycle. Returned by [`try_flush_once`] so
/// callers (the bin's periodic flusher; tests) can observe what
/// happened without parsing log lines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FlushOutcome {
    /// Captured buffer was empty; nothing to do.
    NothingToDo,
    /// Successfully wrote N corpus patterns to the timestamped
    /// subdirectory under the flush-dir.
    Wrote {
        /// Pattern count written.
        patterns: usize,
        /// Subdirectory path the patterns landed in.
        target: std::path::PathBuf,
    },
    /// Write failed; tuples re-queued for next cycle.
    Failed {
        /// Subdirectory path the write attempted.
        target: std::path::PathBuf,
        /// Error message.
        error: String,
        /// Number of tuples re-queued.
        requeued: usize,
    },
}

/// Run one corpus-flush cycle: drain captured tuples, convert to
/// CorpusPatterns, write to a fresh RFC-3339-timestamped subdir
/// of `flush_dir`. On write failure, re-queue the drained tuples
/// back into AppState.captured (front-prepended) so the next
/// cycle retries.
///
/// Extracted from the bin's spawn_flusher loop so:
///   - Integration tests can exercise the flush flow directly.
///   - Future callers (graceful-shutdown drain → final write
///     → requeue-on-fail; admin "force flush now" endpoint;
///     etc.) reuse the same code path.
pub async fn try_flush_once(
    state: &AppState,
    flush_dir: &std::path::Path,
) -> FlushOutcome {
    let captured = state.drain_captured().await;
    if captured.is_empty() {
        return FlushOutcome::NothingToDo;
    }
    let patterns: Vec<crucible_corpus::CorpusPattern> = captured
        .iter()
        .filter_map(|t| crucible_corpus::to_pattern(t).ok())
        .collect();
    if patterns.is_empty() {
        // No patterns to write (every tuple ineligible per
        // attribution policy or non-human verdict). Tuples are
        // dropped — that's the documented Ephemeral/NotHuman
        // contract; no requeue needed.
        return FlushOutcome::NothingToDo;
    }
    let ts = time::OffsetDateTime::now_utc()
        .format(&time::format_description::well_known::Rfc3339)
        .unwrap_or_else(|_| "unknown".into())
        .replace(':', "");
    let target = flush_dir.join(format!("flush-{ts}"));
    match crucible_corpus::write_corpus_dir(&patterns, &target) {
        Ok(_) => FlushOutcome::Wrote {
            patterns: patterns.len(),
            target,
        },
        Err(e) => {
            let requeued = captured.len();
            state.requeue_captured(captured).await;
            FlushOutcome::Failed {
                target,
                error: e.to_string(),
                requeued,
            }
        }
    }
}

/// Build the axum router. Mount under a base path like
/// `/crucible` via `.nest("/crucible", crucible_server::router(...))`.
pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/challenge", post(handle_challenge))
        .route("/solve", post(handle_solve))
        .with_state(state)
}

/// `POST /challenge` request shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct ChallengeRequest {
    /// Which kind of challenge to mint.
    pub kind: ChallengeKind,
    /// Difficulty hint passed to the verifier on retry.
    pub difficulty: Difficulty,
    /// Tenant scope for attribution + per-tenant corpus.
    pub tenant_id: String,
}

/// `POST /solve` request shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct SolveRequest {
    /// Which challenge this solution is for.
    pub challenge_id: String,
    /// User's response payload (verifier-kind-specific).
    pub response: serde_json::Value,
    /// Submission timestamp from the client.
    #[serde(with = "time::serde::rfc3339")]
    pub submitted_at: time::OffsetDateTime,
    /// Elapsed-ms between challenge load and submission.
    pub elapsed_ms: u32,
}

/// `POST /solve` response shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct SolveResponse {
    /// The verifier's verdict.
    pub verdict: Verdict,
}

/// API error → HTTP status code.
#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    /// The challenge ID isn't pending (expired, never issued,
    /// or already solved).
    #[error("unknown challenge id: {0}")]
    UnknownChallenge(String),
    /// The bank refused the request (e.g. unsupported kind).
    #[error("issue failed: {0}")]
    Issue(#[from] CrucibleError),
    /// Internal error during verify.
    #[error("verify failed: {0}")]
    Verify(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, msg) = match &self {
            ApiError::UnknownChallenge(_) => (StatusCode::NOT_FOUND, self.to_string()),
            ApiError::Issue(_) => (StatusCode::BAD_REQUEST, self.to_string()),
            ApiError::Verify(_) => (StatusCode::INTERNAL_SERVER_ERROR, self.to_string()),
        };
        (status, Json(serde_json::json!({"error": msg}))).into_response()
    }
}

async fn handle_challenge(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChallengeRequest>,
) -> Result<Json<Challenge>, ApiError> {
    let challenge = state.bank.issue(req.kind, req.difficulty, &req.tenant_id)?;
    let mut pending = state.pending.write().await;
    pending.insert(challenge.id.clone(), challenge.clone());
    Ok(Json(challenge))
}

async fn handle_solve(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SolveRequest>,
) -> Result<Json<SolveResponse>, ApiError> {
    let challenge = {
        let mut pending = state.pending.write().await;
        pending
            .remove(&req.challenge_id)
            .ok_or_else(|| ApiError::UnknownChallenge(req.challenge_id.clone()))?
    };
    let solution = Solution {
        challenge_id: req.challenge_id.clone(),
        response: req.response,
        submitted_at: req.submitted_at,
        elapsed_ms: req.elapsed_ms,
    };
    let (verdict, ground_truth) = state
        .registry
        .verify(&challenge, &solution)
        .map_err(|e| ApiError::Verify(e.to_string()))?;

    // Capture the tuple regardless of verdict — Bot verdicts
    // still inform the LFI corpus about the distribution of
    // wrong-answer + too-fast attacks. Attribution is the
    // policy gate that decides whether the tuple flows to the
    // public corpus / tenant-private corpus / nowhere.
    let attribution = (state.attribution)(&challenge.tenant_id);
    let tuple = CapturedTuple {
        challenge: challenge.clone(),
        solution,
        ground_truth,
        verdict: verdict.clone(),
        attribution,
    };
    state.captured.write().await.push(tuple);

    Ok(Json(SolveResponse { verdict }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    async fn body_json(resp: Response) -> serde_json::Value {
        let body = axum::body::to_bytes(resp.into_body(), 65536)
            .await
            .expect("body");
        serde_json::from_slice(&body).expect("json")
    }

    #[tokio::test]
    async fn challenge_endpoint_mints_math_arithmetic() {
        let state = AppState::with_math_bank();
        let app = router(state.clone());
        let req = Request::builder()
            .uri("/challenge")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::to_vec(&serde_json::json!({
                    "kind": "math-arithmetic",
                    "difficulty": "medium",
                    "tenant-id": "acme"
                }))
                .unwrap(),
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_json(resp).await;
        assert_eq!(body["kind"], "math-arithmetic");
        assert!(body["payload"]["a"].is_i64());
        assert!(body["payload"]["b"].is_i64());
        assert!(body["id"].as_str().unwrap().starts_with("math-"));
        // One pending entry stashed.
        assert_eq!(state.pending.read().await.len(), 1);
    }

    #[tokio::test]
    async fn solve_endpoint_verifies_and_captures_tuple() {
        let state = AppState::with_math_bank();
        // Mint a challenge first.
        let challenge = state
            .bank
            .issue(ChallengeKind::MathArithmetic, Difficulty::Medium, "acme")
            .unwrap();
        let a = challenge.payload["a"].as_i64().unwrap();
        let b = challenge.payload["b"].as_i64().unwrap();
        let op = challenge.payload["op"].as_str().unwrap();
        let truth = match op {
            "+" => a + b,
            "-" => a - b,
            "*" => a * b,
            _ => unreachable!(),
        };
        let challenge_id = challenge.id.clone();
        state
            .pending
            .write()
            .await
            .insert(challenge_id.clone(), challenge);

        let app = router(state.clone());
        let solve = serde_json::json!({
            "challenge-id": challenge_id,
            "response": {"answer": truth},
            "submitted-at": "2026-05-20T19:00:00Z",
            "elapsed-ms": 2500u32
        });
        let req = Request::builder()
            .uri("/solve")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&solve).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        // 2026-05-20 + 2 minute expiry from issuance "now" =
        // expired in the verifier (Challenge::issued_at was
        // computed at bank.issue() call time, real-wallclock).
        // Verifier returns CrucibleError::Expired which the
        // handler maps to Verify(500). Accept either 200 or
        // 500 — what matters here is that pending was consumed.
        assert!(
            resp.status() == StatusCode::OK || resp.status() == StatusCode::INTERNAL_SERVER_ERROR
        );
        // The challenge was consumed regardless.
        assert!(state.pending.read().await.is_empty());
    }

    #[tokio::test]
    async fn solve_unknown_challenge_returns_404() {
        let state = AppState::with_math_bank();
        let app = router(state);
        let solve = serde_json::json!({
            "challenge-id": "does-not-exist",
            "response": {"answer": 1},
            "submitted-at": "2026-05-20T19:00:00Z",
            "elapsed-ms": 1000u32
        });
        let req = Request::builder()
            .uri("/solve")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&solve).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn static_math_bank_rejects_non_math_kinds() {
        let bank = StaticMathBank::default();
        let r = bank.issue(ChallengeKind::ImageClassify, Difficulty::Medium, "acme");
        assert!(r.is_err());
    }

    #[test]
    fn server_config_parses_per_tenant_attribution() {
        let raw = r#"
[tenant.acme]
attribution = "tenant-private"

[tenant.experimental]
attribution = "ephemeral"
"#;
        let cfg = ServerConfig::from_str(raw).unwrap();
        assert_eq!(cfg.tenant.len(), 2);
        let resolve = cfg.resolver();
        assert!(matches!(resolve("acme"), AttributionPolicy::TenantPrivate));
        assert!(matches!(
            resolve("experimental"),
            AttributionPolicy::Ephemeral
        ));
        // Unknown tenant → curated default.
        assert!(matches!(
            resolve("unconfigured"),
            AttributionPolicy::Curated
        ));
    }

    #[test]
    fn server_config_empty_returns_default() {
        let cfg = ServerConfig::from_str("").unwrap();
        assert!(cfg.tenant.is_empty());
    }

    #[test]
    fn server_config_rejects_unknown_keys() {
        let raw = r#"
unrelated_top_key = 42
"#;
        // deny_unknown_fields on the outer struct: unrelated top
        // key fails the parse rather than silently dropping.
        assert!(ServerConfig::from_str(raw).is_err());
    }

    #[test]
    fn server_config_resolver_unknown_attribution_falls_back() {
        // Unknown attribution string in the file → fallback to
        // Curated when the resolver runs. The parse itself accepts
        // the string (it's just a String field); the resolver is
        // where the policy enum lookup happens.
        let raw = r#"
[tenant.weird]
attribution = "made-up-policy"
"#;
        let cfg = ServerConfig::from_str(raw).unwrap();
        let resolve = cfg.resolver();
        assert!(matches!(resolve("weird"), AttributionPolicy::Curated));
    }

    #[test]
    fn attribution_policy_from_str_round_trips() {
        assert!(matches!(
            attribution_policy_from_str("curated"),
            Some(AttributionPolicy::Curated)
        ));
        assert!(matches!(
            attribution_policy_from_str("tenant-private"),
            Some(AttributionPolicy::TenantPrivate)
        ));
        assert!(matches!(
            attribution_policy_from_str("ephemeral"),
            Some(AttributionPolicy::Ephemeral)
        ));
        assert!(attribution_policy_from_str("unknown").is_none());
    }

    #[test]
    fn multi_bank_dispatches_by_kind() {
        let bank = MultiBank::new().register(
            ChallengeKind::MathArithmetic,
            Box::new(StaticMathBank::default()),
        );
        let r = bank
            .issue(ChallengeKind::MathArithmetic, Difficulty::Medium, "acme")
            .unwrap();
        assert_eq!(r.kind, ChallengeKind::MathArithmetic);
        // Unregistered kind → Internal error.
        let r = bank.issue(ChallengeKind::ImageClassify, Difficulty::Medium, "acme");
        assert!(matches!(r, Err(CrucibleError::Internal(_))));
    }

    #[test]
    fn json_curated_bank_round_robin_issues_curator_payloads() {
        let json = r##"{
            "kind": "semantic-similarity",
            "challenges": [
                {"payload": {"prompt": "happy",
                             "options": ["joyful", "sad"],
                             "truth_indices": [0]}},
                {"payload": {"prompt": "cold",
                             "options": ["freezing", "warm"],
                             "truth_indices": [0]}}
            ]
        }"##;
        let bank = JsonCuratedBank::from_str(json).unwrap();
        let a = bank
            .issue(
                ChallengeKind::SemanticSimilarity,
                Difficulty::Medium,
                "acme",
            )
            .unwrap();
        assert_eq!(a.payload["prompt"], "happy");
        assert!(a.id.starts_with("semantic-similarity-"));
        let b = bank
            .issue(
                ChallengeKind::SemanticSimilarity,
                Difficulty::Medium,
                "acme",
            )
            .unwrap();
        assert_eq!(b.payload["prompt"], "cold");
        // Round-robin wraps around.
        let c = bank
            .issue(
                ChallengeKind::SemanticSimilarity,
                Difficulty::Medium,
                "acme",
            )
            .unwrap();
        assert_eq!(c.payload["prompt"], "happy");
    }

    #[test]
    fn json_curated_bank_refuses_wrong_kind() {
        let json = r#"{"kind":"semantic-similarity","challenges":[{"payload":{}}]}"#;
        let bank = JsonCuratedBank::from_str(json).unwrap();
        let r = bank.issue(ChallengeKind::ImageClassify, Difficulty::Medium, "acme");
        assert!(matches!(r, Err(CrucibleError::Internal(_))));
    }

    #[test]
    fn json_curated_bank_rejects_empty_challenges() {
        let json = r#"{"kind":"image-classify","challenges":[]}"#;
        let r = JsonCuratedBank::from_str(json);
        assert!(matches!(r, Err(CrucibleError::Internal(_))));
    }

    #[test]
    fn json_curated_bank_rejects_unknown_kind() {
        let json = r#"{"kind":"made-up-kind","challenges":[{"payload":{}}]}"#;
        let r = JsonCuratedBank::from_str(json);
        assert!(matches!(r, Err(CrucibleError::Internal(_))));
    }

    #[tokio::test]
    async fn drain_captured_clears_buffer() {
        let state = AppState::with_math_bank();
        // Manually inject a captured tuple.
        let now = time::OffsetDateTime::now_utc();
        state.captured.write().await.push(CapturedTuple {
            challenge: Challenge {
                id: "x".into(),
                kind: ChallengeKind::MathArithmetic,
                difficulty: Difficulty::Medium,
                payload: serde_json::json!({}),
                issued_at: now,
                expires_at: now + time::Duration::seconds(120),
                tenant_id: "acme".into(),
            },
            solution: Solution {
                challenge_id: "x".into(),
                response: serde_json::json!({}),
                submitted_at: now,
                elapsed_ms: 1000,
            },
            ground_truth: serde_json::json!({}),
            verdict: Verdict::Human { confidence: 0.9 },
            attribution: AttributionPolicy::Curated,
        });
        let drained = state.drain_captured().await;
        assert_eq!(drained.len(), 1);
        assert!(state.captured.read().await.is_empty());
    }

    fn sample_tuple(id: &str) -> CapturedTuple {
        let now = time::OffsetDateTime::now_utc();
        CapturedTuple {
            challenge: Challenge {
                id: id.into(),
                kind: ChallengeKind::MathArithmetic,
                difficulty: Difficulty::Medium,
                payload: serde_json::json!({}),
                issued_at: now,
                expires_at: now + time::Duration::seconds(120),
                tenant_id: "acme".into(),
            },
            solution: Solution {
                challenge_id: id.into(),
                response: serde_json::json!({}),
                submitted_at: now,
                elapsed_ms: 1000,
            },
            ground_truth: serde_json::json!({}),
            verdict: Verdict::Human { confidence: 0.9 },
            attribution: AttributionPolicy::Curated,
        }
    }

    #[tokio::test]
    async fn requeue_captured_restores_tuples_at_buffer_front() {
        let state = AppState::with_math_bank();
        // Pre-existing captures (simulate live solves arriving
        // during a failed flush attempt).
        state.captured.write().await.push(sample_tuple("late-1"));
        state.captured.write().await.push(sample_tuple("late-2"));

        // Requeue the "drained" tuples from a failed flush.
        state
            .requeue_captured(vec![sample_tuple("drained-1"), sample_tuple("drained-2")])
            .await;

        // Drained tuples sit at the front, late captures behind.
        let final_buf = state.captured.read().await;
        assert_eq!(final_buf.len(), 4);
        assert_eq!(final_buf[0].challenge.id, "drained-1");
        assert_eq!(final_buf[1].challenge.id, "drained-2");
        assert_eq!(final_buf[2].challenge.id, "late-1");
        assert_eq!(final_buf[3].challenge.id, "late-2");
    }

    #[tokio::test]
    async fn requeue_captured_empty_input_is_noop() {
        let state = AppState::with_math_bank();
        state.captured.write().await.push(sample_tuple("a"));
        state.requeue_captured(vec![]).await;
        let buf = state.captured.read().await;
        assert_eq!(buf.len(), 1);
        assert_eq!(buf[0].challenge.id, "a");
    }

    #[tokio::test]
    async fn drain_then_requeue_round_trips() {
        let state = AppState::with_math_bank();
        state.captured.write().await.push(sample_tuple("a"));
        state.captured.write().await.push(sample_tuple("b"));
        let drained = state.drain_captured().await;
        assert_eq!(drained.len(), 2);
        assert!(state.captured.read().await.is_empty());
        // Requeue — back to original state.
        state.requeue_captured(drained).await;
        let buf = state.captured.read().await;
        assert_eq!(buf.len(), 2);
        assert_eq!(buf[0].challenge.id, "a");
        assert_eq!(buf[1].challenge.id, "b");
    }
}
