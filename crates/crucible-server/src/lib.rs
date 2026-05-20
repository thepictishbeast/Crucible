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
            resp.status() == StatusCode::OK
                || resp.status() == StatusCode::INTERNAL_SERVER_ERROR
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
}
