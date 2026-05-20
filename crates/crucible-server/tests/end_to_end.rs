//! End-to-end integration tests for crucible-server.
//!
//! Drives the full HTTP router with a `MultiBank` mixing
//! `StaticMathBank` + `JsonCuratedBank`, runs both a
//! challenge → solve → verdict round trip and an inspection
//! of the captured-tuple drain. Covers the wire-shape
//! contract the production deployment relies on.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, Response, StatusCode};
use crucible_core::{AttributionPolicy, ChallengeKind};
use crucible_server::{
    router, try_flush_once, AppState, FlushOutcome, JsonCuratedBank, MultiBank, StaticMathBank,
};
use tokio::sync::RwLock;
use tower::ServiceExt;

const SEM_BANK_JSON: &str = r##"{
    "kind": "semantic-similarity",
    "challenges": [
        {"payload": {
            "prompt": "happy",
            "options": ["joyful", "sad", "elated"],
            "truth_indices": [0, 2]
        }}
    ]
}"##;

fn state_with_math_and_semantic() -> Arc<AppState> {
    let multi = MultiBank::new()
        .register(
            ChallengeKind::MathArithmetic,
            Box::new(StaticMathBank::default()),
        )
        .register(
            ChallengeKind::SemanticSimilarity,
            Box::new(JsonCuratedBank::from_str(SEM_BANK_JSON).expect("bank parse")),
        );
    Arc::new(AppState {
        pending: RwLock::new(std::collections::HashMap::new()),
        captured: RwLock::new(Vec::new()),
        registry: crucible_challenges::registry(),
        bank: Arc::new(multi),
        attribution: Arc::new(|_| AttributionPolicy::Curated),
    })
}

async fn body_json(resp: Response<Body>) -> serde_json::Value {
    let body = axum::body::to_bytes(resp.into_body(), 65536)
        .await
        .expect("body");
    serde_json::from_slice(&body).expect("json")
}

async fn post(state: Arc<AppState>, uri: &str, body: serde_json::Value) -> Response<Body> {
    let app = router(state);
    let req = Request::builder()
        .uri(uri)
        .method("POST")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();
    app.oneshot(req).await.unwrap()
}

#[tokio::test]
async fn math_full_round_trip_produces_human_verdict() {
    let state = state_with_math_and_semantic();

    // Mint a math challenge.
    let resp = post(
        state.clone(),
        "/challenge",
        serde_json::json!({
            "kind": "math-arithmetic",
            "difficulty": "medium",
            "tenant-id": "acme"
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let challenge = body_json(resp).await;
    let a = challenge["payload"]["a"].as_i64().unwrap();
    let b = challenge["payload"]["b"].as_i64().unwrap();
    let op = challenge["payload"]["op"].as_str().unwrap();
    let truth = match op {
        "+" => a + b,
        "-" => a - b,
        "*" => a * b,
        _ => unreachable!(),
    };
    let challenge_id = challenge["id"].as_str().unwrap().to_owned();

    // Solve it correctly with a realistic elapsed-ms.
    let resp = post(
        state.clone(),
        "/solve",
        serde_json::json!({
            "challenge-id": challenge_id,
            "response": {"answer": truth},
            "submitted-at": "2026-05-20T19:00:00Z",
            "elapsed-ms": 2_500u32
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp).await;
    assert_eq!(body["verdict"]["kind"], "human");

    // Captured tuple landed in the drain buffer.
    let drained = state.drain_captured().await;
    assert_eq!(drained.len(), 1);
    assert!(matches!(
        drained[0].verdict,
        crucible_core::Verdict::Human { .. }
    ));
}

#[tokio::test]
async fn semantic_curated_round_trip_produces_human_verdict() {
    let state = state_with_math_and_semantic();

    let resp = post(
        state.clone(),
        "/challenge",
        serde_json::json!({
            "kind": "semantic-similarity",
            "difficulty": "medium",
            "tenant-id": "acme"
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let challenge = body_json(resp).await;
    let challenge_id = challenge["id"].as_str().unwrap().to_owned();
    assert_eq!(challenge["payload"]["prompt"], "happy");
    // Truth indices come back unchanged from the curator file.
    assert_eq!(
        challenge["payload"]["truth_indices"],
        serde_json::json!([0, 2])
    );

    // Submit picks that exactly match the truth set.
    let resp = post(
        state.clone(),
        "/solve",
        serde_json::json!({
            "challenge-id": challenge_id,
            "response": {"picks": [0, 2]},
            "submitted-at": "2026-05-20T19:00:00Z",
            "elapsed-ms": 4_000u32
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp).await;
    assert_eq!(body["verdict"]["kind"], "human");

    let drained = state.drain_captured().await;
    assert_eq!(drained.len(), 1);
    assert_eq!(drained[0].challenge.kind, ChallengeKind::SemanticSimilarity);
}

#[tokio::test]
async fn solve_wrong_answer_captures_bot_tuple() {
    let state = state_with_math_and_semantic();
    let resp = post(
        state.clone(),
        "/challenge",
        serde_json::json!({
            "kind": "math-arithmetic",
            "difficulty": "medium",
            "tenant-id": "acme"
        }),
    )
    .await;
    let challenge = body_json(resp).await;
    let challenge_id = challenge["id"].as_str().unwrap().to_owned();
    let a = challenge["payload"]["a"].as_i64().unwrap();
    let b = challenge["payload"]["b"].as_i64().unwrap();
    let op = challenge["payload"]["op"].as_str().unwrap();
    let truth = match op {
        "+" => a + b,
        "-" => a - b,
        "*" => a * b,
        _ => unreachable!(),
    };
    // Submit a deliberately wrong answer.
    let wrong = truth + 999;
    let resp = post(
        state.clone(),
        "/solve",
        serde_json::json!({
            "challenge-id": challenge_id,
            "response": {"answer": wrong},
            "submitted-at": "2026-05-20T19:00:00Z",
            "elapsed-ms": 2_500u32
        }),
    )
    .await;
    let body = body_json(resp).await;
    assert_eq!(body["verdict"]["kind"], "bot");

    // Bot verdicts are STILL captured — that's the LFI training
    // signal about attack distributions.
    let drained = state.drain_captured().await;
    assert_eq!(drained.len(), 1);
    assert!(matches!(
        drained[0].verdict,
        crucible_core::Verdict::Bot { .. }
    ));
}

#[tokio::test]
async fn challenge_for_unregistered_kind_returns_400() {
    let state = state_with_math_and_semantic();
    let resp = post(
        state,
        "/challenge",
        serde_json::json!({
            "kind": "image-classify", // no bank registered for this kind
            "difficulty": "medium",
            "tenant-id": "acme"
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn multiple_challenges_can_run_concurrently() {
    let state = state_with_math_and_semantic();

    // Mint three challenges before solving any. The pending
    // map must hold all three — challenges are independent.
    let mut ids = Vec::new();
    for _ in 0..3 {
        let resp = post(
            state.clone(),
            "/challenge",
            serde_json::json!({
                "kind": "math-arithmetic",
                "difficulty": "medium",
                "tenant-id": "acme"
            }),
        )
        .await;
        let challenge = body_json(resp).await;
        ids.push(challenge["id"].as_str().unwrap().to_owned());
    }
    assert_eq!(state.pending.read().await.len(), 3);
    // Solving one only removes that one.
    let _ = post(
        state.clone(),
        "/solve",
        serde_json::json!({
            "challenge-id": ids[0],
            "response": {"answer": 99},
            "submitted-at": "2026-05-20T19:00:00Z",
            "elapsed-ms": 2_500u32
        }),
    )
    .await;
    assert_eq!(state.pending.read().await.len(), 2);
}

fn tmpdir(label: &str) -> std::path::PathBuf {
    let pid = std::process::id();
    let n = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    std::env::temp_dir().join(format!("crucible-flush-{label}-{pid}-{n}"))
}

#[tokio::test]
async fn try_flush_once_empty_buffer_is_nothing_to_do() {
    let state = AppState::with_math_bank();
    let dir = tmpdir("empty");
    let outcome = try_flush_once(&state, &dir).await;
    assert!(matches!(outcome, FlushOutcome::NothingToDo));
}

#[tokio::test]
async fn try_flush_once_writes_human_verdicts_to_disk() {
    let state = state_with_math_and_semantic();

    // Drive a math challenge to a Human verdict so the buffer
    // holds a real captured tuple.
    let resp = post(
        state.clone(),
        "/challenge",
        serde_json::json!({
            "kind": "math-arithmetic",
            "difficulty": "medium",
            "tenant-id": "acme"
        }),
    )
    .await;
    let challenge = body_json(resp).await;
    let challenge_id = challenge["id"].as_str().unwrap().to_owned();
    let a = challenge["payload"]["a"].as_i64().unwrap();
    let b = challenge["payload"]["b"].as_i64().unwrap();
    let op = challenge["payload"]["op"].as_str().unwrap();
    let truth = match op {
        "+" => a + b,
        "-" => a - b,
        "*" => a * b,
        _ => unreachable!(),
    };
    let _ = post(
        state.clone(),
        "/solve",
        serde_json::json!({
            "challenge-id": challenge_id,
            "response": {"answer": truth},
            "submitted-at": "2026-05-20T19:00:00Z",
            "elapsed-ms": 2_500u32
        }),
    )
    .await;

    let dir = tmpdir("wrote");
    let outcome = try_flush_once(&state, &dir).await;
    match outcome {
        FlushOutcome::Wrote { patterns, target } => {
            assert_eq!(patterns, 1);
            assert!(target.exists(), "target dir should exist after Wrote");
            assert!(target.join("index.json").exists(), "manifest should land");
        }
        other => panic!("expected Wrote, got {other:?}"),
    }
    // Buffer is now empty.
    assert!(state.captured.read().await.is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn try_flush_once_requeues_on_write_failure() {
    let state = state_with_math_and_semantic();
    // Drive one Human verdict into the buffer.
    let resp = post(
        state.clone(),
        "/challenge",
        serde_json::json!({
            "kind": "math-arithmetic",
            "difficulty": "medium",
            "tenant-id": "acme"
        }),
    )
    .await;
    let challenge = body_json(resp).await;
    let challenge_id = challenge["id"].as_str().unwrap().to_owned();
    let a = challenge["payload"]["a"].as_i64().unwrap();
    let b = challenge["payload"]["b"].as_i64().unwrap();
    let op = challenge["payload"]["op"].as_str().unwrap();
    let truth = match op {
        "+" => a + b,
        "-" => a - b,
        "*" => a * b,
        _ => unreachable!(),
    };
    let _ = post(
        state.clone(),
        "/solve",
        serde_json::json!({
            "challenge-id": challenge_id,
            "response": {"answer": truth},
            "submitted-at": "2026-05-20T19:00:00Z",
            "elapsed-ms": 2_500u32
        }),
    )
    .await;
    assert_eq!(state.captured.read().await.len(), 1);

    // Force write failure: point the flusher at /proc which is
    // read-only on Linux. write_corpus_dir mkdir-all fails.
    let outcome = try_flush_once(&state, std::path::Path::new("/proc/cannot-write")).await;
    match outcome {
        FlushOutcome::Failed { requeued, .. } => {
            assert_eq!(requeued, 1, "one tuple should be re-queued");
        }
        other => panic!("expected Failed, got {other:?}"),
    }
    // Buffer holds the requeued tuple.
    assert_eq!(state.captured.read().await.len(), 1);
}
