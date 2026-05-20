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
use crucible_server::{router, AppState, JsonCuratedBank, MultiBank, StaticMathBank};
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
