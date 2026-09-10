//! Every curated bank in `banks/` is checked against the verifier that
//! will actually grade it.
//!
//! A bank is data, and data is where a silent mistake lives longest: a
//! `truth_indices` off by one, a copied item whose answer was not
//! updated, a distractor that is genuinely a paraphrase. None of those
//! break a build or fail a unit test — they ship a challenge no human
//! can pass, and the failure looks like "the form is broken" weeks later.
//!
//! So each item is graded twice: once with the curator's answer, which
//! must come back `Human`, and once with the opposite, which must not.
//! That second half is the one that catches an item whose truth is
//! wrong, because a mislabelled item passes the first check happily.

use std::collections::BTreeSet;

use crucible_challenges::registry;
use crucible_core::{Challenge, ChallengeKind, Difficulty, Solution, Verdict};

fn banks_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../banks")
}

fn load(name: &str) -> serde_json::Value {
    let p = banks_dir().join(name);
    let raw = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse {}: {e}", p.display()))
}

fn items(bank: &serde_json::Value) -> Vec<serde_json::Value> {
    bank["challenges"]
        .as_array()
        .expect("challenges array")
        .iter()
        .map(|c| c["payload"].clone())
        .collect()
}

/// Build a Challenge/Solution pair `elapsed_ms` apart.
fn graded(
    kind: ChallengeKind,
    payload: &serde_json::Value,
    response: serde_json::Value,
    elapsed_ms: u32,
) -> Verdict {
    let issued = time::OffsetDateTime::from_unix_timestamp(1_700_000_000).unwrap();
    let challenge = Challenge {
        id: "bank-check".to_owned(),
        kind,
        difficulty: Difficulty::Medium,
        payload: payload.clone(),
        issued_at: issued,
        expires_at: issued + time::Duration::minutes(30),
        tenant_id: "test".to_owned(),
    };
    let solution = Solution {
        challenge_id: challenge.id.clone(),
        response,
        submitted_at: issued + time::Duration::milliseconds(i64::from(elapsed_ms)),
        elapsed_ms,
    };
    let (verdict, _truth) = registry()
        .verify(&challenge, &solution)
        .expect("verifier ran");
    verdict
}

fn is_human(v: &Verdict) -> bool {
    matches!(v, Verdict::Human { .. })
}

// ---------------------------------------------------------------- injection

#[test]
fn every_injection_item_is_solvable_and_its_truth_is_right() {
    let bank = load("prompt-injection.json");
    assert_eq!(bank["kind"], "prompt-injection-detect");
    let items = items(&bank);
    assert!(items.len() >= 12, "bank is too small to be worth serving");

    for (i, p) in items.iter().enumerate() {
        let prompt = p["prompt"]
            .as_str()
            .unwrap_or_else(|| panic!("item {i}: no prompt"));
        assert!(!prompt.trim().is_empty(), "item {i}: empty prompt");
        let truth = p["is_injection"]
            .as_bool()
            .unwrap_or_else(|| panic!("item {i}: is_injection must be a bool"));

        let right = if truth { "unsafe" } else { "safe" };
        let wrong = if truth { "safe" } else { "unsafe" };

        let v = graded(
            ChallengeKind::PromptInjectionDetect,
            p,
            serde_json::json!({ "verdict": right }),
            5_000,
        );
        assert!(
            is_human(&v),
            "item {i} ({prompt:?}) is labelled {truth} but answering {right:?} did not \
             pass: {v:?}"
        );

        let v = graded(
            ChallengeKind::PromptInjectionDetect,
            p,
            serde_json::json!({ "verdict": wrong }),
            5_000,
        );
        assert!(
            !is_human(&v),
            "item {i} ({prompt:?}) accepted the WRONG answer {wrong:?} — its \
             is_injection label is probably backwards"
        );
    }
}

#[test]
fn the_injection_bank_is_balanced() {
    // An unbalanced bank teaches a prior: a solver that always answers
    // "unsafe" would pass most of the time, and the corpus would carry
    // that skew downstream into LFI.
    let items = items(&load("prompt-injection.json"));
    let n = items.len();
    let unsafe_n = items
        .iter()
        .filter(|p| p["is_injection"].as_bool() == Some(true))
        .count();
    assert_eq!(
        unsafe_n * 2,
        n,
        "{unsafe_n} of {n} are injections; a balanced bank is what stops \
         always-answering-one-way from working"
    );
}

#[test]
fn no_injection_prompt_is_duplicated() {
    let items = items(&load("prompt-injection.json"));
    let mut seen = BTreeSet::new();
    for p in &items {
        let t = p["prompt"].as_str().unwrap().trim().to_lowercase();
        assert!(seen.insert(t.clone()), "duplicate prompt: {t:?}");
    }
}

// ---------------------------------------------------------------- semantic

#[test]
fn every_semantic_item_is_solvable_and_its_truth_is_right() {
    let bank = load("semantic-similarity.json");
    assert_eq!(bank["kind"], "semantic-similarity");
    let items = items(&bank);
    assert!(items.len() >= 10, "bank is too small to be worth serving");

    for (i, p) in items.iter().enumerate() {
        let opts = p["options"]
            .as_array()
            .unwrap_or_else(|| panic!("item {i}: options"));
        let truth: Vec<i64> = p["truth_indices"]
            .as_array()
            .unwrap_or_else(|| panic!("item {i}: truth_indices"))
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();

        assert!(opts.len() >= 3, "item {i}: needs at least 3 options");
        assert!(!truth.is_empty(), "item {i}: no correct option");
        assert!(
            truth.len() < opts.len(),
            "item {i}: every option is correct, so the item discriminates nothing"
        );
        let uniq: BTreeSet<i64> = truth.iter().copied().collect();
        assert_eq!(uniq.len(), truth.len(), "item {i}: duplicate truth index");
        for t in &truth {
            assert!(
                *t >= 0 && (*t as usize) < opts.len(),
                "item {i}: truth index {t} is outside the {} options",
                opts.len()
            );
        }

        // Curator's answer must pass.
        let v = graded(
            ChallengeKind::SemanticSimilarity,
            p,
            serde_json::json!({ "picks": truth }),
            5_000,
        );
        assert!(
            is_human(&v),
            "item {i}: the curator's own answer failed: {v:?}"
        );

        // The complement must not. This is what catches an item whose
        // distractors are accidentally paraphrases too.
        let complement: Vec<i64> = (0..opts.len() as i64)
            .filter(|j| !uniq.contains(j))
            .collect();
        let v = graded(
            ChallengeKind::SemanticSimilarity,
            p,
            serde_json::json!({ "picks": complement }),
            5_000,
        );
        assert!(
            !is_human(&v),
            "item {i}: selecting exactly the WRONG options also passed — the \
             distractors are not distinct enough in meaning"
        );
    }
}

#[test]
fn a_correct_but_instant_answer_is_refused_by_every_bank() {
    // The banks are only half the gate; the latency floor is the other
    // half, and it has to hold for curated content too.
    type Answer = fn(&serde_json::Value) -> serde_json::Value;

    let injection: Answer = |p| {
        let t = p["is_injection"].as_bool().unwrap();
        serde_json::json!({ "verdict": if t { "unsafe" } else { "safe" } })
    };
    let semantic: Answer = |p| serde_json::json!({ "picks": p["truth_indices"] });

    let cases: [(&str, ChallengeKind, Answer); 2] = [
        (
            "prompt-injection.json",
            ChallengeKind::PromptInjectionDetect,
            injection,
        ),
        (
            "semantic-similarity.json",
            ChallengeKind::SemanticSimilarity,
            semantic,
        ),
    ];

    for (bank_file, kind, answer) in cases {
        for p in items(&load(bank_file)) {
            let v = graded(kind, &p, answer(&p), 0);
            assert!(
                !is_human(&v),
                "{bank_file}: a correct answer submitted instantly passed; the \
                 latency floor is not doing its job"
            );
        }
    }
}
