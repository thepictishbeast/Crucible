//! `crucible-challenges` — verifier implementations per kind.
//!
//! Every [`ChallengeKind`] has a matching `Verifier` impl that
//! takes a [`Challenge`] + a [`Solution`] and produces a typed
//! [`Verdict`] + the ground truth used for the comparison.
//!
//! ## Adding a kind
//!
//! 1. Add the variant to `ChallengeKind` in `crucible-core`.
//! 2. Add the verifier impl to this crate.
//! 3. Decide if the kind contributes labels to LFI's corpus
//!    (`ChallengeKind::trains_lfi`).
//! 4. Add at least one unit test covering Human / Bot /
//!    Inconclusive verdicts.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use crucible_core::{Challenge, ChallengeKind, CrucibleError, Difficulty, Solution, Verdict};

/// Per-kind verifier trait. Stateless — verifiers consult the
/// challenge payload + the solution.
pub trait Verifier: Send + Sync {
    /// Which kind this verifier handles.
    fn kind(&self) -> ChallengeKind;

    /// Verify a solution. Returns `(Verdict, ground_truth_json)`
    /// so the caller can pass the ground truth on to
    /// `crucible-corpus` for LFI export.
    fn verify(
        &self,
        challenge: &Challenge,
        solution: &Solution,
    ) -> Result<(Verdict, serde_json::Value), CrucibleError>;
}

/// Default registry — picks the right verifier by kind.
pub fn registry() -> Registry {
    let mut r = Registry::new();
    r.register(Box::new(ImageClassifyVerifier));
    r.register(Box::new(SemanticSimilarityVerifier));
    r.register(Box::new(AudioTranscribeVerifier));
    r.register(Box::new(MathArithmeticVerifier));
    r.register(Box::new(DrawingReconstructVerifier));
    r.register(Box::new(PromptInjectionDetectVerifier));
    r
}

/// Registry of verifiers, lookup by kind.
pub struct Registry {
    verifiers: Vec<Box<dyn Verifier>>,
}

impl Default for Registry {
    fn default() -> Self {
        registry()
    }
}

impl Registry {
    /// Empty.
    pub fn new() -> Self {
        Self {
            verifiers: Vec::new(),
        }
    }
    /// Register one verifier.
    pub fn register(&mut self, v: Box<dyn Verifier>) {
        self.verifiers.push(v);
    }
    /// Look up by kind.
    pub fn get(&self, kind: ChallengeKind) -> Option<&dyn Verifier> {
        self.verifiers
            .iter()
            .find(|v| v.kind() == kind)
            .map(|b| b.as_ref())
    }
    /// Verify any challenge by dispatching to the right verifier.
    pub fn verify(
        &self,
        challenge: &Challenge,
        solution: &Solution,
    ) -> Result<(Verdict, serde_json::Value), CrucibleError> {
        // Expiry check applies to every kind.
        if solution.submitted_at > challenge.expires_at {
            return Err(CrucibleError::Expired(challenge.id.clone()));
        }
        let v = self.get(challenge.kind).ok_or_else(|| {
            CrucibleError::Internal(format!("no verifier for {:?}", challenge.kind))
        })?;
        v.verify(challenge, solution)
    }
}

/// Image-classify verifier — stub.
pub struct ImageClassifyVerifier;
impl Verifier for ImageClassifyVerifier {
    fn kind(&self) -> ChallengeKind {
        ChallengeKind::ImageClassify
    }
    fn verify(
        &self,
        challenge: &Challenge,
        _solution: &Solution,
    ) -> Result<(Verdict, serde_json::Value), CrucibleError> {
        Ok((inconclusive(challenge), serde_json::Value::Null))
    }
}

/// Semantic-similarity verifier — stub.
pub struct SemanticSimilarityVerifier;
impl Verifier for SemanticSimilarityVerifier {
    fn kind(&self) -> ChallengeKind {
        ChallengeKind::SemanticSimilarity
    }
    fn verify(
        &self,
        challenge: &Challenge,
        _solution: &Solution,
    ) -> Result<(Verdict, serde_json::Value), CrucibleError> {
        Ok((inconclusive(challenge), serde_json::Value::Null))
    }
}

/// Audio-transcribe verifier — stub.
pub struct AudioTranscribeVerifier;
impl Verifier for AudioTranscribeVerifier {
    fn kind(&self) -> ChallengeKind {
        ChallengeKind::AudioTranscribe
    }
    fn verify(
        &self,
        challenge: &Challenge,
        _solution: &Solution,
    ) -> Result<(Verdict, serde_json::Value), CrucibleError> {
        Ok((inconclusive(challenge), serde_json::Value::Null))
    }
}

/// Math-arithmetic verifier — naive impl. The payload carries
/// `{"a": n, "op": "+|-|*", "b": n}`; the solution carries the
/// number. Exact match → Human; too-fast (< 800ms) → Bot;
/// otherwise Inconclusive.
pub struct MathArithmeticVerifier;
impl Verifier for MathArithmeticVerifier {
    fn kind(&self) -> ChallengeKind {
        ChallengeKind::MathArithmetic
    }
    fn verify(
        &self,
        challenge: &Challenge,
        solution: &Solution,
    ) -> Result<(Verdict, serde_json::Value), CrucibleError> {
        let a = challenge
            .payload
            .get("a")
            .and_then(|v| v.as_i64())
            .ok_or_else(|| CrucibleError::MalformedSolution("missing a".into()))?;
        let b = challenge
            .payload
            .get("b")
            .and_then(|v| v.as_i64())
            .ok_or_else(|| CrucibleError::MalformedSolution("missing b".into()))?;
        let op = challenge
            .payload
            .get("op")
            .and_then(|v| v.as_str())
            .ok_or_else(|| CrucibleError::MalformedSolution("missing op".into()))?;
        let truth: i64 = match op {
            "+" => a + b,
            "-" => a - b,
            "*" => a * b,
            other => {
                return Err(CrucibleError::MalformedSolution(format!("bad op {other}")));
            }
        };
        let got = solution
            .response
            .get("answer")
            .and_then(|v| v.as_i64())
            .ok_or_else(|| CrucibleError::MalformedSolution("missing answer".into()))?;
        let gt = serde_json::json!({"answer": truth});
        if got != truth {
            return Ok((
                Verdict::Bot {
                    confidence: 0.7,
                    reason: Some("wrong-answer".into()),
                },
                gt,
            ));
        }
        if solution.elapsed_ms < 800 {
            return Ok((
                Verdict::Bot {
                    confidence: 0.85,
                    reason: Some("too-fast".into()),
                },
                gt,
            ));
        }
        Ok((Verdict::Human { confidence: 0.9 }, gt))
    }
}

/// Drawing-reconstruct verifier — stub.
pub struct DrawingReconstructVerifier;
impl Verifier for DrawingReconstructVerifier {
    fn kind(&self) -> ChallengeKind {
        ChallengeKind::DrawingReconstruct
    }
    fn verify(
        &self,
        challenge: &Challenge,
        _solution: &Solution,
    ) -> Result<(Verdict, serde_json::Value), CrucibleError> {
        Ok((inconclusive(challenge), serde_json::Value::Null))
    }
}

/// Prompt-injection-detect verifier — stub.
pub struct PromptInjectionDetectVerifier;
impl Verifier for PromptInjectionDetectVerifier {
    fn kind(&self) -> ChallengeKind {
        ChallengeKind::PromptInjectionDetect
    }
    fn verify(
        &self,
        challenge: &Challenge,
        _solution: &Solution,
    ) -> Result<(Verdict, serde_json::Value), CrucibleError> {
        Ok((inconclusive(challenge), serde_json::Value::Null))
    }
}

fn inconclusive(challenge: &Challenge) -> Verdict {
    let retry_with = match challenge.difficulty {
        Difficulty::Easy => Difficulty::Medium,
        Difficulty::Medium => Difficulty::Hard,
        Difficulty::Hard | Difficulty::Adversarial => Difficulty::Adversarial,
    };
    Verdict::Inconclusive { retry_with }
}

#[cfg(test)]
mod tests {
    use super::*;
    use time::macros::datetime;

    fn challenge(kind: ChallengeKind, payload: serde_json::Value) -> Challenge {
        Challenge {
            id: "t-1".into(),
            kind,
            difficulty: Difficulty::Medium,
            payload,
            issued_at: datetime!(2026-05-19 00:00:00 UTC),
            expires_at: datetime!(2026-05-19 00:02:00 UTC),
            tenant_id: "acme".into(),
        }
    }

    fn solution(response: serde_json::Value, elapsed_ms: u32) -> Solution {
        Solution {
            challenge_id: "t-1".into(),
            response,
            submitted_at: datetime!(2026-05-19 00:00:05 UTC),
            elapsed_ms,
        }
    }

    #[test]
    fn registry_has_every_kind() {
        let r = registry();
        for k in [
            ChallengeKind::ImageClassify,
            ChallengeKind::SemanticSimilarity,
            ChallengeKind::AudioTranscribe,
            ChallengeKind::MathArithmetic,
            ChallengeKind::DrawingReconstruct,
            ChallengeKind::PromptInjectionDetect,
        ] {
            assert!(r.get(k).is_some(), "missing verifier for {k:?}");
        }
    }

    #[test]
    fn math_correct_answer_is_human() {
        let r = registry();
        let c = challenge(
            ChallengeKind::MathArithmetic,
            serde_json::json!({"a": 3, "op": "+", "b": 5}),
        );
        let s = solution(serde_json::json!({"answer": 8}), 2_500);
        let (v, gt) = r.verify(&c, &s).unwrap();
        assert!(matches!(v, Verdict::Human { .. }));
        assert_eq!(gt, serde_json::json!({"answer": 8}));
    }

    #[test]
    fn math_too_fast_is_bot() {
        let r = registry();
        let c = challenge(
            ChallengeKind::MathArithmetic,
            serde_json::json!({"a": 7, "op": "*", "b": 6}),
        );
        let s = solution(serde_json::json!({"answer": 42}), 200);
        let (v, _) = r.verify(&c, &s).unwrap();
        assert!(matches!(
            v,
            Verdict::Bot {
                reason: Some(_),
                ..
            }
        ));
    }

    #[test]
    fn math_wrong_answer_is_bot() {
        let r = registry();
        let c = challenge(
            ChallengeKind::MathArithmetic,
            serde_json::json!({"a": 10, "op": "-", "b": 4}),
        );
        let s = solution(serde_json::json!({"answer": 99}), 2_500);
        let (v, _) = r.verify(&c, &s).unwrap();
        assert!(matches!(v, Verdict::Bot { .. }));
    }

    #[test]
    fn expired_challenge_errors() {
        let r = registry();
        let c = challenge(
            ChallengeKind::MathArithmetic,
            serde_json::json!({"a": 1, "op": "+", "b": 1}),
        );
        let mut s = solution(serde_json::json!({"answer": 2}), 1_000);
        s.submitted_at = datetime!(2026-05-19 00:03:00 UTC); // past expiry
        assert!(matches!(r.verify(&c, &s), Err(CrucibleError::Expired(_))));
    }

    #[test]
    fn stub_verifiers_return_inconclusive() {
        let r = registry();
        for k in [
            ChallengeKind::ImageClassify,
            ChallengeKind::SemanticSimilarity,
            ChallengeKind::AudioTranscribe,
            ChallengeKind::DrawingReconstruct,
            ChallengeKind::PromptInjectionDetect,
        ] {
            let c = challenge(k, serde_json::Value::Null);
            let s = solution(serde_json::Value::Null, 1_000);
            let (v, _) = r.verify(&c, &s).unwrap();
            assert!(
                matches!(v, Verdict::Inconclusive { .. }),
                "{k:?} should be inconclusive in stub state"
            );
        }
    }
}
