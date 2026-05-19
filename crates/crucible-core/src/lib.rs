//! `crucible-core` — typed transport for the Crucible challenge
//! platform.
//!
//! Closed-enum types every embedder, server, and downstream
//! corpus consumer agrees on:
//!
//! - [`Challenge`] — a single screening task served to a session
//! - [`Solution`] — the response a user submits
//! - [`Verdict`] — typed outcome (Human / Bot / Inconclusive)
//! - [`Difficulty`] — typed ramp the verifier consults
//! - [`ChallengeKind`] — closed enum of multi-modal kinds
//! - [`AttributionPolicy`] — per-tenant capture/retention rule

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use serde::{Deserialize, Serialize};

/// Closed enum of the challenge kinds Crucible ships.
///
/// Adding a kind requires (a) implementing the verifier in
/// `crucible-challenges`, (b) extending `Challenge` with the new
/// variant's payload, (c) deciding the kind's contribution to
/// `crucible-corpus` (some kinds produce useful labels for LFI,
/// others are pure bot-gate filler).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ChallengeKind {
    /// Multi-image classification ("pick all photos that contain
    /// a bird"). Cheap to serve, hard for current vision models
    /// at adversarial difficulty.
    ImageClassify,
    /// Semantic-similarity ("which of these three sentences mean
    /// the same thing as the prompt?"). Trains LFI's HDLM.
    SemanticSimilarity,
    /// Audio transcription with realistic noise. Trains LFI's
    /// audio HDC bind.
    AudioTranscribe,
    /// Short arithmetic. Cheap human, cheap machine — useful
    /// only at the very-low-difficulty tier.
    MathArithmetic,
    /// Drawing-from-spec ("draw a circle inside the square").
    /// Generates pointer-stroke trajectories — useful labels for
    /// LFI's gesture corpus.
    DrawingReconstruct,
    /// Prompt-injection-detect: presented with a sentence that
    /// MAY contain an embedded prompt-injection payload, the
    /// user marks safe/unsafe. Trains LFI's adversarial-input
    /// detector.
    PromptInjectionDetect,
}

impl ChallengeKind {
    /// Stable kebab-case slug.
    pub fn slug(&self) -> &'static str {
        match self {
            Self::ImageClassify => "image-classify",
            Self::SemanticSimilarity => "semantic-similarity",
            Self::AudioTranscribe => "audio-transcribe",
            Self::MathArithmetic => "math-arithmetic",
            Self::DrawingReconstruct => "drawing-reconstruct",
            Self::PromptInjectionDetect => "prompt-injection-detect",
        }
    }

    /// Whether this kind produces useful labels for LFI's corpus.
    /// `MathArithmetic` returns false — humans solving 3+5 don't
    /// teach LFI anything new.
    pub fn trains_lfi(&self) -> bool {
        !matches!(self, Self::MathArithmetic)
    }
}

/// Difficulty ramp. Verifier consults to set scoring thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Difficulty {
    /// Easiest — high recall, lower precision.
    Easy,
    /// Default — balanced.
    Medium,
    /// Hard — adversarial-quality, used when prior signals
    /// suggest bot-likely.
    Hard,
    /// Maximum — last gate before refusing the action entirely.
    Adversarial,
}

/// Per-tenant attribution + retention rule for captured
/// (challenge, solution) tuples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AttributionPolicy {
    /// Tuple may enter the curated LFI corpus + be redistributed
    /// open-source. Strongest contribution to LFI; signs the
    /// user's role as a co-contributor in the corpus metadata.
    #[default]
    Curated,
    /// Tuple stays in this tenant's private corpus only.
    TenantPrivate,
    /// Tuple is discarded after the verdict is computed (used
    /// for high-privacy tenants who only want the bot-gate).
    Ephemeral,
}

/// A single challenge instance served to one session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct Challenge {
    /// Stable challenge id (kebab-case slug + nonce).
    pub id: String,
    /// What kind.
    pub kind: ChallengeKind,
    /// Difficulty ramp.
    pub difficulty: Difficulty,
    /// The payload — opaque JSON; per-kind crates know how to
    /// render + verify it.
    pub payload: serde_json::Value,
    /// When the challenge was issued (RFC 3339).
    pub issued_at: time::OffsetDateTime,
    /// Issuer-supplied expiry. Solutions submitted after this
    /// are auto-rejected.
    pub expires_at: time::OffsetDateTime,
    /// Tenant id this challenge was served for.
    pub tenant_id: String,
}

/// A user's response to a challenge.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct Solution {
    /// Challenge id the solution responds to.
    pub challenge_id: String,
    /// The response payload — opaque JSON; per-kind crates know
    /// how to compare it to ground truth.
    pub response: serde_json::Value,
    /// When the response was submitted.
    pub submitted_at: time::OffsetDateTime,
    /// Wall-clock time from issue to submit, ms. Verifier
    /// consults; impossibly-fast submissions weight toward Bot.
    pub elapsed_ms: u32,
}

/// Verdict.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum Verdict {
    /// Confidently human.
    Human {
        /// Verifier's confidence in [0.0, 1.0].
        confidence: f32,
    },
    /// Confidently bot. Refuse the gated action.
    Bot {
        /// Verifier's confidence in [0.0, 1.0].
        confidence: f32,
        /// Optional kebab-case slug of the signal that fired.
        reason: Option<String>,
    },
    /// Inconclusive. Reissue with higher difficulty.
    Inconclusive {
        /// Recommended next [`Difficulty`].
        retry_with: Difficulty,
    },
}

/// One captured tuple eligible for export to LFI's corpus.
///
/// This is the payload Crucible flushes downstream when a
/// human-attested challenge meets the tenant's AttributionPolicy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct CapturedTuple {
    /// The challenge that was served.
    pub challenge: Challenge,
    /// The solution submitted.
    pub solution: Solution,
    /// Verifier's ground-truth comparison.
    pub ground_truth: serde_json::Value,
    /// Verdict the verifier returned.
    pub verdict: Verdict,
    /// Attribution + retention rule from tenant config.
    pub attribution: AttributionPolicy,
}

/// Verifier errors.
#[derive(Debug, thiserror::Error)]
pub enum CrucibleError {
    /// Challenge expired.
    #[error("challenge expired (id={0})")]
    Expired(String),
    /// Solution malformed.
    #[error("solution malformed: {0}")]
    MalformedSolution(String),
    /// Internal verifier error.
    #[error("internal: {0}")]
    Internal(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use time::macros::datetime;

    fn now() -> time::OffsetDateTime {
        datetime!(2026-05-19 00:00:00 UTC)
    }

    #[test]
    fn kind_slugs_distinct() {
        let slugs = [
            ChallengeKind::ImageClassify,
            ChallengeKind::SemanticSimilarity,
            ChallengeKind::AudioTranscribe,
            ChallengeKind::MathArithmetic,
            ChallengeKind::DrawingReconstruct,
            ChallengeKind::PromptInjectionDetect,
        ];
        let mut seen = std::collections::HashSet::new();
        for k in slugs {
            assert!(seen.insert(k.slug()), "duplicate slug for {:?}", k);
        }
    }

    #[test]
    fn math_does_not_train_lfi() {
        assert!(!ChallengeKind::MathArithmetic.trains_lfi());
        assert!(ChallengeKind::SemanticSimilarity.trains_lfi());
        assert!(ChallengeKind::PromptInjectionDetect.trains_lfi());
    }

    #[test]
    fn difficulty_orders() {
        assert!(Difficulty::Easy < Difficulty::Medium);
        assert!(Difficulty::Medium < Difficulty::Hard);
        assert!(Difficulty::Hard < Difficulty::Adversarial);
    }

    #[test]
    fn challenge_serde_roundtrip() {
        let c = Challenge {
            id: "test-1".into(),
            kind: ChallengeKind::ImageClassify,
            difficulty: Difficulty::Medium,
            payload: serde_json::json!({"prompt": "bird"}),
            issued_at: now(),
            expires_at: now() + time::Duration::seconds(120),
            tenant_id: "acme".into(),
        };
        let j = serde_json::to_string(&c).unwrap();
        let back: Challenge = serde_json::from_str(&j).unwrap();
        assert_eq!(c, back);
    }

    #[test]
    fn verdict_variants_serialise() {
        let h = Verdict::Human { confidence: 0.92 };
        assert!(serde_json::to_string(&h).unwrap().contains("human"));
        let b = Verdict::Bot {
            confidence: 0.88,
            reason: Some("too-fast".into()),
        };
        assert!(serde_json::to_string(&b).unwrap().contains("bot"));
        let i = Verdict::Inconclusive {
            retry_with: Difficulty::Hard,
        };
        assert!(serde_json::to_string(&i).unwrap().contains("inconclusive"));
    }
}
