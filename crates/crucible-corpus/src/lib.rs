//! `crucible-corpus` — export Crucible captured tuples to
//! PlausiDen-LFI's corpus format.
//!
//! This is the seam where the bot-gate becomes a training-data
//! pipeline. Every `CapturedTuple` with `AttributionPolicy::Curated`
//! flows downstream into `lfi_corpus::Pattern` so PlausiDen-LFI's
//! evaluator learns from real human responses at scale.
//!
//! ## Honest-error contract
//!
//! Tuples with `AttributionPolicy::Ephemeral` are silently dropped
//! by the exporter — never recorded.
//!
//! Tuples with `AttributionPolicy::TenantPrivate` are exported but
//! the resulting `Pattern` carries the tenant id; downstream
//! corpus consumers MUST respect tenant isolation (per LFI's
//! per-tenant corpus rule).
//!
//! Tuples with `AttributionPolicy::Curated` are eligible for the
//! public LFI training corpus + carry contributor credit.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use crucible_core::{AttributionPolicy, CapturedTuple, ChallengeKind, Verdict};
use serde::{Deserialize, Serialize};

/// A corpus-ready training pattern derived from a CapturedTuple.
///
/// Mirrors PlausiDen-LFI's `lfi_corpus::Pattern` shape closely
/// enough that the downstream LFI ingest can rehydrate without
/// re-encoding the pattern body.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct CorpusPattern {
    /// Stable kebab-case slug — derived from the challenge id.
    pub slug: String,
    /// Description suitable for operator review.
    pub description: String,
    /// Challenge kind (informs which LFI sub-corpus this enters).
    pub kind: ChallengeKind,
    /// The raw challenge payload + the verified ground truth.
    /// LFI consumers HDC-encode this lazily on ingest.
    pub challenge_body: serde_json::Value,
    /// The human's solution (only present when verdict was Human).
    pub solution_body: Option<serde_json::Value>,
    /// Ground truth.
    pub ground_truth: serde_json::Value,
    /// Tenant scope (`None` = curated, else tenant id).
    pub tenant_id: Option<String>,
    /// Tags ("image-classify", "semantic-similarity", per-kind
    /// sub-tags).
    pub tags: Vec<String>,
}

/// Export error.
#[derive(Debug, thiserror::Error)]
pub enum ExportError {
    /// Verdict was not Human; we don't capture non-human patterns
    /// into the training corpus.
    #[error("verdict was not human; nothing to export")]
    NotHuman,
    /// Attribution forbids export.
    #[error("attribution policy forbids corpus export")]
    Ephemeral,
}

/// Convert a captured tuple to a corpus pattern, if eligible.
pub fn to_pattern(t: &CapturedTuple) -> Result<CorpusPattern, ExportError> {
    if matches!(t.attribution, AttributionPolicy::Ephemeral) {
        return Err(ExportError::Ephemeral);
    }
    let Verdict::Human { .. } = t.verdict else {
        return Err(ExportError::NotHuman);
    };
    let tenant_id = match t.attribution {
        AttributionPolicy::Curated => None,
        AttributionPolicy::TenantPrivate => Some(t.challenge.tenant_id.clone()),
        AttributionPolicy::Ephemeral => unreachable!(),
    };
    let mut tags = vec![t.challenge.kind.slug().to_string()];
    tags.push(format!("difficulty-{:?}", t.challenge.difficulty).to_lowercase());
    Ok(CorpusPattern {
        slug: format!("crucible-{}", t.challenge.id),
        description: format!(
            "Human-verified Crucible {} challenge",
            t.challenge.kind.slug()
        ),
        kind: t.challenge.kind,
        challenge_body: t.challenge.payload.clone(),
        solution_body: Some(t.solution.response.clone()),
        ground_truth: t.ground_truth.clone(),
        tenant_id,
        tags,
    })
}

/// Bulk convert. Returns one Ok / Err per input tuple, in order.
pub fn to_patterns(tuples: &[CapturedTuple]) -> Vec<Result<CorpusPattern, ExportError>> {
    tuples.iter().map(to_pattern).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crucible_core::{Challenge, ChallengeKind, Difficulty, Solution};
    use time::macros::datetime;

    fn human_tuple(attribution: AttributionPolicy) -> CapturedTuple {
        CapturedTuple {
            challenge: Challenge {
                id: "abc-1".into(),
                kind: ChallengeKind::SemanticSimilarity,
                difficulty: Difficulty::Medium,
                payload: serde_json::json!({"prompt": "synonyms"}),
                issued_at: datetime!(2026-05-19 00:00:00 UTC),
                expires_at: datetime!(2026-05-19 00:02:00 UTC),
                tenant_id: "acme".into(),
            },
            solution: Solution {
                challenge_id: "abc-1".into(),
                response: serde_json::json!({"picks": [0, 2]}),
                submitted_at: datetime!(2026-05-19 00:00:05 UTC),
                elapsed_ms: 3_400,
            },
            ground_truth: serde_json::json!({"picks": [0, 2]}),
            verdict: Verdict::Human { confidence: 0.93 },
            attribution,
        }
    }

    #[test]
    fn curated_tuple_exports_with_no_tenant() {
        let t = human_tuple(AttributionPolicy::Curated);
        let p = to_pattern(&t).unwrap();
        assert_eq!(p.tenant_id, None);
        assert_eq!(p.slug, "crucible-abc-1");
        assert!(p.tags.contains(&"semantic-similarity".to_string()));
    }

    #[test]
    fn tenant_private_exports_with_tenant() {
        let t = human_tuple(AttributionPolicy::TenantPrivate);
        let p = to_pattern(&t).unwrap();
        assert_eq!(p.tenant_id.as_deref(), Some("acme"));
    }

    #[test]
    fn ephemeral_refuses_export() {
        let t = human_tuple(AttributionPolicy::Ephemeral);
        assert!(matches!(to_pattern(&t), Err(ExportError::Ephemeral)));
    }

    #[test]
    fn non_human_refuses_export() {
        let mut t = human_tuple(AttributionPolicy::Curated);
        t.verdict = Verdict::Bot {
            confidence: 0.9,
            reason: None,
        };
        assert!(matches!(to_pattern(&t), Err(ExportError::NotHuman)));
    }
}
