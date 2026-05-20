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

/// Errors when writing a corpus pattern set to disk.
#[derive(Debug, thiserror::Error)]
pub enum WriteError {
    /// The target directory could not be created or written into.
    #[error("io error at {path}: {source}")]
    Io {
        /// Path where the failure happened.
        path: String,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// Pattern serialization failed.
    #[error("serialize pattern {slug}: {source}")]
    Serialize {
        /// Slug of the offending pattern.
        slug: String,
        /// Underlying serde error.
        #[source]
        source: serde_json::Error,
    },
}

/// Manifest of a corpus directory write.
///
/// Written to `<target>/index.json` so downstream consumers (LFI
/// ingest, audits, replication) can iterate the pattern set
/// without listing the directory. Slugs are emitted in
/// stable-sort order so the manifest is byte-deterministic across
/// runs with the same input set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct CorpusManifest {
    /// Format version. Bump on incompatible layout changes.
    pub version: u32,
    /// Crate that wrote the manifest. `"crucible-corpus"` here.
    pub writer: String,
    /// Stable kebab-case slugs of every pattern in this directory.
    pub patterns: Vec<String>,
}

/// Current manifest version. Bump when the on-disk layout
/// changes shape (renamed field, new required field, etc).
pub const MANIFEST_VERSION: u32 = 1;

/// Write a set of corpus patterns to `target_dir` as one
/// `<slug>.json` file per pattern, plus an `index.json`
/// manifest listing every slug. The target directory is
/// created if it doesn't exist.
///
/// Writes are atomic per-file (write to `.tmp.<pid>.<nanos>`
/// then rename) so a half-written corpus never serves to a
/// downstream ingest. The manifest is written LAST so a
/// consumer that finds a manifest can trust every listed slug
/// resolves; partial writes leave no manifest, and the consumer
/// treats the directory as in-flight.
///
/// Returns the manifest that was written.
pub fn write_corpus_dir(
    patterns: &[CorpusPattern],
    target_dir: &std::path::Path,
) -> Result<CorpusManifest, WriteError> {
    std::fs::create_dir_all(target_dir).map_err(|e| WriteError::Io {
        path: target_dir.display().to_string(),
        source: e,
    })?;

    let mut slugs: Vec<String> = Vec::with_capacity(patterns.len());
    for p in patterns {
        let body = serde_json::to_vec_pretty(p).map_err(|e| WriteError::Serialize {
            slug: p.slug.clone(),
            source: e,
        })?;
        let dest = target_dir.join(format!("{}.json", p.slug));
        atomic_write(&dest, &body)?;
        slugs.push(p.slug.clone());
    }
    slugs.sort();
    slugs.dedup();

    let manifest = CorpusManifest {
        version: MANIFEST_VERSION,
        writer: "crucible-corpus".to_owned(),
        patterns: slugs,
    };
    let body = serde_json::to_vec_pretty(&manifest).map_err(|e| WriteError::Serialize {
        slug: "index".to_owned(),
        source: e,
    })?;
    atomic_write(&target_dir.join("index.json"), &body)?;
    Ok(manifest)
}

fn atomic_write(dest: &std::path::Path, bytes: &[u8]) -> Result<(), WriteError> {
    let parent = dest.parent().ok_or_else(|| WriteError::Io {
        path: dest.display().to_string(),
        source: std::io::Error::new(std::io::ErrorKind::InvalidInput, "dest has no parent"),
    })?;
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let tmp_name = format!(
        ".{}.tmp.{pid}.{nanos}",
        dest.file_name().and_then(|s| s.to_str()).unwrap_or("out")
    );
    let tmp = parent.join(tmp_name);
    std::fs::write(&tmp, bytes).map_err(|e| WriteError::Io {
        path: tmp.display().to_string(),
        source: e,
    })?;
    std::fs::rename(&tmp, dest).map_err(|e| WriteError::Io {
        path: dest.display().to_string(),
        source: e,
    })?;
    Ok(())
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

    fn tmpdir(label: &str) -> std::path::PathBuf {
        let pid = std::process::id();
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        std::env::temp_dir().join(format!("crucible-corpus-{label}-{pid}-{n}"))
    }

    #[test]
    fn write_corpus_dir_emits_one_file_per_pattern() {
        let p1 = to_pattern(&human_tuple(AttributionPolicy::Curated)).unwrap();
        let mut p2 = p1.clone();
        p2.slug = "crucible-second".to_owned();
        let dir = tmpdir("emit");
        let manifest = write_corpus_dir(&[p1.clone(), p2.clone()], &dir).expect("write");
        assert_eq!(manifest.patterns.len(), 2);
        assert!(manifest.patterns.contains(&p1.slug));
        assert!(manifest.patterns.contains(&p2.slug));
        assert!(dir.join("crucible-abc-1.json").exists());
        assert!(dir.join("crucible-second.json").exists());
        assert!(dir.join("index.json").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_corpus_dir_roundtrips_pattern_payload() {
        let p = to_pattern(&human_tuple(AttributionPolicy::TenantPrivate)).unwrap();
        let dir = tmpdir("roundtrip");
        write_corpus_dir(&[p.clone()], &dir).expect("write");
        let read = std::fs::read_to_string(dir.join("crucible-abc-1.json")).expect("read");
        let parsed: CorpusPattern = serde_json::from_str(&read).expect("parse");
        assert_eq!(parsed, p);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_corpus_dir_writes_versioned_manifest_with_sorted_slugs() {
        let mut p1 = to_pattern(&human_tuple(AttributionPolicy::Curated)).unwrap();
        let mut p2 = p1.clone();
        let mut p3 = p1.clone();
        // Out-of-order slugs in the input — manifest must sort.
        p1.slug = "z-last".to_owned();
        p2.slug = "a-first".to_owned();
        p3.slug = "m-middle".to_owned();
        let dir = tmpdir("manifest");
        let manifest =
            write_corpus_dir(&[p1.clone(), p2.clone(), p3.clone()], &dir).expect("write");
        assert_eq!(manifest.version, MANIFEST_VERSION);
        assert_eq!(manifest.writer, "crucible-corpus");
        assert_eq!(
            manifest.patterns,
            vec![
                "a-first".to_owned(),
                "m-middle".to_owned(),
                "z-last".to_owned()
            ]
        );
        let read = std::fs::read_to_string(dir.join("index.json")).expect("read");
        let on_disk: CorpusManifest = serde_json::from_str(&read).expect("parse");
        assert_eq!(on_disk, manifest);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_corpus_dir_handles_missing_parent_dir() {
        let dir = tmpdir("nested").join("deep").join("nesting");
        let p = to_pattern(&human_tuple(AttributionPolicy::Curated)).unwrap();
        write_corpus_dir(&[p], &dir).expect("create + write");
        assert!(dir.join("index.json").exists());
        let _ = std::fs::remove_dir_all(dir.ancestors().nth(2).unwrap());
    }
}
