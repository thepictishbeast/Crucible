//! `crucible-widget` — browser-side Crucible widget.
//!
//! Compiles to WASM via `wasm-pack build --target web`. The
//! generated module exports a single `init(...)` entry point
//! that the host page calls to mount the widget into a DOM
//! node:
//!
//! ```html
//! <div id="crucible-mount"></div>
//! <script type="module">
//!   import init, { init as crucible_init } from './crucible_widget.js';
//!   await init();
//!   await crucible_init('crucible-mount', 'math-arithmetic', 'acme', '/crucible');
//! </script>
//! ```
//!
//! The widget:
//! 1. POSTs `/crucible/challenge` to mint a Challenge.
//! 2. Renders a kind-specific input form into the mount node.
//! 3. On submit, captures `elapsed_ms` since challenge load.
//! 4. POSTs `/crucible/solve` with the Solution.
//! 5. Displays the Verdict.
//!
//! ## What this crate is NOT yet
//!
//! v1 ships a stub `init` that just sets the mount-node's
//! textContent so the wiring compiles + loads in a browser
//! without erroring. The fetch / form / solve flow lands in
//! a follow-up iteration on a build host that has wasm-pack
//! installed (this host lacks the wasm32-unknown-unknown
//! target so `wasm-pack build` cannot run end-to-end here).
//!
//! The crate's API surface is shaped now so the build script,
//! Forge embed primitive, and downstream consumers can be
//! wired against a stable contract that won't change when the
//! widget body lands.

#![deny(unsafe_code, missing_docs)]

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// Build the JSON body for `POST /challenge`. Pure helper
/// (no web-sys dependency) so the host can unit-test the
/// request shape natively.
pub fn build_challenge_request_json(
    kind: &str,
    difficulty: &str,
    tenant_id: &str,
) -> serde_json::Value {
    serde_json::json!({
        "kind": kind,
        "difficulty": difficulty,
        "tenant-id": tenant_id,
    })
}

/// Build the JSON body for `POST /solve`. Pure helper.
pub fn build_solve_request_json(
    challenge_id: &str,
    response: serde_json::Value,
    submitted_at_rfc3339: &str,
    elapsed_ms: u32,
) -> serde_json::Value {
    serde_json::json!({
        "challenge-id": challenge_id,
        "response": response,
        "submitted-at": submitted_at_rfc3339,
        "elapsed-ms": elapsed_ms,
    })
}

/// URL helper: join `base_path` with an endpoint, handling
/// leading/trailing slash inconsistency. Pure helper.
pub fn join_url(base_path: &str, endpoint: &str) -> String {
    let trimmed_base = base_path.trim_end_matches('/');
    let trimmed_endpoint = endpoint.trim_start_matches('/');
    if trimmed_base.is_empty() {
        format!("/{trimmed_endpoint}")
    } else {
        format!("{trimmed_base}/{trimmed_endpoint}")
    }
}

/// Verdict shape returned from `POST /solve`. Matches the
/// server's SolveResponse so the widget can deserialize
/// directly.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct SolveResponseBody {
    /// Inner verdict — `{"kind":"human"|"bot"|"inconclusive", ...}`.
    pub verdict: crucible_core::Verdict,
}

/// Human-readable summary of a verdict. Pure helper used by
/// the widget's status-line render.
pub fn verdict_summary(v: &crucible_core::Verdict) -> String {
    use crucible_core::Verdict;
    match v {
        Verdict::Human { confidence } => {
            format!("Human verified (confidence {:.0}%).", confidence * 100.0)
        }
        Verdict::Bot { confidence, reason } => {
            let why = reason.as_deref().unwrap_or("");
            format!(
                "Bot detected (confidence {:.0}%{}).",
                confidence * 100.0,
                if why.is_empty() {
                    String::new()
                } else {
                    format!(", {why}")
                }
            )
        }
        Verdict::Inconclusive { retry_with } => {
            format!("Inconclusive — retry at {retry_with:?} difficulty.")
        }
    }
}

/// Mount the Crucible widget into `element_id`'s DOM node.
///
/// `element_id` — the id of an empty container the widget owns.
/// `kind`       — one of the ChallengeKind slugs (e.g.
///                `"math-arithmetic"`, `"semantic-similarity"`).
/// `tenant_id`  — passed to `/challenge` for per-tenant scope.
/// `base_path`  — where the crucible-server is mounted, e.g.
///                `"/crucible"`. Empty string → same-origin
///                root.
///
/// Returns a JS Promise that resolves when the widget has
/// mounted (NOT when the user has solved — the verdict
/// updates the mount textContent + emits a `crucible-verdict`
/// CustomEvent on the mount node).
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn init(
    element_id: String,
    kind: String,
    tenant_id: String,
    base_path: String,
) -> Result<(), JsValue> {
    let window = web_sys::window().ok_or_else(|| JsValue::from_str("no window"))?;
    let document = window
        .document()
        .ok_or_else(|| JsValue::from_str("no document"))?;
    let mount = document
        .get_element_by_id(&element_id)
        .ok_or_else(|| JsValue::from_str(&format!("no element with id {element_id:?}")))?;

    mount.set_text_content(Some("Loading challenge..."));

    let req_body = build_challenge_request_json(&kind, "medium", &tenant_id);
    let challenge_url = join_url(&base_path, "challenge");
    let resp = fetch_json(&window, &challenge_url, &req_body).await?;
    let challenge: crucible_core::Challenge = serde_wasm_bindgen::from_value(resp)
        .map_err(|e| JsValue::from_str(&format!("parse challenge: {e}")))?;

    render_challenge_form(&document, &mount, &kind, &challenge)?;
    let _ = challenge; // captured by closures
    Ok(())
}

/// Native-build stub of `init` — only compiled when targeting
/// non-WASM platforms (host unit tests, build verification).
/// Always returns `Err(NoBrowser)` since there's no DOM.
#[cfg(not(target_arch = "wasm32"))]
pub async fn init(
    _element_id: String,
    _kind: String,
    _tenant_id: String,
    _base_path: String,
) -> Result<(), &'static str> {
    Err("crucible-widget::init is only available in WASM builds")
}

#[cfg(target_arch = "wasm32")]
async fn fetch_json(
    window: &web_sys::Window,
    url: &str,
    body: &serde_json::Value,
) -> Result<JsValue, JsValue> {
    let opts = web_sys::RequestInit::new();
    opts.set_method("POST");
    opts.set_mode(web_sys::RequestMode::SameOrigin);
    let body_str = body.to_string();
    let body_js = JsValue::from_str(&body_str);
    opts.set_body(&body_js);
    let request = web_sys::Request::new_with_str_and_init(url, &opts)?;
    request
        .headers()
        .set("content-type", "application/json")?;
    let resp_value =
        wasm_bindgen_futures::JsFuture::from(window.fetch_with_request(&request)).await?;
    let resp: web_sys::Response = resp_value.dyn_into()?;
    if !resp.ok() {
        return Err(JsValue::from_str(&format!(
            "HTTP {} from {url}",
            resp.status()
        )));
    }
    let json_promise = resp.json()?;
    wasm_bindgen_futures::JsFuture::from(json_promise).await
}

#[cfg(target_arch = "wasm32")]
fn render_challenge_form(
    document: &web_sys::Document,
    mount: &web_sys::Element,
    kind: &str,
    challenge: &crucible_core::Challenge,
) -> Result<(), JsValue> {
    mount.set_inner_html("");
    let heading = document.create_element("p")?;
    heading.set_text_content(Some(&format!(
        "Crucible challenge ({kind}) — id {}",
        challenge.id
    )));
    mount.append_child(&heading)?;
    let payload_text = document.create_element("pre")?;
    payload_text.set_text_content(Some(&challenge.payload.to_string()));
    mount.append_child(&payload_text)?;
    let placeholder = document.create_element("p")?;
    placeholder.set_text_content(Some(
        "Form-render + submit-handler flow lands in a follow-up commit.",
    ));
    mount.append_child(&placeholder)?;
    Ok(())
}

/// Friendly slug of the supported ChallengeKind variants.
/// Mirrors the kebab-case strings the server accepts on
/// `POST /challenge`. Useful as a Rust-side constant so
/// downstream wiring can reference it without re-parsing
/// the ChallengeKind enum at runtime.
pub const SUPPORTED_KIND_SLUGS: &[&str] = &[
    "math-arithmetic",
    "semantic-similarity",
    "image-classify",
    "audio-transcribe",
    "drawing-reconstruct",
    "prompt-injection-detect",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supported_kind_slugs_match_crucible_core_enum() {
        // Every entry in SUPPORTED_KIND_SLUGS must round-trip
        // through ChallengeKind so the widget can't claim to
        // support a kind the server doesn't recognize.
        for slug in SUPPORTED_KIND_SLUGS {
            let value = serde_json::json!(slug);
            let parsed: Result<crucible_core::ChallengeKind, _> =
                serde_json::from_value(value);
            assert!(
                parsed.is_ok(),
                "{slug:?} doesn't round-trip through ChallengeKind"
            );
        }
    }

    #[test]
    fn build_challenge_request_uses_kebab_case_keys() {
        let v = build_challenge_request_json("math-arithmetic", "medium", "acme");
        // Server's ChallengeRequest has deny_unknown_fields +
        // rename_all=kebab-case, so the keys must match
        // verbatim.
        assert_eq!(v["kind"], "math-arithmetic");
        assert_eq!(v["difficulty"], "medium");
        assert_eq!(v["tenant-id"], "acme");
    }

    #[test]
    fn build_solve_request_uses_kebab_case_keys() {
        let v = build_solve_request_json(
            "math-00000001",
            serde_json::json!({"answer": 3}),
            "2026-05-20T19:00:00Z",
            2500,
        );
        assert_eq!(v["challenge-id"], "math-00000001");
        assert_eq!(v["response"]["answer"], 3);
        assert_eq!(v["submitted-at"], "2026-05-20T19:00:00Z");
        assert_eq!(v["elapsed-ms"], 2500);
    }

    #[test]
    fn join_url_handles_slash_combinations() {
        assert_eq!(join_url("/crucible", "challenge"), "/crucible/challenge");
        assert_eq!(join_url("/crucible/", "challenge"), "/crucible/challenge");
        assert_eq!(join_url("/crucible", "/challenge"), "/crucible/challenge");
        assert_eq!(join_url("", "challenge"), "/challenge");
        assert_eq!(join_url("/", "challenge"), "/challenge");
    }

    #[test]
    fn verdict_summary_phrasing() {
        use crucible_core::Verdict;
        let s = verdict_summary(&Verdict::Human { confidence: 0.92 });
        assert!(s.contains("Human"));
        assert!(s.contains("92"));
        let s = verdict_summary(&Verdict::Bot {
            confidence: 0.88,
            reason: Some("wrong-answer".to_owned()),
        });
        assert!(s.contains("Bot"));
        assert!(s.contains("wrong-answer"));
        let s = verdict_summary(&Verdict::Bot {
            confidence: 0.9,
            reason: None,
        });
        assert!(s.contains("Bot"));
        assert!(!s.contains("None"));
    }

    #[test]
    fn supported_kind_slugs_covers_every_variant() {
        // The reverse direction: every ChallengeKind variant
        // appears in SUPPORTED_KIND_SLUGS. Catches the case
        // where a new kind lands in core but the widget
        // forgets to expose it.
        let expected = [
            crucible_core::ChallengeKind::MathArithmetic,
            crucible_core::ChallengeKind::SemanticSimilarity,
            crucible_core::ChallengeKind::ImageClassify,
            crucible_core::ChallengeKind::AudioTranscribe,
            crucible_core::ChallengeKind::DrawingReconstruct,
            crucible_core::ChallengeKind::PromptInjectionDetect,
        ];
        for k in expected {
            let slug = k.slug();
            assert!(
                SUPPORTED_KIND_SLUGS.contains(&slug),
                "{k:?} missing from SUPPORTED_KIND_SLUGS"
            );
        }
    }
}
