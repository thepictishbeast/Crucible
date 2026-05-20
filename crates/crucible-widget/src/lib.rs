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

use wasm_bindgen::prelude::*;

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
/// mounted (NOT when the user has solved — the verdict event
/// fires separately on the mount node).
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

    // v1 stub: write a friendly placeholder so the host page
    // can confirm the WASM loaded and the widget reached its
    // mount. Subsequent iterations replace this with the
    // challenge-fetch + render-form + solve-post flow.
    mount.set_text_content(Some(&format!(
        "Crucible widget loaded — kind={kind} tenant={tenant_id} base={base_path}"
    )));
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
