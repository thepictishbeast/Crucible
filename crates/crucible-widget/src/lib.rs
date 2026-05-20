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
#[cfg(target_arch = "wasm32")]
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

    render_challenge_form(&document, &mount, &kind, &challenge, &base_path)?;
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
    base_path: &str,
) -> Result<(), JsValue> {
    mount.set_inner_html("");
    match kind {
        "math-arithmetic" => render_math_form(document, mount, challenge, base_path)?,
        "prompt-injection-detect" => {
            render_injection_form(document, mount, challenge, base_path)?
        }
        "semantic-similarity" => {
            render_picks_form(document, mount, challenge, base_path, PicksKind::Text)?
        }
        "image-classify" => {
            render_picks_form(document, mount, challenge, base_path, PicksKind::Image)?
        }
        "audio-transcribe" => {
            render_audio_form(document, mount, challenge, base_path)?
        }
        other => {
            let p = document.create_element("p")?;
            p.set_text_content(Some(&format!(
                "Crucible widget: kind {other:?} form-render not implemented yet — \
                 raw payload: {}",
                challenge.payload
            )));
            mount.append_child(&p)?;
        }
    }
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn render_math_form(
    document: &web_sys::Document,
    mount: &web_sys::Element,
    challenge: &crucible_core::Challenge,
    base_path: &str,
) -> Result<(), JsValue> {
    use wasm_bindgen::JsCast;
    let a = challenge.payload.get("a").and_then(|v| v.as_i64()).unwrap_or(0);
    let b = challenge.payload.get("b").and_then(|v| v.as_i64()).unwrap_or(0);
    let op = challenge
        .payload
        .get("op")
        .and_then(|v| v.as_str())
        .unwrap_or("+");

    let heading = document.create_element("p")?;
    heading.set_text_content(Some(&math_prompt(a, op, b)));
    mount.append_child(&heading)?;

    let input: web_sys::HtmlInputElement = document
        .create_element("input")?
        .dyn_into()
        .map_err(|_| JsValue::from_str("input cast"))?;
    input.set_type("number");
    input.set_id("crucible-math-answer");
    mount.append_child(&input)?;

    let button: web_sys::HtmlButtonElement = document
        .create_element("button")?
        .dyn_into()
        .map_err(|_| JsValue::from_str("button cast"))?;
    button.set_text_content(Some("Submit"));
    mount.append_child(&button)?;

    let status = document.create_element("p")?;
    status.set_id("crucible-status");
    mount.append_child(&status)?;

    // Capture state for the submit handler.
    let challenge_id = challenge.id.clone();
    let mount_clone = mount.clone();
    let document_clone = document.clone();
    let base_path = base_path.to_owned();
    let load_time_ms = now_ms_or_zero();

    let closure = wasm_bindgen::closure::Closure::wrap(Box::new(move || {
        let challenge_id = challenge_id.clone();
        let mount = mount_clone.clone();
        let document = document_clone.clone();
        let base_path = base_path.clone();
        wasm_bindgen_futures::spawn_local(async move {
            if let Err(e) = submit_math(
                &document,
                &mount,
                &challenge_id,
                &base_path,
                load_time_ms,
            )
            .await
            {
                let _ = update_status(&document, &mount, &format!("Error: {e:?}"));
            }
        });
    }) as Box<dyn FnMut()>);
    button.set_onclick(Some(closure.as_ref().unchecked_ref()));
    // Leak the closure so the JS callback stays alive for the
    // lifetime of the widget. Cleanup happens on page unload.
    closure.forget();
    Ok(())
}

/// Format the math prompt as ASCII so the renderer doesn't depend
/// on any specific font supporting Unicode operator glyphs.
pub fn math_prompt(a: i64, op: &str, b: i64) -> String {
    format!("What is {a} {op} {b}?")
}

#[cfg(target_arch = "wasm32")]
fn render_injection_form(
    document: &web_sys::Document,
    mount: &web_sys::Element,
    challenge: &crucible_core::Challenge,
    base_path: &str,
) -> Result<(), JsValue> {
    use wasm_bindgen::JsCast;
    let prompt_text = challenge
        .payload
        .get("prompt")
        .and_then(|v| v.as_str())
        .unwrap_or("(no prompt)");

    let heading = document.create_element("p")?;
    heading.set_text_content(Some(
        "Is the following prompt safe or attempting prompt injection?",
    ));
    mount.append_child(&heading)?;

    let prompt_box = document.create_element("blockquote")?;
    prompt_box.set_text_content(Some(prompt_text));
    mount.append_child(&prompt_box)?;

    let safe_button: web_sys::HtmlButtonElement = document
        .create_element("button")?
        .dyn_into()
        .map_err(|_| JsValue::from_str("button cast"))?;
    safe_button.set_text_content(Some("Safe"));
    safe_button.set_id("crucible-inj-safe");
    mount.append_child(&safe_button)?;

    let unsafe_button: web_sys::HtmlButtonElement = document
        .create_element("button")?
        .dyn_into()
        .map_err(|_| JsValue::from_str("button cast"))?;
    unsafe_button.set_text_content(Some("Unsafe"));
    unsafe_button.set_id("crucible-inj-unsafe");
    mount.append_child(&unsafe_button)?;

    let status = document.create_element("p")?;
    status.set_id("crucible-status");
    mount.append_child(&status)?;

    let load_time_ms = now_ms_or_zero();
    let bind_button = |btn: &web_sys::HtmlButtonElement, verdict: &'static str| {
        let challenge_id = challenge.id.clone();
        let mount = mount.clone();
        let document = document.clone();
        let base_path = base_path.to_owned();
        let closure = wasm_bindgen::closure::Closure::wrap(Box::new(move || {
            let challenge_id = challenge_id.clone();
            let mount = mount.clone();
            let document = document.clone();
            let base_path = base_path.clone();
            wasm_bindgen_futures::spawn_local(async move {
                if let Err(e) = submit_injection(
                    &document,
                    &mount,
                    &challenge_id,
                    &base_path,
                    load_time_ms,
                    verdict,
                )
                .await
                {
                    let _ = update_status(&document, &mount, &format!("Error: {e:?}"));
                }
            });
        }) as Box<dyn FnMut()>);
        btn.set_onclick(Some(closure.as_ref().unchecked_ref()));
        closure.forget();
    };
    bind_button(&safe_button, "safe");
    bind_button(&unsafe_button, "unsafe");
    Ok(())
}

/// Discriminator for the shared picks-form renderer.
/// SemanticSimilarity uses Text labels; ImageClassify uses Image
/// thumbnails. Both produce the same `{"picks": [i,i,...]}` body
/// shape on submit.
#[cfg(target_arch = "wasm32")]
#[derive(Debug, Clone, Copy)]
enum PicksKind {
    Text,
    Image,
}

#[cfg(target_arch = "wasm32")]
fn render_picks_form(
    document: &web_sys::Document,
    mount: &web_sys::Element,
    challenge: &crucible_core::Challenge,
    base_path: &str,
    picks_kind: PicksKind,
) -> Result<(), JsValue> {
    use wasm_bindgen::JsCast;

    let prompt_text = challenge
        .payload
        .get("prompt")
        .and_then(|v| v.as_str())
        .unwrap_or("(no prompt)");
    let options: Vec<serde_json::Value> = challenge
        .payload
        .get(match picks_kind {
            PicksKind::Text => "options",
            PicksKind::Image => "image_urls",
        })
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let heading = document.create_element("p")?;
    heading.set_text_content(Some(prompt_text));
    mount.append_child(&heading)?;

    let list = document.create_element("ul")?;
    list.set_attribute("style", "list-style:none;padding:0;")?;
    for (i, opt) in options.iter().enumerate() {
        let li = document.create_element("li")?;
        let label = document.create_element("label")?;
        label.set_attribute("style", "display:inline-flex;gap:.5em;align-items:center;")?;
        let checkbox: web_sys::HtmlInputElement = document
            .create_element("input")?
            .dyn_into()
            .map_err(|_| JsValue::from_str("input cast"))?;
        checkbox.set_type("checkbox");
        checkbox.set_attribute("data-crucible-index", &i.to_string())?;
        checkbox.set_class_name("crucible-pick");
        label.append_child(&checkbox)?;
        match picks_kind {
            PicksKind::Text => {
                let text = opt.as_str().unwrap_or("").to_owned();
                let span = document.create_element("span")?;
                span.set_text_content(Some(&text));
                label.append_child(&span)?;
            }
            PicksKind::Image => {
                let src = opt.as_str().unwrap_or("").to_owned();
                let img = document.create_element("img")?;
                img.set_attribute("src", &src)?;
                img.set_attribute("alt", &format!("option {i}"))?;
                img.set_attribute("style", "max-width:120px;max-height:120px;")?;
                label.append_child(&img)?;
            }
        }
        li.append_child(&label)?;
        list.append_child(&li)?;
    }
    mount.append_child(&list)?;

    let button: web_sys::HtmlButtonElement = document
        .create_element("button")?
        .dyn_into()
        .map_err(|_| JsValue::from_str("button cast"))?;
    button.set_text_content(Some("Submit"));
    mount.append_child(&button)?;

    let status = document.create_element("p")?;
    status.set_id("crucible-status");
    mount.append_child(&status)?;

    let load_time_ms = now_ms_or_zero();
    let challenge_id = challenge.id.clone();
    let mount_clone = mount.clone();
    let document_clone = document.clone();
    let base_path = base_path.to_owned();

    let closure = wasm_bindgen::closure::Closure::wrap(Box::new(move || {
        let challenge_id = challenge_id.clone();
        let mount = mount_clone.clone();
        let document = document_clone.clone();
        let base_path = base_path.clone();
        wasm_bindgen_futures::spawn_local(async move {
            if let Err(e) =
                submit_picks(&document, &mount, &challenge_id, &base_path, load_time_ms).await
            {
                let _ = update_status(&document, &mount, &format!("Error: {e:?}"));
            }
        });
    }) as Box<dyn FnMut()>);
    button.set_onclick(Some(closure.as_ref().unchecked_ref()));
    closure.forget();
    Ok(())
}

#[cfg(target_arch = "wasm32")]
async fn submit_picks(
    document: &web_sys::Document,
    mount: &web_sys::Element,
    challenge_id: &str,
    base_path: &str,
    load_time_ms: f64,
) -> Result<(), JsValue> {
    use wasm_bindgen::JsCast;
    let elapsed_ms = (now_ms_or_zero() - load_time_ms).max(0.0) as u32;
    let submitted_at = now_iso_utc();
    let picks = collect_picks(mount)?;

    let body = build_solve_request_json(
        challenge_id,
        serde_json::json!({"picks": picks}),
        &submitted_at,
        elapsed_ms,
    );
    let url = join_url(base_path, "solve");
    let window = web_sys::window().ok_or_else(|| JsValue::from_str("no window"))?;
    let resp = fetch_json(&window, &url, &body).await?;
    let parsed: SolveResponseBody = serde_wasm_bindgen::from_value(resp)
        .map_err(|e| JsValue::from_str(&format!("parse verdict: {e}")))?;
    update_status(document, mount, &verdict_summary(&parsed.verdict))?;

    let event_init = web_sys::CustomEventInit::new();
    let detail = serde_wasm_bindgen::to_value(&parsed).unwrap_or(JsValue::NULL);
    event_init.set_detail(&detail);
    let event = web_sys::CustomEvent::new_with_event_init_dict(
        "crucible-verdict",
        &event_init,
    )?;
    mount.dispatch_event(&event)?;
    let _ = picks; // keep alive — already serialized into body
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn collect_picks(mount: &web_sys::Element) -> Result<Vec<i64>, JsValue> {
    use wasm_bindgen::JsCast;
    let nodes = mount.query_selector_all(".crucible-pick")?;
    let mut out = Vec::new();
    for i in 0..nodes.length() {
        let Some(node) = nodes.item(i) else { continue };
        let Ok(input) = node.dyn_into::<web_sys::HtmlInputElement>() else {
            continue;
        };
        if !input.checked() {
            continue;
        }
        let idx_attr = input
            .get_attribute("data-crucible-index")
            .unwrap_or_default();
        if let Ok(idx) = idx_attr.parse::<i64>() {
            out.push(idx);
        }
    }
    Ok(out)
}

#[cfg(target_arch = "wasm32")]
fn render_audio_form(
    document: &web_sys::Document,
    mount: &web_sys::Element,
    challenge: &crucible_core::Challenge,
    base_path: &str,
) -> Result<(), JsValue> {
    use wasm_bindgen::JsCast;
    let audio_url = challenge
        .payload
        .get("audio_url")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let heading = document.create_element("p")?;
    heading.set_text_content(Some("Listen and type what you hear:"));
    mount.append_child(&heading)?;

    let audio: web_sys::HtmlAudioElement = document
        .create_element("audio")?
        .dyn_into()
        .map_err(|_| JsValue::from_str("audio cast"))?;
    audio.set_src(audio_url);
    audio.set_controls(true);
    audio.set_attribute("preload", "metadata")?;
    mount.append_child(&audio)?;

    let input: web_sys::HtmlInputElement = document
        .create_element("input")?
        .dyn_into()
        .map_err(|_| JsValue::from_str("input cast"))?;
    input.set_type("text");
    input.set_id("crucible-audio-transcript");
    input.set_attribute("autocomplete", "off")?;
    input.set_attribute("spellcheck", "false")?;
    mount.append_child(&input)?;

    let button: web_sys::HtmlButtonElement = document
        .create_element("button")?
        .dyn_into()
        .map_err(|_| JsValue::from_str("button cast"))?;
    button.set_text_content(Some("Submit"));
    mount.append_child(&button)?;

    let status = document.create_element("p")?;
    status.set_id("crucible-status");
    mount.append_child(&status)?;

    let load_time_ms = now_ms_or_zero();
    let challenge_id = challenge.id.clone();
    let mount_clone = mount.clone();
    let document_clone = document.clone();
    let base_path = base_path.to_owned();

    let closure = wasm_bindgen::closure::Closure::wrap(Box::new(move || {
        let challenge_id = challenge_id.clone();
        let mount = mount_clone.clone();
        let document = document_clone.clone();
        let base_path = base_path.clone();
        wasm_bindgen_futures::spawn_local(async move {
            if let Err(e) =
                submit_audio(&document, &mount, &challenge_id, &base_path, load_time_ms).await
            {
                let _ = update_status(&document, &mount, &format!("Error: {e:?}"));
            }
        });
    }) as Box<dyn FnMut()>);
    button.set_onclick(Some(closure.as_ref().unchecked_ref()));
    closure.forget();
    Ok(())
}

#[cfg(target_arch = "wasm32")]
async fn submit_audio(
    document: &web_sys::Document,
    mount: &web_sys::Element,
    challenge_id: &str,
    base_path: &str,
    load_time_ms: f64,
) -> Result<(), JsValue> {
    use wasm_bindgen::JsCast;
    let input: web_sys::HtmlInputElement = document
        .get_element_by_id("crucible-audio-transcript")
        .ok_or_else(|| JsValue::from_str("no transcript input"))?
        .dyn_into()
        .map_err(|_| JsValue::from_str("input cast"))?;
    let transcript = input.value();
    let elapsed_ms = (now_ms_or_zero() - load_time_ms).max(0.0) as u32;
    let submitted_at = now_iso_utc();
    let body = build_solve_request_json(
        challenge_id,
        serde_json::json!({"transcript": transcript}),
        &submitted_at,
        elapsed_ms,
    );
    let url = join_url(base_path, "solve");
    let window = web_sys::window().ok_or_else(|| JsValue::from_str("no window"))?;
    let resp = fetch_json(&window, &url, &body).await?;
    let parsed: SolveResponseBody = serde_wasm_bindgen::from_value(resp)
        .map_err(|e| JsValue::from_str(&format!("parse verdict: {e}")))?;
    update_status(document, mount, &verdict_summary(&parsed.verdict))?;

    let event_init = web_sys::CustomEventInit::new();
    let detail = serde_wasm_bindgen::to_value(&parsed).unwrap_or(JsValue::NULL);
    event_init.set_detail(&detail);
    let event = web_sys::CustomEvent::new_with_event_init_dict(
        "crucible-verdict",
        &event_init,
    )?;
    mount.dispatch_event(&event)?;
    Ok(())
}

#[cfg(target_arch = "wasm32")]
async fn submit_injection(
    document: &web_sys::Document,
    mount: &web_sys::Element,
    challenge_id: &str,
    base_path: &str,
    load_time_ms: f64,
    verdict: &str,
) -> Result<(), JsValue> {
    let elapsed_ms = (now_ms_or_zero() - load_time_ms).max(0.0) as u32;
    let submitted_at = now_iso_utc();
    let body = build_solve_request_json(
        challenge_id,
        serde_json::json!({"verdict": verdict}),
        &submitted_at,
        elapsed_ms,
    );
    let url = join_url(base_path, "solve");
    let window = web_sys::window().ok_or_else(|| JsValue::from_str("no window"))?;
    let resp = fetch_json(&window, &url, &body).await?;
    let parsed: SolveResponseBody = serde_wasm_bindgen::from_value(resp)
        .map_err(|e| JsValue::from_str(&format!("parse verdict: {e}")))?;
    update_status(document, mount, &verdict_summary(&parsed.verdict))?;

    let event_init = web_sys::CustomEventInit::new();
    let detail = serde_wasm_bindgen::to_value(&parsed).unwrap_or(JsValue::NULL);
    event_init.set_detail(&detail);
    let event = web_sys::CustomEvent::new_with_event_init_dict(
        "crucible-verdict",
        &event_init,
    )?;
    mount.dispatch_event(&event)?;
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn now_ms_or_zero() -> f64 {
    web_sys::window()
        .and_then(|w| w.performance())
        .map(|p| p.now())
        .unwrap_or(0.0)
}

#[cfg(target_arch = "wasm32")]
async fn submit_math(
    document: &web_sys::Document,
    mount: &web_sys::Element,
    challenge_id: &str,
    base_path: &str,
    load_time_ms: f64,
) -> Result<(), JsValue> {
    use wasm_bindgen::JsCast;
    let input: web_sys::HtmlInputElement = document
        .get_element_by_id("crucible-math-answer")
        .ok_or_else(|| JsValue::from_str("no answer input"))?
        .dyn_into()
        .map_err(|_| JsValue::from_str("input cast"))?;
    let raw = input.value();
    let parsed: i64 = raw
        .trim()
        .parse()
        .map_err(|_| JsValue::from_str("answer must be a whole number"))?;

    let elapsed_ms = (now_ms_or_zero() - load_time_ms).max(0.0) as u32;
    let submitted_at = now_iso_utc();

    let body = build_solve_request_json(
        challenge_id,
        serde_json::json!({"answer": parsed}),
        &submitted_at,
        elapsed_ms,
    );
    let url = join_url(base_path, "solve");
    let window = web_sys::window().ok_or_else(|| JsValue::from_str("no window"))?;
    let resp = fetch_json(&window, &url, &body).await?;
    let parsed: SolveResponseBody = serde_wasm_bindgen::from_value(resp)
        .map_err(|e| JsValue::from_str(&format!("parse verdict: {e}")))?;
    update_status(document, mount, &verdict_summary(&parsed.verdict))?;

    // Emit a CustomEvent so host pages can react without
    // polling the mount DOM.
    let event_init = web_sys::CustomEventInit::new();
    let detail = serde_wasm_bindgen::to_value(&parsed)
        .unwrap_or(JsValue::NULL);
    event_init.set_detail(&detail);
    let event = web_sys::CustomEvent::new_with_event_init_dict(
        "crucible-verdict",
        &event_init,
    )?;
    mount.dispatch_event(&event)?;
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn update_status(
    _document: &web_sys::Document,
    mount: &web_sys::Element,
    msg: &str,
) -> Result<(), JsValue> {
    use wasm_bindgen::JsCast;
    if let Some(node) = mount
        .query_selector("#crucible-status")
        .ok()
        .flatten()
    {
        if let Ok(el) = node.dyn_into::<web_sys::HtmlElement>() {
            el.set_text_content(Some(msg));
        }
    }
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn now_iso_utc() -> String {
    // Best-effort wall-clock RFC 3339 string. Falls back to
    // a fixed epoch string if Date isn't available (shouldn't
    // happen in any real browser).
    let date = js_sys::Date::new_0();
    date.to_iso_string().as_string().unwrap_or_else(|| "1970-01-01T00:00:00.000Z".to_owned())
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
    fn math_prompt_renders_ascii() {
        assert_eq!(math_prompt(3, "+", 5), "What is 3 + 5?");
        assert_eq!(math_prompt(7, "*", 6), "What is 7 * 6?");
        assert_eq!(math_prompt(10, "-", 4), "What is 10 - 4?");
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
