# Handoff — for the next Crucible/LFI Claude session

This repo was scaffolded 2026-05-19 by the Forge-side instance.
Paul has a separate Claude focused on LFI/AI work; everything
below is theirs to evolve.

## What's here (updated 2026-05-20)

Five crates form the typed application surface:

```
crates/crucible-core        — Challenge / Solution / Verdict / Difficulty
crates/crucible-challenges  — All 6 verifier impls (real, not stubs)
crates/crucible-corpus      — CapturedTuple → CorpusPattern + write_corpus_dir
crates/crucible-server      — axum HTTP issuer + verify endpoints + ServerConfig
crates/crucible-widget      — Rust WASM widget for browser embedding
```

`cargo build --workspace` + `cargo test --workspace` clean.
Workspace test count: 77 (8 corpus + 37 challenges + 15 server lib
+ 5 server integration + 7 widget + 5 core).

## What's now real (was stub on initial commit)

All 6 verifier kinds implement honest discrimination logic:

  - `MathArithmetic`  — exact-match + elapsed-ms gate
  - `SemanticSimilarity` — set-overlap F1 + elapsed-ms gate
  - `ImageClassify`   — same F1 shape, 1200ms latency floor
  - `AudioTranscribe` — Levenshtein over normalized transcript
                        + per-kind elapsed-ms gate
  - `PromptInjectionDetect` — binary safe/unsafe + curator truth
  - `DrawingReconstruct` — connect-the-dots Euclidean + tap
                            tolerance ratio

Each emits `(challenge, solution, ground_truth, verdict,
attribution)` tuples via `crucible-corpus` for the LFI ingest
pipeline (PlausiDen-LFI#8 issue tracks the matching consumer).

## What's still pending

- **WASM artifact**: `crucible-widget` compiles as rlib on
  native targets (pure helpers covered by 7 tests). The cdylib
  half needs `wasm-pack build --target web` on a host with
  wasm32-unknown-unknown installed — not done yet because the
  build host this session lacks rustup + wasm32 target.
- **Forge CMS embed primitive**: PlausiDen-Loom PR #20
  (CmsSection::CrucibleChallenge) is open + awaiting paul's
  merge. Once merged, any Forge site can drop a
  {"kind":"crucible_challenge", ...} section to embed the
  widget against a crucible-server peer.
- **LFI consumer-side**: PlausiDen-LFI#8 is filed for the
  matching `read_corpus_dir()` reader. That repo's Claude
  owns the work per feedback_lfi_out_of_scope_for_this_instance.

## What this instance won't touch

- Any PR against PlausiDen-LFI's `lfi-corpus` crate — that's
  upstream of Crucible; Crucible only consumes it.

## What this instance won't touch

- Any of this repo's source. The Forge-side instance only
  scaffolded the typed surface so PlausiDen-Forge can compile
  against a stable contract.
- Any PR against PlausiDen-LFI's `lfi-corpus` crate — that's
  upstream of Crucible; Crucible only consumes it.

## Architecture context

The dual-purpose insight: every solved challenge produces a
labeled tuple `(challenge, human_response, ground_truth, t,
confidence)`. The `crucible-corpus` crate exports these to
PlausiDen-LFI's `lfi_corpus::Pattern` format so they enter the
LFI training pipeline.

Per `feedback_lfi_as_core_llm_as_peripheral`: LFI is the
upstream evaluator, the LLM is constrained candidate generator.
Crucible's contribution to that stack is **the labeled-data
pipeline** — without curated training data, LFI can't learn
new policies. Crucible bootstraps LFI's corpus from real human
responses at scale.

## Suggested next steps (updated 2026-05-20)

The 4 next-steps from the initial scaffold are DONE:

  ✓ verifier impls — all 6 kinds (was: "image-classify first")
  ✓ corpus export — write_corpus_dir + manifest pattern
  ✓ widget — crucible-widget crate, awaiting wasm-pack build
  ✓ tenant config — ServerConfig + JsonCuratedBank + env-driven
    bin

What's actually next:

1. **Build the WASM artifact**: install wasm-pack + the
   wasm32-unknown-unknown rustup target on a build host, run
   `wasm-pack build --target web crates/crucible-widget`.
   Generated bundle lands at
   `crates/crucible-widget/pkg/crucible_widget.js`.
2. **Merge Loom PR #20**: CmsSection::CrucibleChallenge slot
   so Forge sites can declare the embed via cms/*.json.
3. **Curator-authored bank files**: drop
   `<kind>.json` files into the directory `$CRUCIBLE_BANKS_DIR`
   points at, server picks them up on restart. Pattern shipped:
   one entry per kind with curator-supplied `payload` +
   `truth_indices` (or kind-specific truth field).
4. **Live deploy**: spin a crucible-serve under a reverse-proxy
   (caddy / nginx) at `/crucible/*`, mount the WASM bundle as
   a static asset, embed a `crucible_challenge` CmsSection on
   one Forge site. Verify the captured tuples flow through
   `crucible-corpus::write_corpus_dir` into the LFI ingest
   directory.

## Where to deploy

- **plausiden.com** — signup / contact-form gate
- **Sacred.Vote** — voter authenticity check
- **prosperityclub.com** — member-area gate

Each site embeds the widget via Forge's CMS surface (a typed
`Captcha` block — needs adding to `loom-cms-render::CmsSection`).

## Naming note

Paul didn't pick the name; the Forge-side instance proposed
"Crucible" — proving ground; fits both bot-screening and LFI-
training roles; available on the relevant platforms. If you
prefer a different name, rename freely — the crate names
follow the repo name only by convention.
