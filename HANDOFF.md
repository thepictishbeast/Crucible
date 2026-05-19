# Handoff — for the next Crucible/LFI Claude session

This repo was scaffolded 2026-05-19 by the Forge-side instance.
Paul has a separate Claude focused on LFI/AI work; everything
below is theirs to evolve.

## What's here

Three crates form the typed application surface:

```
crates/crucible-core        — Challenge / Solution / Verdict / Difficulty
crates/crucible-challenges  — Challenge implementations (kinds, verifiers)
crates/crucible-corpus      — Export to PlausiDen-LFI's lfi-corpus format
```

`cargo build --workspace` + `cargo test --workspace` clean as of
the initial commit.

## What's NOT here

- No web frontend / widget yet. The crates expose a typed
  server-side API; the embeddable JS widget lands later.
- No verifier impls — `Challenge::verify` returns `Inconclusive`
  for every challenge kind (the typed shape is fixed; algorithms
  fill in).
- No actual challenge content. Each kind ships an example shape;
  curators add real challenges.

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

## Suggested next steps

1. Implement the first verifier — likely `image-classify` since
   the algorithm is well-understood + the dataset can come from
   already-public image-classification benchmarks.
2. Wire `crucible-corpus::export` to actually call into
   `lfi_corpus::Corpus::add_pattern` with HDC-encoded entries.
3. Build the embeddable widget (separate workspace member;
   probably WASM-compiled Rust with a thin JS adapter).
4. Add tenant config (per-tenant challenge mix, attribution
   policy, difficulty ramp).

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
