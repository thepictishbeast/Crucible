# Crucible

**Bot-screening + LFI training-data generation, in one widget.**

Crucible is a multi-modal challenge platform that does two things
from the same interaction:

- **Stops bots** by asking them to do tasks humans find easy and
  machines find expensive or hard.
- **Generates labeled training data** for PlausiDen's open-source
  neurosymbolic AI ([PlausiDen-LFI](https://github.com/thepictishbeast/PlausiDen-LFI)).
  Every human-solved challenge produces a
  `(challenge, human_response, ground_truth, confidence)` tuple
  that flows into LFI's corpus.

The bot-gate **is** the training-data pipeline. The same
interaction that proves you're human grows our open-source AI's
knowledge.

## Why this exists

reCAPTCHA, hCaptcha, and Cloudflare Turnstile capture the labels
your users generate — and use them to train **their** vision
models. Crucible inverts that. Every challenge solved on a
PlausiDen-powered site trains **our** open-source AI. The
training data is auditable, attribution-bearing, and tenant-
private where it should be.

## Used by

Currently embedded on (or planned for):

- [Sacred.Vote](https://sacred.vote) — voter authenticity
- [plausiden.com](https://plausiden.com) — signup / form gate
- prosperityclub.com — member gate

Any site using PlausiDen-Forge can embed Crucible via a typed
widget; tenant config picks which challenge mix runs.

## Crates

| Crate                  | Role                                                         |
|------------------------|--------------------------------------------------------------|
| `crucible-core`        | Typed transport: `Challenge`, `Solution`, `Verdict`, `Difficulty`. |
| `crucible-challenges`  | Implementations: image-classify, semantic-similarity, audio-transcribe, math-arithmetic, drawing-reconstruct, prompt-injection-detect. |
| `crucible-corpus`      | Export of human-verified `(challenge, solution, ground-truth)` pairs to PlausiDen-LFI's `lfi-corpus` format. |

## Status

Scaffold. Typed surface defined; verifier impls + widget UI land
incrementally. The Crucible-to-LFI export shape is stable; the
challenges themselves grow per-PR.

## License

MIT OR Apache-2.0.
