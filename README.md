# Crucible

**Bot-screening + LFI training-data generation, from one interaction.**

Crucible is a multi-modal challenge platform that does two things
at once:

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
your users generate — and use them to train **their** models.
Crucible inverts that. Every challenge solved on a PlausiDen-powered
site trains **our** open-source AI. The training data is auditable,
attribution-bearing, and tenant-private where it should be.

There is a second reason, and on a privacy site it is the bigger
one: each of those three is a script fetched from a domain you do
not control. Embedding one breaks `default-src 'self'` and hands
your visitor's IP to a third party. Crucible runs on your own
infrastructure, or as a library in your own process.

## What actually stops a bot

Not the difficulty of the puzzle. Three properties, and an embedder
that drops any of them has a decoration rather than a gate:

1. **Unforgeable** — the challenge is bound to its answer by a MAC
   the embedder holds. Verify the MAC *before* parsing the payload;
   nothing inside a challenge is trustworthy until the signature
   says you issued it.
2. **Single-use** — one solved challenge buys one action. Without
   this, harvesting a single working (challenge, answer) pair and
   replaying it is the cheapest attack there is, and difficulty
   buys you nothing against it.
3. **Not instant** — every verifier enforces a per-kind latency
   floor (`MIN_ELAPSED_MS`) below which it returns `Bot`, because a
   script that fetches and answers in the same breath is not
   reading anything.

Difficulty is what stops a *targeted* solver. Arithmetic will not:
a script that parses `7 + 2`, computes, waits a second and submits
is a dozen lines. That is what the curated banks below are for.

## Challenge kinds

| Kind | Solver does | What it teaches LFI |
|---|---|---|
| `prompt-injection-detect` | Marks a message safe / unsafe | Labelled adversarial-input corpus |
| `semantic-similarity` | Picks which options mean the same as a prompt | Paraphrase pairs for the HDLM |
| `image-classify` | Picks images matching a description | Vision labels |
| `audio-transcribe` | Transcribes noisy audio | Audio HDC bind |
| `drawing-reconstruct` | Traces a shape | Pointer-stroke / gesture corpus |
| `math-arithmetic` | Adds two numbers | Nothing — see below |

All six have working verifiers. `math-arithmetic` is deliberately
the weakest: it exists for the very-low-difficulty tier and it
generates no training data worth having. Do not reach for it
because it is the easiest to render.

## Banks

A verifier grades a challenge; a **bank** supplies them.
`StaticMathBank` mints arithmetic procedurally. Everything else is
curated content, loaded from JSON by `JsonCuratedBank`:

    banks/prompt-injection.json      18 items, balanced 9 safe / 9 unsafe
    banks/semantic-similarity.json   12 items, 2 correct of 4 each

Both are written to be adversarial rather than easy. The
injection bank's "safe" half is full of messages that *talk about*
AI, system prompts and instructions without trying to redirect
anything, because a detector trained on obvious injections versus
unrelated small talk learns the topic and not the attack. The
semantic bank's distractors deliberately share most of their
vocabulary with the prompt, so surface-token overlap gives the
wrong answer and the label is only reachable from meaning.

Every item in every bank is graded by its real verifier in CI
(`crates/crucible-challenges/tests/banks.rs`): once with the
curator's answer, which must return `Human`, and once with the
opposite, which must not. That second grading is the one that
matters — a mislabelled item passes the first check happily.

Adding items is a JSON edit. Adding a *kind* means a verifier.

## Crates

| Crate | Role |
|---|---|
| `crucible-core` | Typed transport: `Challenge`, `Solution`, `Verdict`, `Difficulty`, `ChallengeKind`. |
| `crucible-challenges` | The six verifiers, plus the `Verifier`/`Registry` traits. |
| `crucible-server` | Axum service: `POST /challenge`, `POST /solve`. Banks (`StaticMathBank`, `JsonCuratedBank`, `MultiBank`), captured-tuple buffer, periodic flush. |
| `crucible-widget` | WASM browser widget for the hosted flow. |
| `crucible-corpus` | Export of human-verified tuples to PlausiDen-LFI's `lfi-corpus` format. |

## Two ways to embed

**As a service** — run `crucible-serve`, point the widget at it.
Use this when several sites share one bank, or when you want the
captured-tuple flush in one place.

**As a library** — depend on `crucible-core` + `crucible-challenges`,
mint and verify in your own request handler, and keep the answer in
a signed token. No JavaScript, no extra service, no CSP exception,
and no session store for anonymous visitors.

plausiden.com's contact form uses the library form; see
`src/crucible_gate.rs` there for a worked example, including the
token construction and the replay burn.

## Used by

- [plausiden.com](https://plausiden.com) — contact form, live since
  2026-09-10, library embed.

Planned: [Sacred.Vote](https://sacred.vote) (voter authenticity),
prosperityclub.com (member gate). Neither is wired yet — this list
says what is true, not what is intended.

## Status

Working, and in production on one site.

- 4,923 lines, 88 tests, no `todo!()` or `unimplemented!()`.
- All six verifiers implemented; two curated banks shipped.
- `~/.claude/gate/gate.sh` green (8 passed, 0 failed).

Known gaps, in rough order of how much they matter:

- **No per-kind bank for image-classify, audio-transcribe or
  drawing-reconstruct.** The verifiers work; nobody has authored
  the content, so in practice those kinds cannot be served.
- **No difficulty escalation.** `Verdict::Inconclusive` carries a
  `retry_with` and nothing consumes it yet.
- **Curator-authored truth only.** Intentional for screening — the
  discriminator is human-versus-script latency and correctness, not
  a model in the loop — but it caps bank size at what a person will
  write by hand.

## License

MIT OR Apache-2.0.
