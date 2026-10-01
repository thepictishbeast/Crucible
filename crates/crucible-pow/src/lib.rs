//! Proof-of-work gate for public forms, speaking the ALTCHA v2 protocol.
//!
//! WHY THIS EXISTS
//!
//! Crucible's own challenges are meant to be the gate. Until they are
//! proven, this one puts a real cost on every submission: a bot passed
//! plausiden.com's arithmetic gate twice in September 2026. ALTCHA is an
//! open-source (MIT) proof-of-work captcha. Its widget solves the challenge
//! in the visitor's browser and everything here runs on our own server, so
//! no third party sees the visitor.
//!
//! ## Shape
//!
//! The embedder mints a signed challenge when it renders the form
//! ([`mint`]) and inlines its JSON in the widget, so there is no extra
//! endpoint and no per-visitor state until a submission arrives. The widget
//! posts one field: a base64 JSON `{challenge, solution}` payload. [`check`]
//! verifies it and returns an id that the embedder burns in its own store,
//! so a solved challenge buys exactly one submission.
//!
//! ## What it accepts
//!
//! Only a proof-of-work payload signed with this key, unexpired, minted for
//! the same scope, with a correct solution. ALTCHA also defines a "server
//! signature" payload issued by its hosted service. That is refused:
//! accepting it would let a third party vouch for a visitor.
//!
//! Challenges are deterministic: the server picks the counter, so every
//! challenge costs the same known amount of work, and verification checks an
//! HMAC of the derived key instead of re-deriving it.

use altcha::{CreateChallengeOptions, HmacAlgorithm, Payload, VerifySolutionOptions};
use base64::engine::general_purpose::STANDARD;
use base64::Engine as _;
use rand::Rng as _;
use std::collections::BTreeMap;
use std::ops::RangeInclusive;
use std::time::{SystemTime, UNIX_EPOCH};

pub use altcha::Error as MintError;

/// Largest payload accepted, in base64 characters. A real payload is well
/// under 1 KiB; anything bigger is refused before it is decoded.
pub const MAX_PAYLOAD: usize = 8 * 1024;

/// The two HMAC secrets: one signs the challenge, one signs its derived key.
#[derive(Clone)]
pub struct PowKey {
    signature: String,
    key_signature: String,
}

impl std::fmt::Debug for PowKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PowKey").finish_non_exhaustive()
    }
}

impl PowKey {
    /// Derive both secrets from a secret the embedder already holds. Domain
    /// separation keeps them distinct from each other and from any other use
    /// of the same secret, so no new secret has to be provisioned.
    #[must_use]
    pub fn derive(secret: &[u8]) -> Self {
        let signature = blake3::derive_key("crucible-pow 2026-10-01 challenge signature", secret);
        let key_signature =
            blake3::derive_key("crucible-pow 2026-10-01 derived-key signature", secret);
        Self {
            signature: hex(&signature),
            key_signature: hex(&key_signature),
        }
    }

    /// Random secrets for the life of the process. A restart invalidates
    /// challenges minted before it, which fails closed.
    #[must_use]
    pub fn ephemeral() -> Self {
        let mut secret = [0u8; 32];
        rand::thread_rng().fill(&mut secret);
        Self::derive(&secret)
    }
}

/// How much work one challenge costs whoever solves it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cost {
    /// Key-derivation algorithm, as the widget names it.
    pub algorithm: &'static str,
    /// Hash iterations per attempt.
    pub iterations: u32,
    /// The counter is drawn from this range, and the solver counts up from
    /// zero until it finds it, so this is the number of attempts.
    pub attempts: RangeInclusive<u32>,
}

impl Cost {
    /// For a public form. 1.5 million PBKDF2 iterations on average: 0.17 s
    /// natively in a release build on one core (2026-10-01), which is the
    /// least a bot pays per submission. The widget solves while the visitor
    /// is still typing; its browser time is measured on the deployed form.
    pub const FORM: Self = Self {
        algorithm: "PBKDF2/SHA-256",
        iterations: 1_000,
        attempts: 1_000..=2_000,
    };

    /// Almost no work, for an embedder's tests: a debug build solves the
    /// form cost in seconds, and a suite submits many forms. The gate's
    /// own behaviour at the real cost is tested here.
    #[cfg(any(test, feature = "test-support"))]
    pub const TRIVIAL: Self = Self {
        algorithm: "PBKDF2/SHA-256",
        iterations: 1,
        attempts: 1..=2,
    };
}

/// A minted challenge, ready to inline in the widget's `challenge` attribute.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Issued {
    /// The signed challenge as JSON.
    pub challenge_json: String,
    /// Expiry, unix seconds.
    pub expires_at: u64,
}

/// Mint a challenge for `scope` (one name per form), valid for `ttl_secs`.
/// `now` is injected so tests can control the clock.
///
/// # Errors
/// Only if the key derivation itself fails, which a fixed, supported
/// algorithm does not.
pub fn mint_at(
    key: &PowKey,
    cost: &Cost,
    scope: &str,
    now: u64,
    ttl_secs: u64,
) -> Result<Issued, MintError> {
    let expires_at = now + ttl_secs;
    let counter = rand::thread_rng().gen_range(cost.attempts.clone());
    let data = BTreeMap::from([("scope".to_string(), serde_json::Value::from(scope))]);
    let challenge = altcha::create_challenge(CreateChallengeOptions {
        algorithm: cost.algorithm.to_string(),
        cost: cost.iterations,
        counter: Some(counter),
        data: Some(data),
        expires_at: Some(expires_at),
        hmac_signature_secret: Some(key.signature.clone()),
        hmac_key_signature_secret: Some(key.key_signature.clone()),
        ..Default::default()
    })?;
    let challenge_json = serde_json::to_string(&challenge).map_err(MintError::from)?;
    Ok(Issued {
        challenge_json,
        expires_at,
    })
}

/// Mint a challenge against the wall clock.
///
/// # Errors
/// As [`mint_at`].
pub fn mint(key: &PowKey, cost: &Cost, scope: &str, ttl_secs: u64) -> Result<Issued, MintError> {
    mint_at(key, cost, scope, unix_now(), ttl_secs)
}

/// Why a submission was refused. For logs only: telling a client which
/// check it failed helps it tune against that check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refusal {
    /// No payload at all, which is what a client without JavaScript sends.
    Missing,
    /// Longer than [`MAX_PAYLOAD`].
    TooLarge,
    /// Not base64, not JSON, or not a proof-of-work payload's shape.
    Malformed,
    /// A hosted-service "server signature" payload. Never accepted.
    NotProofOfWork,
    /// Not signed by this key, or edited after signing.
    BadSignature,
    /// Past its expiry, or minted without one.
    Expired,
    /// Minted for a different form.
    WrongScope,
    /// The work was not done.
    BadSolution,
}

impl Refusal {
    /// Stable kebab-case slug for logs and metrics.
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::Missing => "pow-missing",
            Self::TooLarge => "pow-too-large",
            Self::Malformed => "pow-malformed",
            Self::NotProofOfWork => "pow-not-proof-of-work",
            Self::BadSignature => "pow-bad-signature",
            Self::Expired => "pow-expired",
            Self::WrongScope => "pow-wrong-scope",
            Self::BadSolution => "pow-bad-solution",
        }
    }
}

/// A payload that verified: an id for the embedder to burn, and when the
/// challenge expires, which bounds how long the burn record must be kept.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Accepted {
    /// Unique per challenge; burn it so the solution is single-use.
    pub id: String,
    /// Expiry, unix seconds.
    pub expires_at: u64,
}

/// Check the widget's `payload` for `scope` at time `now`.
///
/// This does everything except burn the challenge: the caller owns the
/// store, must burn [`Accepted::id`], and must treat an id it has seen before
/// as a replay. Nothing in the payload is trusted before its signature
/// verifies.
///
/// # Errors
/// The [`Refusal`] that fired.
pub fn check_at(key: &PowKey, payload: &str, scope: &str, now: u64) -> Result<Accepted, Refusal> {
    let payload = payload.trim();
    if payload.is_empty() {
        return Err(Refusal::Missing);
    }
    if payload.len() > MAX_PAYLOAD {
        return Err(Refusal::TooLarge);
    }
    let bytes = STANDARD.decode(payload).map_err(|_| Refusal::Malformed)?;
    let value: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|_| Refusal::Malformed)?;
    if value.get("verificationData").is_some() {
        return Err(Refusal::NotProofOfWork);
    }
    let payload: Payload = serde_json::from_value(value).map_err(|_| Refusal::Malformed)?;
    let challenge = &payload.challenge;

    let signature = challenge
        .signature
        .as_deref()
        .ok_or(Refusal::BadSignature)?;
    let expected = altcha::sign_challenge(
        &HmacAlgorithm::Sha256,
        &mut challenge.parameters.clone(),
        None,
        &key.signature,
        None,
    )
    .map_err(|_| Refusal::Malformed)?;
    if !expected
        .signature
        .as_deref()
        .is_some_and(|e| ct_eq(e, signature))
    {
        return Err(Refusal::BadSignature);
    }

    let params = &challenge.parameters;
    let expires_at = params.expires_at.ok_or(Refusal::Expired)?;
    if now > expires_at {
        return Err(Refusal::Expired);
    }
    let minted_for = params
        .data
        .as_ref()
        .and_then(|d| d.get("scope"))
        .and_then(serde_json::Value::as_str);
    if minted_for != Some(scope) {
        return Err(Refusal::WrongScope);
    }

    let result = altcha::verify_solution(VerifySolutionOptions {
        hmac_key_signature_secret: Some(key.key_signature.clone()),
        ..VerifySolutionOptions::new(challenge, &payload.solution, key.signature.clone())
    })
    .map_err(|_| Refusal::Malformed)?;
    if result.expired {
        return Err(Refusal::Expired);
    }
    if !result.verified {
        return Err(Refusal::BadSolution);
    }
    Ok(Accepted {
        id: format!("pow:{signature}"),
        expires_at,
    })
}

/// Check against the wall clock.
///
/// # Errors
/// As [`check_at`].
pub fn check(key: &PowKey, payload: &str, scope: &str) -> Result<Accepted, Refusal> {
    check_at(key, payload, scope, unix_now())
}

/// Solve a challenge in-process and return the payload the widget would
/// post. For an embedder's tests: a form submission that must get past the
/// gate needs one.
///
/// # Panics
/// If `issued` is not a challenge this crate minted, or cannot be solved.
#[cfg(any(test, feature = "test-support"))]
#[must_use]
pub fn solve(issued: &Issued) -> String {
    let challenge: altcha::Challenge =
        serde_json::from_str(&issued.challenge_json).expect("a minted challenge");
    let solution = altcha::solve_challenge(altcha::SolveChallengeOptions::new(&challenge))
        .expect("solvable")
        .expect("solved before the timeout");
    let payload = serde_json::json!({ "challenge": challenge, "solution": solution });
    STANDARD.encode(payload.to_string())
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut s, b| {
            let _ = write!(s, "{b:02x}");
            s
        })
}

/// Constant-time comparison of two strings of our own making; mismatched
/// lengths short-circuit.
fn ct_eq(a: &str, b: &str) -> bool {
    a.len() == b.len() && a.bytes().zip(b.bytes()).fold(0u8, |d, (x, y)| d | (x ^ y)) == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    const TTL: u64 = 45 * 60;
    // The library also checks expiry against the real clock, so the tests'
    // clock starts from it rather than from a fixed date.
    #[allow(non_snake_case)]
    fn NOW() -> u64 {
        unix_now()
    }

    fn key() -> PowKey {
        PowKey::derive(b"a test secret that is not used anywhere")
    }

    fn issued() -> Issued {
        mint_at(&key(), &Cost::FORM, "contact", NOW(), TTL).expect("mint")
    }

    fn edit(payload: &str, f: impl FnOnce(&mut serde_json::Value)) -> String {
        let mut v: serde_json::Value =
            serde_json::from_slice(&STANDARD.decode(payload).unwrap()).unwrap();
        f(&mut v);
        STANDARD.encode(v.to_string())
    }

    #[test]
    fn a_solved_challenge_is_accepted_once_per_id() {
        let i = issued();
        let ok = check_at(&key(), &solve(&i), "contact", NOW() + 60).expect("accepted");
        assert!(ok.id.starts_with("pow:"));
        assert_eq!(ok.expires_at, i.expires_at);
        // The same payload yields the same id, which is what lets the
        // embedder's store refuse it the second time.
        let again = check_at(&key(), &solve(&i), "contact", NOW() + 61).expect("still valid");
        assert_eq!(ok.id, again.id);
    }

    #[test]
    fn every_challenge_has_its_own_id() {
        let (a, b) = (issued(), issued());
        let ida = check_at(&key(), &solve(&a), "contact", NOW()).unwrap().id;
        let idb = check_at(&key(), &solve(&b), "contact", NOW()).unwrap().id;
        assert_ne!(ida, idb);
    }

    #[test]
    fn nothing_and_garbage_are_refused() {
        assert_eq!(
            check_at(&key(), "", "contact", NOW()),
            Err(Refusal::Missing)
        );
        assert_eq!(
            check_at(&key(), "  \n", "contact", NOW()),
            Err(Refusal::Missing)
        );
        assert_eq!(
            check_at(&key(), &"A".repeat(MAX_PAYLOAD + 4), "contact", NOW()),
            Err(Refusal::TooLarge)
        );
        assert_eq!(
            check_at(&key(), "not base64 at all!", "contact", NOW()),
            Err(Refusal::Malformed)
        );
        assert_eq!(
            check_at(&key(), &STANDARD.encode("not json"), "contact", NOW()),
            Err(Refusal::Malformed)
        );
        assert_eq!(
            check_at(&key(), &STANDARD.encode("{}"), "contact", NOW()),
            Err(Refusal::Malformed)
        );
    }

    #[test]
    fn a_hosted_server_signature_payload_is_never_accepted() {
        let p = serde_json::json!({
            "algorithm": "SHA-256", "signature": "00", "verified": true,
            "verificationData": "classification=GOOD&verified=true",
        });
        assert_eq!(
            check_at(&key(), &STANDARD.encode(p.to_string()), "contact", NOW()),
            Err(Refusal::NotProofOfWork)
        );
    }

    #[test]
    fn another_key_or_an_edited_challenge_is_a_bad_signature() {
        let good = solve(&issued());
        let other = PowKey::derive(b"some other secret");
        assert_eq!(
            check_at(&other, &good, "contact", NOW()),
            Err(Refusal::BadSignature)
        );
        // Making the work cheaper after signing must not survive.
        let cheaper = edit(&good, |v| v["challenge"]["parameters"]["cost"] = 1.into());
        assert_eq!(
            check_at(&key(), &cheaper, "contact", NOW()),
            Err(Refusal::BadSignature)
        );
        // Nor may the expiry be pushed out, or the scope changed.
        let later = edit(&good, |v| {
            v["challenge"]["parameters"]["expiresAt"] = (NOW() * 2).into()
        });
        assert_eq!(
            check_at(&key(), &later, "contact", NOW()),
            Err(Refusal::BadSignature)
        );
        let unsigned = edit(&good, |v| {
            v["challenge"]["signature"] = serde_json::Value::Null
        });
        assert_eq!(
            check_at(&key(), &unsigned, "contact", NOW()),
            Err(Refusal::BadSignature)
        );
    }

    #[test]
    fn a_forged_payload_is_a_bad_signature_even_when_expired() {
        // Signature first: an unsigned claim never gets as far as being
        // reported as merely expired.
        let other = PowKey::derive(b"some other secret");
        let forged = solve(&mint_at(&other, &Cost::FORM, "contact", 1, 1).unwrap());
        assert_eq!(
            check_at(&key(), &forged, "contact", NOW()),
            Err(Refusal::BadSignature)
        );
    }

    #[test]
    fn expiry_and_scope_are_enforced() {
        let i = issued();
        let p = solve(&i);
        assert!(
            check_at(&key(), &p, "contact", i.expires_at).is_ok(),
            "valid up to and including expiry"
        );
        assert_eq!(
            check_at(&key(), &p, "contact", i.expires_at + 1),
            Err(Refusal::Expired)
        );
        assert_eq!(
            check_at(&key(), &p, "feedback", NOW()),
            Err(Refusal::WrongScope)
        );
    }

    #[test]
    fn work_not_done_is_a_bad_solution() {
        let p = solve(&issued());
        let wrong_key = edit(&p, |v| v["solution"]["derivedKey"] = "00".repeat(32).into());
        assert_eq!(
            check_at(&key(), &wrong_key, "contact", NOW()),
            Err(Refusal::BadSolution)
        );
        let not_hex = edit(&p, |v| v["solution"]["derivedKey"] = "zz".into());
        assert!(check_at(&key(), &not_hex, "contact", NOW()).is_err());
    }

    #[test]
    fn the_key_never_appears_in_debug_output() {
        let k = key();
        let shown = format!("{k:?}");
        assert!(
            !shown.contains(&k.signature) && !shown.contains(&k.key_signature),
            "{shown}"
        );
    }

    #[test]
    fn the_challenge_json_carries_what_the_widget_reads() {
        let v: serde_json::Value = serde_json::from_str(&issued().challenge_json).unwrap();
        let p = &v["parameters"];
        for field in [
            "algorithm",
            "cost",
            "keyLength",
            "keyPrefix",
            "keySignature",
            "nonce",
            "salt",
            "expiresAt",
        ] {
            assert!(!p[field].is_null(), "missing {field}: {v}");
        }
        assert_eq!(p["algorithm"], "PBKDF2/SHA-256");
        assert_eq!(p["data"]["scope"], "contact");
        assert!(v["signature"].is_string());
    }

    /// Native solve time for the form cost: a floor for what a bot pays per
    /// submission. Run with `--ignored --nocapture`.
    #[test]
    #[ignore = "a measurement, not a check"]
    fn measure_native_solve_time() {
        let mut total = 0.0;
        for _ in 0..5 {
            let start = std::time::Instant::now();
            let _ = solve(&issued());
            total += start.elapsed().as_secs_f64();
        }
        println!(
            "native solve, form cost: {:.3}s average over 5",
            total / 5.0
        );
    }
}
