"""Prompt-injection detection for untrusted text (queries, articles, source passages).

Defence layers in this system (detection here is only one of them):
1. Pattern detection -> flag + log (query) or quarantine from index (source paragraph).
2. Untrusted text is wrapped in <passage>/<article>/<question> tags; the system prompt states
   that tag content is data, never instructions.
3. Constrained JSON decoding: the model cannot emit free-form text that bypasses the schema.
4. Code gates after generation (citations exist, quotes verbatim, numbers grounded). These hold
   even if 1-3 fail, because a successful injection still has to survive deterministic checks.

Edge cases NOT handled: obfuscated/encoded instructions (base64, homoglyphs, zero-width
characters beyond NFKC normalisation), non-English injections, instructions split across
paragraphs. Pattern list is a heuristic and will produce false negatives.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from cgprr.textnorm import normalise

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(rx, re.IGNORECASE))
    for name, rx in (
        ("override_instructions", r"\b(ignore|disregard|forget|override)\b.{0,40}\b(instruction|prompt|rule|document|source|passage|context)s?\b"),
        # "act as" / "you are now" / "pretend to be" only with an AI-role object: real prose says
        # "regolith could act as a thermal blanket" (a false positive found on a Wikipedia page).
        ("role_hijack", (r"\b(you are now|act as|pretend to be|behave as)\b\s+(an?\s+|the\s+|my\s+)?(\w+\s+){0,2}?"
                         r"(ai|assistant|chatbot|bot|language model|llm|model|gpt|dan|jailbroken|unrestricted|unfiltered|"
                         r"admin|administrator|developer|system|root|god mode|evil)\b"
                         r"|\b(new instructions|system prompt|developer mode|jailbreak mode)\b")),
        ("forced_output", r"\b(just say|only say|reply only|respond only|answer only) (yes|no|true|that)\b"),
        ("confirm_planted", r"\bconfirm (that|it)\b"),
        ("hidden_directive", r"\b(do not read anything below|automatically (approve|shortlist|accept|mark))\b"),
        ("tag_injection", r"</?\s*(passage|article|question|system|assistant|user)\b"),
        # Text addressed at the reader that asks for credentials or account verification (phishing).
        ("credential_request", (r"\b(enter|type|provide|send|share|submit|confirm|verify|update|reply with)\b.{0,30}"
                                r"\b(your|the|ur)\b.{0,20}\b(password|passcode|pass code|otp|one[- ]time (pass)?code|pin|cvv|"
                                r"api[- ]?key|login (details|credentials)|credentials|bank details|card details)\b")),
        ("phishing_verify", (r"\b(click|tap|visit|open)\b.{0,25}\b(link|here|below|url)\b.{0,40}\b(verify|confirm|unlock|"
                             r"reactivate|restore|claim)\b.{0,30}\b(account|identity|payment|prize|refund|kyc)\b")),
    )
)
# Rules that a preceding negation turns into factual/advisory text ("never share your OTP").
_NEGATABLE = {"credential_request", "phishing_verify"}
_NEGATION = re.compile(r"\b(never|not|don't|do not|dont|no one should|nobody should|warned against|urged .{0,20} not to|"
                       r"should not|must not|won't|will never|asks? you not)\b[^.]{0,40}$")


@dataclass(frozen=True)
class InjectionScan:
    suspected: bool
    rules: tuple[str, ...]


def _hit(name: str, rx: re.Pattern[str], t: str) -> bool:
    for m in rx.finditer(t):
        if name in _NEGATABLE and _NEGATION.search(t[max(0, m.start() - 60):m.start()]):
            continue  # advisory or reported speech, not a request
        return True
    return False


def scan(text: str) -> InjectionScan:
    t = normalise(text)
    hits = tuple(name for name, rx in _PATTERNS if _hit(name, rx, t))
    return InjectionScan(suspected=bool(hits), rules=hits)


def neutralise(text: str) -> str:
    """Strip characters that could close our delimiter tags inside untrusted text."""
    return text.replace("<", "‹").replace(">", "›")
