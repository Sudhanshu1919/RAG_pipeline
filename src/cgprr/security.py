"""Content security for ingested chunks (layers 2-4; layer 1, file safety, is in `cgprr.loaders`).

Policy — facts are kept; what is removed is (a) text aimed at manipulating the model or the reader,
(b) secrets and personal identifiers, (c) operational instructions for serious harm, and (d) sexual
content involving minors. Subject matter alone (crime, violence, drugs, extremism, abuse, slurs
quoted in reporting) is never a reason to drop a fact: such chunks are indexed with a
`content_warning:<category>` label that travels with every citation.

| finding                                   | action                                          |
|-------------------------------------------|-------------------------------------------------|
| prompt injection / credential request     | quarantine (not indexed; human can release)     |
| API key, private key, password value      | redact in place  -> [REDACTED:secret]           |
| card (Luhn) / Aadhaar (Verhoeff) / email / phone | redact in place -> [REDACTED:<kind>]     |
| step-by-step weapon/explosive/malware/drug synthesis | quarantine                           |
| sexual content + minors (heuristic)       | quarantine for review                           |
| sexual content + minors (classifier S4)   | reject the whole file; nothing stored           |
| violence, crime, drugs, extremism, hate, self-harm, profanity | keep + label                |

Detectors here are deterministic (regex + checksums). An optional local classifier (Llama Guard 3
via Ollama, `classify.py`) runs at build time and can only make an outcome stricter or add labels;
it never releases a chunk the rules quarantined. Human decisions live in
`collections/<name>/review.json`, keyed by chunk content hash, and survive rebuilds.

Edge cases NOT handled: non-English text (patterns are English), obfuscated secrets (base64,
split across lines), images (no OCR). Keyword labels are a coarse signal, not a judgement.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from cgprr import guard
from cgprr.collections import Collection

QUARANTINE_INJECTION = {"override_instructions", "role_hijack", "hidden_directive", "tag_injection",
                        "credential_request", "phishing_verify"}

# --------------------------------------------------------------------------- layer 3: secrets / PII

_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple((n, re.compile(rx)) for n, rx in (
    ("private_key", r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z ]*PRIVATE KEY-----"),
    ("aws_key", r"\b(AKIA|ASIA)[0-9A-Z]{16}\b"),
    ("github_token", r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    ("slack_token", r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"),
    ("google_key", r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    ("api_key", r"\b(sk|pk|rk)[-_](live|test|proj|ant)?[-_]?[A-Za-z0-9_\-]{20,}\b"),
    ("jwt", r"\beyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\b"),
))
# "password: hunter2", "api_key = abcd..." -> keep the label, redact the value.
_ASSIGNED_SECRET = re.compile(
    r"(?i)\b(password|passwd|pwd|passcode|pass code|pin|otp|secret|api[_ -]?key|access[_ -]?token|auth[_ -]?token|token)"
    r"(\s*(?:is|=|:)\s*)([\"']?)([^\s\"',;]{4,})\3")
# Number boundaries: not glued to another digit and not part of a decimal / thousands figure ('.' or ','
# directly between digits, as in "11.2" or "1,200"); a number followed by ", " or ". " is its own token.
_NB = r"(?<!\d)(?<!\d[.,])"
_NA = r"(?!\d)(?![.,]\d)"
_CARD = re.compile(_NB + r"(?:\d[ -]?){12,18}\d" + _NA)
_AADHAAR = re.compile(_NB + r"[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}" + _NA)
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_PHONE = re.compile(
    r"(?<![\w.+])(?:\+\d{1,3}[ \-]?)?(?:\(?\d{2,5}\)?[ \-]?)?\d{3,5}[ \-]\d{4}(?!\w)(?![.,]\d)"
    r"|(?<![\w.+])(?:\+91[ \-]?)?[6-9]\d{4}[ \-]?\d{5}(?!\w)(?![.,]\d)")

_VERHOEFF_D = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5], [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
               [3, 4, 0, 1, 2, 8, 9, 5, 6, 7], [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
               [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3], [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
               [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
_VERHOEFF_P = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4], [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
               [8, 9, 1, 6, 0, 4, 3, 5, 2, 7], [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
               [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]]


def luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
        alt = not alt
    return total % 10 == 0


def verhoeff_ok(digits: str) -> bool:
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(ch)]]
    return c == 0


def redact(text: str, kinds: set[str]) -> tuple[str, list[str]]:
    """Return (redacted text, rules that fired). Kinds: secret, card, aadhaar, email, phone."""
    fired: list[str] = []
    out = text
    if "secret" in kinds:
        for name, rx in _SECRET_PATTERNS:
            out, n = rx.subn("[REDACTED:secret]", out)
            if n:
                fired.append(f"secret:{name}")
        out, n = _ASSIGNED_SECRET.subn(lambda m: f"{m.group(1)}{m.group(2)}[REDACTED:secret]", out)
        if n:
            fired.append("secret:assigned_value")
    if "aadhaar" in kinds:
        def _aad(m: re.Match[str]) -> str:
            d = re.sub(r"\D", "", m.group(0))
            return "[REDACTED:aadhaar]" if len(d) == 12 and verhoeff_ok(d) else m.group(0)
        new = _AADHAAR.sub(_aad, out)
        if new != out:
            fired.append("pii:aadhaar")
        out = new
    if "card" in kinds:
        def _card(m: re.Match[str]) -> str:
            d = re.sub(r"\D", "", m.group(0))
            return "[REDACTED:card]" if 13 <= len(d) <= 19 and luhn_ok(d) and len(set(d)) > 1 else m.group(0)
        new = _CARD.sub(_card, out)
        if new != out:
            fired.append("pii:card")
        out = new
    if "email" in kinds:
        out, n = _EMAIL.subn("[REDACTED:email]", out)
        if n:
            fired.append("pii:email")
    if "phone" in kinds:
        def _ph(m: re.Match[str]) -> str:
            d = re.sub(r"\D", "", m.group(0))
            return "[REDACTED:phone]" if 10 <= len(d) <= 13 else m.group(0)
        new = _PHONE.sub(_ph, out)
        if new != out:
            fired.append("pii:phone")
        out = new
    return out, fired


def leaks(text: str) -> list[str]:
    """Output-side check (gate 2): which secret/PII kinds appear in generated text."""
    return redact(text, {"secret", "card", "aadhaar", "email", "phone"})[1]


# --------------------------------------------------------------------------- layer 4: subject matter

_HARM_VERBS = r"(make|making|build|building|assembl\w*|synthesi[sz]\w*|manufactur\w*|cook\w*|prepar\w*|construct\w*|creat\w*|produc\w*|writ\w*)"
_HARM_OBJECTS = (r"(bomb|explosive|ied|pipe bomb|detonator|nerve agent|sarin|ricin|anthrax|chemical weapon|bioweapon|"
                 r"methamphetamine|meth|fentanyl|heroin|ghost gun|untraceable (gun|firearm)|silencer|suppressor|"
                 r"ransomware|keylogger|malware|botnet)")
# Operational = a harm object AND instruction form (numbered steps, or quantities + process verbs).
# Reporting ("he learned how to make bombs online") has the object but not the form: label only.
_HARM_OBJECT_RE = re.compile(rf"\b{_HARM_OBJECTS}s?\b", re.IGNORECASE)
_INSTRUCTIONAL = re.compile(rf"\b(step[- ]by[- ]step|instructions? (to|for)|how to|recipe for|guide to)\b.{{0,40}}\b{_HARM_VERBS}\b", re.IGNORECASE)
_STEP_MARKER = re.compile(r"(?:^|[\s(])(?:step|stage)\s*\d+\s*[:.)\-]|(?:^|\n)\s*\d{1,2}[.)]\s+[A-Z][a-z]+", re.IGNORECASE)
_QUANTITY_RECIPE = re.compile(r"\b\d+(\.\d+)?\s?(g|grams?|ml|millilit\w+|kg|oz|%|parts?)\b.{0,60}\b(mix|combine|heat|dissolve|add|grind|pack)\b", re.IGNORECASE)


def operational_harm(text: str) -> bool:
    if not _HARM_OBJECT_RE.search(text):
        return False
    steps = len(_STEP_MARKER.findall(text))
    return steps >= 2 or bool(_QUANTITY_RECIPE.search(text)) or (bool(_INSTRUCTIONAL.search(text)) and steps >= 1)

_SEXUAL = re.compile(r"\b(sexual(ly)?|sex|nude|naked|explicit|porn\w*|erotic\w*|intercourse)\b", re.IGNORECASE)
_MINOR = re.compile(r"\b(child|children|minor|minors|underage|under-age|kid|kids|boy|girl|teen|teenager|\d{1,2}[- ]year[- ]old)\b", re.IGNORECASE)
_REPORTING = re.compile(r"\b(arrest\w*|charged|convicted|sentenced|court|police|prosecut\w*|trial|accused|alleged|"
                        r"investigat\w*|abuse case|survivor|victim|ngo|law|act|report\w*)\b", re.IGNORECASE)

_PROTECTIVE = re.compile(r"\b(education|awareness|protection|safeguard\w*|safety|helpline|prevention|health)\b", re.IGNORECASE)

_LABELS: tuple[tuple[str, re.Pattern[str]], ...] = tuple((n, re.compile(rx, re.IGNORECASE)) for n, rx in (
    ("weapons", r"\b(bomb\w*|explosive\w*|ied|firearm\w*|rifle\w*|grenade\w*|missile\w*|chemical weapon\w*)\b"),
    ("violence", r"\b(killed|murder\w*|shot dead|stabb\w*|massacre|assault\w*|bombing|blast|attack(ed|s)?|wounded|lynch\w*)\b"),
    ("crime", r"\b(fraud|scam\w*|theft|robbery|smuggl\w*|trafficking|bribe\w*|extortion|money laundering|arrested|charged with)\b"),
    ("drugs", r"\b(narcotic\w*|cocaine|heroin|methamphetamine|fentanyl|opioid\w*|drug (seizure|haul|cartel|trafficking))\b"),
    ("extremism", r"\b(terroris\w*|extremis\w*|jihad\w*|insurgen\w*|militant\w*|radicali[sz]\w*)\b"),
    ("hate", r"\b(hate speech|racial slur|slur|casteist|communal (violence|riot)|bigot\w*)\b"),
    ("self_harm", r"\b(suicide|self[- ]harm|took (his|her|their) own life)\b"),
    ("sexual_violence", r"\b(rape\w*|sexual assault|molest\w*|sexual abuse)\b"),
    ("profanity", r"\b(fuck\w*|shit\w*|bastard\w*|bitch\w*|asshole\w*|motherfuck\w*)\b"),
))


@dataclass
class ScreenResult:
    text: str
    action: str = "keep"  # keep | quarantine | reject_file
    rules: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    released: bool = False

    def action_for(self, rule: str) -> str:
        if rule.startswith(("secret:", "pii:")):
            return "redacted"
        if rule.startswith("label:"):
            return "labelled"
        if self.released:
            return "released_by_review"
        return {"keep": "flagged", "quarantine": "quarantined", "reject_file": "rejected"}[self.action]


def chunk_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()[:16]


def load_review(coll: Collection) -> dict[str, str]:
    p = coll.root / "review.json"
    if not p.is_file():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in data.get("decisions", {}).items()}


def save_review(coll: Collection, decisions: dict[str, str]) -> Path:
    p = coll.root / "review.json"
    p.write_text(json.dumps({"decisions": decisions}, indent=2, sort_keys=True), encoding="utf-8")
    return p


class Screener:
    """Deterministic per-chunk screening. Collection options (collection.toml):
    redact = ["secret", "card", "aadhaar", "email", "phone"]   (default: all)
    mask_profanity = false
    """

    def __init__(self, coll: Collection | None = None) -> None:
        extra = coll.extra if coll is not None else {}
        kinds = extra.get("redact", ["secret", "card", "aadhaar", "email", "phone"])
        self.redact_kinds = {str(k) for k in kinds} if isinstance(kinds, list) else {"secret"}
        self.redact_kinds.add("secret")  # secrets are always redacted, whatever the collection says
        self.mask_profanity = bool(extra.get("mask_profanity", False))
        self.review = load_review(coll) if coll is not None else {}

    def screen(self, text: str) -> ScreenResult:
        rules: list[str] = []
        action = "keep"

        inj = guard.scan(text)
        hits = [r for r in inj.rules if r in QUARANTINE_INJECTION]
        if hits:
            rules += [f"injection:{r}" for r in hits]
            action = "quarantine"

        if operational_harm(text):
            rules.append("harm:operational_instructions")
            action = "quarantine"

        for sent in re.split(r"(?<=[.!?])\s+", text):
            if _SEXUAL.search(sent) and _MINOR.search(sent) and not _REPORTING.search(sent) and not _PROTECTIVE.search(sent):
                rules.append("child_safety:review")
                action = "quarantine"
                break

        labels = [f"content_warning:{name}" for name, rx in _LABELS if rx.search(text)]
        h = chunk_hash(text)
        released = False
        if action == "quarantine" and self.review.get(h) == "release":
            action, released = "keep", True

        out, red = redact(text, self.redact_kinds)
        rules += red
        if self.mask_profanity and "content_warning:profanity" in labels:
            out = _LABELS[-1][1].sub(lambda m: m.group(0)[0] + "*" * (len(m.group(0)) - 1), out)
        rules += [f"label:{lb.split(':', 1)[1]}" for lb in labels]
        return ScreenResult(text=out, action=action, rules=rules, labels=labels, released=released)


def quarantine_record(path: Path, rec: dict[str, object]) -> None:
    """Quarantined chunks are kept for human review in the index folder, outside the searchable index."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
