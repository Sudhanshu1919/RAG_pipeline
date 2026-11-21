"""Deterministic text utilities: tokenisation, query-term coverage, entity candidates,
and number extraction/normalisation used by the hallucination gates.

Edge cases NOT handled (documented limits):
- Numbers written as words above "twenty" (e.g. "forty-two") are not extracted.
- Fractions ("three quarters"), ranges ("5-7%") are split into separate numbers.
- Unit conversion across systems (metres vs feet, USD vs INR) is not attempted.
- "one" is excluded from word-number extraction (too often a pronoun -> false refusals).
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import snowballstemmer

_STEMMER = snowballstemmer.stemmer("english")

STOPWORDS: frozenset[str] = frozenset(
    ["a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "at", "by", "for", "with", "from", "as", "is", "are", "was", "were", "be", "been", "being", "it", "its", "this", "that", "these", "those", "there", "their", "them", "they", "he", "she", "his", "her", "we", "our", "you", "your", "i", "me", "my", "do", "does", "did", "doing", "done", "have", "has", "had", "having", "will", "would", "shall", "should", "can", "could", "may", "might", "must", "what", "which", "who", "whom", "whose", "when", "where", "why", "how", "than", "then", "so", "such", "not", "no", "nor", "only", "own", "same", "too", "very", "just", "also", "about", "into", "over", "under", "again", "further", "once", "all", "any", "both", "each", "few", "more", "most", "other", "some", "up", "down", "out", "off", "above", "below", "between", "through", "during", "before", "after", "said", "says", "say", "per", "much", "many", "right", "yes", "ignore", "confirm", "tell", "please"]
)

# Passage ids ("S03:p1", "kotp-press-3fa21c:p2") and prompt-local labels ("P3", "[P3]") are stripped
# before number/entity checks, so citation digits are never mistaken for claimed figures.
_PID_RE = re.compile(r"\b[A-Za-z0-9][A-Za-z0-9_./-]{0,150}:p\d+\b|\[?\bP\d{1,2}\b\]?")
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']+")

_MULT = {
    "thousand": 1e3, "k": 1e3,
    "lakh": 1e5, "lakhs": 1e5,
    "million": 1e6, "mn": 1e6, "m": 1e6,
    "crore": 1e7, "crores": 1e7,
    "billion": 1e9, "bn": 1e9, "b": 1e9,
    "trillion": 1e12, "tn": 1e12,
}
_WORD_NUM = {
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
}
_NUM_RE = re.compile(
    r"(?<![\d.,])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"
    r"(?:\s?(%|per\s?cent|percent))?"
    r"(?:\s?(thousand|lakhs?|million|mn|crores?|billion|bn|trillion|tn|k|m|b)\b)?",
    re.IGNORECASE,
)
_WORD_NUM_RE = re.compile(r"\b(" + "|".join(_WORD_NUM) + r")\b", re.IGNORECASE)

_GENERIC_CAPS: frozenset[str] = frozenset(
    ["usd", "inr", "eur", "gdp", "teu", "cpi", "fy", "q1", "q2", "q3", "q4", "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "i"]
)


# Terminal/control characters (ANSI escapes, BEL, NUL...) and invisible format characters that can hide
# text from a human reader (zero-width space, word joiner, BOM, soft hyphen, bidi overrides/isolates).
# ZWJ/ZWNJ (U+200C/U+200D) are kept: Indic scripts need them.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_INVISIBLE_RE = re.compile(r"[\u00ad\u200b\u2060-\u2064\ufeff\u202a-\u202e\u2066-\u2069]")


def clean_text(text: str) -> str:
    """Remove control and invisible characters from untrusted document text (kept: \\t \\n \\r).

    Why: an ANSI escape in a document would otherwise be printed raw by `cgprr ask` / `collection review`
    (terminal injection), and zero-width characters split words so injection patterns do not match."""
    return _INVISIBLE_RE.sub("", _CONTROL_RE.sub("", text))


def normalise(text: str) -> str:
    """Canonical form for exact-quote comparison: NFKC, unified quotes/dashes, collapsed space, casefold."""
    t = unicodedata.normalize("NFKC", clean_text(text)).replace("\u200c", "").replace("\u200d", "")
    t = t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    t = re.sub(r"[‐-―]", "-", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t.casefold()


def strip_citations(text: str) -> str:
    return _PID_RE.sub(" ", text)


@lru_cache(maxsize=50_000)
def _stem(word: str) -> str:
    return str(_STEMMER.stemWord(word))


def tokenize(text: str) -> list[str]:
    """BM25 tokens: lowercase alnum (numbers keep internal '.' and ','), stopwords removed, stemmed."""
    toks = _TOKEN_RE.findall(normalise(strip_citations(text)).replace("'s ", " "))
    return [_stem(t) if not t[0].isdigit() else t.replace(",", "") for t in toks if t not in STOPWORDS]


def content_terms(text: str) -> set[str]:
    return set(tokenize(text))


def coverage(query: str, passage_tokens: set[str]) -> float:
    """Fraction of the query's content terms present in the passage. 0.0 if query has none."""
    q = content_terms(query)
    if not q:
        return 0.0
    return len(q & passage_tokens) / len(q)


def entity_candidates(text: str) -> set[str]:
    """Capitalised tokens that are not sentence-initial: cheap proper-noun proxy for gate 1."""
    out: set[str] = set()
    for sent in re.split(r"(?<=[.!?])\s+", strip_citations(text)):
        words = _WORD_RE.findall(sent)
        for i, w in enumerate(words):
            if i == 0 or not w[0].isupper():
                continue
            # Check in the same token space as the vocabulary: `tokenize` splits on '-' and "'", so
            # "Chandrayaan-3", "WSJ-only" or "O'Brien" must be split too, or a name that is plainly in
            # the documents is reported as unknown (measured: 7/28 false refusals on public-sample).
            # Keep the first part and any capitalised later part ("Kestria-based" -> "kestria").
            for j, part in enumerate(re.split(r"[-']", w.removesuffix("'s"))):
                if not part or (j > 0 and not part[0].isupper()):
                    continue
                lw = part.lower()
                if len(lw) < 3 or lw in STOPWORDS or lw in _GENERIC_CAPS:
                    continue
                out.add(_stem(lw))
    return out


def _canon(v: float) -> float:
    return float(f"{v:.6g}")


def extract_numbers(text: str) -> list[tuple[str, frozenset[float]]]:
    """Return (surface, canonical values). Each number yields its bare value and, when a
    magnitude word follows, its scaled value, so "USD 450 million" matches "$450m" and "450"."""
    clean = strip_citations(text)
    out: list[tuple[str, frozenset[float]]] = []
    for m in _NUM_RE.finditer(clean):
        raw, _pct, mult = m.group(1), m.group(2), m.group(3)
        try:
            base = float(raw.replace(",", ""))
        except ValueError:  # pragma: no cover - regex guarantees numeric
            continue
        vals = {_canon(base)}
        if mult:
            vals.add(_canon(base * _MULT[mult.lower()]))
        out.append((m.group(0).strip(), frozenset(vals)))
    for m in _WORD_NUM_RE.finditer(clean):
        out.append((m.group(0), frozenset({_canon(float(_WORD_NUM[m.group(1).lower()]))})))
    return out


def number_values(text: str) -> set[float]:
    vals: set[float] = set()
    for _, v in extract_numbers(text):
        vals |= v
    return vals


def ungrounded_numbers(text: str, evidence_texts: list[str]) -> list[str]:
    """Surfaces of numbers in `text` whose canonical values appear in none of `evidence_texts`."""
    pool: set[float] = set()
    for e in evidence_texts:
        pool |= number_values(e)
    return [surface for surface, vals in extract_numbers(text) if not (vals & pool)]


def is_substring_quote(quote: str, source: str) -> bool:
    q = normalise(quote).strip(" .\"'")
    return len(q) >= 3 and q in normalise(source)
