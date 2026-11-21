"""Gate 2: deterministic post-generation checks. The model cannot override these.

A statement passes only if:
1. it cites at least one passage id,
2. every cited id was actually shown to the model for this request,
3. every number in the statement appears in at least one cited passage,
4. every proper-noun candidate (capitalised, non-sentence-initial) appears in a cited document,
5. it contains no secret / personal-identifier pattern (security.leaks),
6. every URL it contains appears in a cited passage (no invented or injected links).

Quote checks: evidence quotes must be verbatim substrings (after whitespace/quote/dash/case
normalisation) of the passage they claim to come from.

Edge cases NOT handled:
- Correct numbers in a false relation ("rose" vs "fell", wrong entity) pass the number check.
  This is the main residual hallucination risk; it is measured in eval (A04-C3).
- Entity check is a capitalisation heuristic, not NER: sentence-initial names and lowercase
  invented details are not caught; a correct entity in the wrong role passes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from cgprr.schemas import LLMStatement, Passage, VerifiedStatement
from cgprr.security import leaks
from cgprr.textnorm import entity_candidates, is_substring_quote, ungrounded_numbers


@dataclass
class StatementCheck:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    detail: dict[str, list[str]] = field(default_factory=dict)


def ungrounded_entities(text: str, doc_ids: list[str], doc_vocab: dict[str, set[str]]) -> list[str]:
    vocab: set[str] = set()
    for d in doc_ids:
        vocab |= doc_vocab.get(d, set())
    return sorted(e for e in entity_candidates(text) if e not in vocab)


def check_statement(stmt: LLMStatement, allowed: dict[str, Passage],
                    doc_vocab: dict[str, set[str]] | None = None) -> StatementCheck:
    reasons: list[str] = []
    detail: dict[str, list[str]] = {}
    cites = list(dict.fromkeys(stmt.citations))  # dedupe, keep order
    if not cites:
        reasons.append("no_citation")
    unknown = [c for c in cites if c not in allowed]
    if unknown:
        reasons.append("citation_not_in_context")
        detail["unknown_citations"] = unknown
    cited_texts = [allowed[c].text for c in cites if c in allowed]
    missing = ungrounded_numbers(stmt.text, cited_texts) if cited_texts else []
    if missing:
        reasons.append("number_not_in_cited_passage")
        detail["ungrounded_numbers"] = missing
    if doc_vocab is not None and cited_texts:
        ents = ungrounded_entities(stmt.text, [allowed[c].doc_id for c in cites if c in allowed], doc_vocab)
        if ents:
            reasons.append("entity_not_in_cited_source")
            detail["ungrounded_entities"] = ents
    # A kind already present in the cited sources was deliberately kept by the collection's redaction
    # policy (e.g. a press-office email); only identifiers the sources do not contain count as a leak.
    src_kinds = set(leaks(" ".join(cited_texts))) if cited_texts else set()
    leaked = [k for k in leaks(stmt.text) if k not in src_kinds]
    if leaked:
        reasons.append("secret_or_pii_in_output")
        detail["leak_kinds"] = leaked
    urls = [u for u in _URL_RE.findall(stmt.text) if not any(u.rstrip("/.,").lower() in t.lower() for t in cited_texts)]
    if urls:
        reasons.append("url_not_in_cited_source")
        detail["urls"] = urls[:3]
    return StatementCheck(ok=not reasons, reasons=reasons, detail=detail)


_URL_RE = re.compile(r"\b(?:https?://|www\.)[^\s<>\"\]\)]+", re.IGNORECASE)
_LABEL_RE = re.compile(r"\[?\b(P\d{1,2})\b\]?")


def resolve_label(ref: str, labels: dict[str, str]) -> str:
    """Prompt label ("P3") -> real passage id. Anything else is returned unchanged, so a real id
    still works and an unknown label still fails gate 2 as citation_not_in_context."""
    return labels.get(ref.strip().strip("[]"), ref)


def resolve_labels(stmt: LLMStatement, labels: dict[str, str]) -> LLMStatement:
    """Map citations to real ids and drop inline "[P3]" markers from the statement text."""
    text = _LABEL_RE.sub(lambda m: "" if m.group(1) in labels else m.group(0), stmt.text)
    text = re.sub(r"\s+([.,;:])", r"\1", re.sub(r"\s{2,}", " ", text)).strip() or stmt.text
    return LLMStatement(text=text, citations=[resolve_label(c, labels) for c in stmt.citations])


def to_verified(stmt: LLMStatement) -> VerifiedStatement:
    return VerifiedStatement(text=stmt.text.strip(), citations=list(dict.fromkeys(stmt.citations)))


def quote_valid(quote: str, passage: Passage | None) -> bool:
    return passage is not None and is_substring_quote(quote, passage.text)


def planted_numbers(query: str, retrieved: list[Passage]) -> list[str]:
    """Numbers the user supplied that no retrieved passage contains (e.g. 'the cut is 60%')."""
    return ungrounded_numbers(query, [p.text for p in retrieved])
