"""Typed contracts. LLM outputs are constrained to these JSON Schemas at decode time
(Ollama `format=<schema>` / vLLM guided JSON) and re-validated with Pydantic on receipt.
Raw-text parsing of model output is not used anywhere."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PassageId = str  # "<doc_id>:p<n>", e.g. "S03:p1" or "kotp-press-release-3fa21c:p2"
DOC_ID_RX = r"[A-Za-z0-9][A-Za-z0-9_.-]{0,80}"
# When several collections are queried together, passage ids are namespaced: "<collection>/<doc>:p<n>".
NS_RX = r"(?:[a-z0-9][a-z0-9_-]{0,63}/)?"
Verdict = Literal["supported", "unsupported", "unverified"]
Confidence = Literal["high", "medium", "low"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False)


# --------------------------------------------------------------------------- corpus


class Document(_Strict):
    doc_id: str = Field(pattern=rf"^{DOC_ID_RX}$")
    title: str
    publisher: str
    date: str
    doc_type: str
    paragraphs: list[str]
    flagged_paragraphs: list[int] = Field(default_factory=list)
    headings: list[str] = Field(default_factory=list)       # per chunk, "A > B"
    pages: list[int | None] = Field(default_factory=list)  # per chunk, first page (PDF)
    source: str = ""                                       # file name inside the collection
    labels: dict[str, list[str]] = Field(default_factory=dict)  # chunk no -> content labels


class Passage(_Strict):
    pid: PassageId = Field(pattern=rf"^{NS_RX}{DOC_ID_RX}:p\d+$")
    doc_id: str
    para_no: int = Field(ge=1)
    title: str
    publisher: str
    date: str
    doc_type: str
    text: str
    heading: str = ""
    page: int | None = None
    source: str = ""
    labels: list[str] = Field(default_factory=list)  # e.g. content_warning:violence (shown with any citation)


class Hit(_Strict):
    passage: Passage
    rank: int
    rrf: float
    bm25: float
    dense: float | None
    coverage: float = Field(ge=0.0, le=1.0)
    relevant: bool = False      # passed gate-1 thresholds of the collection it came from
    collection: str = ""


# --------------------------------------------------------------------------- LLM output schemas
# Kept small: fewer fields -> fewer tokens and more reliable constrained decoding on a 7B model.


class LLMStatement(_Strict):
    text: str = Field(min_length=1, max_length=500)
    citations: list[PassageId] = Field(default_factory=list, max_length=6)


class LLMAnswer(_Strict):
    answerable: bool
    statements: list[LLMStatement] = Field(default_factory=list, max_length=6)


class LLMClaim(_Strict):
    text: str = Field(min_length=5, max_length=400)
    article_quote: str = Field(min_length=5, max_length=400)


class LLMClaims(_Strict):
    claims: list[LLMClaim] = Field(default_factory=list, max_length=12)


class LLMEvidence(_Strict):
    passage_id: PassageId
    quote: str = Field(min_length=3, max_length=400)


class LLMVerdict(_Strict):
    verdict: Verdict
    evidence: list[LLMEvidence] = Field(default_factory=list, max_length=4)
    rationale: str = Field(min_length=1, max_length=600)
    confidence: Confidence


class LLMSummary(_Strict):
    sentences: list[LLMStatement] = Field(default_factory=list, max_length=6)


# --------------------------------------------------------------------------- pipeline results


class GateResult(_Strict):
    passed: bool
    reason: str | None = None
    detail: dict[str, object] = Field(default_factory=dict)


class VerifiedStatement(_Strict):
    text: str
    citations: list[PassageId]


class AskResult(_Strict):
    trace_id: str
    status: Literal["answered", "refused", "extracted"]  # extracted = --no-llm: verbatim source sentences
    answer: str
    statements: list[VerifiedStatement] = Field(default_factory=list)
    sources: dict[PassageId, str] = Field(default_factory=dict)
    refusal_reason: str | None = None
    retrieved: list[PassageId] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    labels: list[str] = Field(default_factory=list)  # content warnings of the cited passages
    timings_ms: dict[str, float] = Field(default_factory=dict)


class EvidenceSpan(_Strict):
    passage_id: PassageId
    quote: str
    valid: bool


class ClaimResult(_Strict):
    claim_id: str
    text: str
    article_quote: str
    verdict: Verdict
    model_verdict: Verdict | None
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    rationale: str
    confidence: Confidence
    model_confidence: Confidence | None
    flags: list[str] = Field(default_factory=list)
    retrieved: list[PassageId] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)


class FactCheckReport(_Strict):
    trace_id: str
    article_id: str
    article_title: str
    model: str
    summary: list[VerifiedStatement]
    summary_dropped: int
    claims: list[ClaimResult]
    counts: dict[Verdict, int]
    uncertainty_note: str
    flags: list[str] = Field(default_factory=list)
    sources: dict[PassageId, str] = Field(default_factory=dict)
    latency_ms: float = 0.0
