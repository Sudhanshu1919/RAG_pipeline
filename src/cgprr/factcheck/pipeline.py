"""Track C orchestration: extract -> verify each claim (concurrent) -> summarise -> report.

Concurrency: claim checks run with asyncio.gather; the LLM client's semaphore bounds load on
the single GPU. A failure in one claim check is isolated: that claim becomes "unverified" with
flag `check_error`, the rest of the report proceeds (degrade gracefully). Extraction failure is
critical and propagates.
"""

from __future__ import annotations

import asyncio
import time

from cgprr import obs
from cgprr.factcheck.claims import extract_claims
from cgprr.factcheck.summarise import summarise
from cgprr.factcheck.verdict import check_claim
from cgprr.ingest import Article
from cgprr.llm import LLM, LLMError
from cgprr.retrieve import MultiRetriever, Retriever
from cgprr.schemas import ClaimResult, FactCheckReport, Passage, Verdict, VerifiedStatement


async def _safe_check(cid: str, text: str, quote: str, retriever: Retriever | MultiRetriever, llm: LLM, consistency: bool) -> ClaimResult:
    try:
        return await check_claim(cid, text, quote, retriever, llm, consistency=consistency)
    except LLMError as exc:
        obs.log("claim.error", claim_id=cid, error=type(exc).__name__)
        return ClaimResult(claim_id=cid, text=text, article_quote=quote, verdict="unverified", model_verdict=None,
                           evidence=[], rationale="Check failed; not verified.", confidence="low",
                           model_confidence=None, flags=["check_error"], retrieved=[])


async def factcheck_article(article: Article, retriever: Retriever | MultiRetriever, llm: LLM, *, consistency: bool = False) -> FactCheckReport:
    tid = obs.new_trace()
    t0 = time.perf_counter()
    with obs.span("factcheck.extract", article=article.article_id):
        ext = await extract_claims(article, llm)

    results = list(await asyncio.gather(*[
        _safe_check(f"{article.article_id}-X{i}", c.text, c.article_quote, retriever, llm, consistency)
        for i, c in enumerate(ext.claims, 1)
    ]))

    by_id = retriever.idx.by_id
    flags = list(ext.flags)
    try:
        summary, dropped = await summarise(article.title, results, by_id, llm, retriever.doc_vocab)
    except LLMError as exc:
        summary, dropped = [], 0
        flags.append("summary_unavailable")
        obs.log("summary.error", error=type(exc).__name__)

    counts: dict[Verdict, int] = {"supported": 0, "unsupported": 0, "unverified": 0}
    for r in results:
        counts[r.verdict] += 1
    report = FactCheckReport(
        trace_id=tid, article_id=article.article_id, article_title=article.title, model=llm.model,
        summary=summary, summary_dropped=dropped, claims=results, counts=counts,
        uncertainty_note=_uncertainty_note(results, summary, dropped, ext.flags), flags=flags,
        sources=_sources(results, summary, by_id), latency_ms=round((time.perf_counter() - t0) * 1000, 2),
    )
    obs.log("factcheck.result", article=article.article_id, counts=counts, n_claims=len(results),
            summary_sentences=len(summary), summary_dropped=dropped, latency_ms=report.latency_ms)
    return report


def _sources(results: list[ClaimResult], summary: list[VerifiedStatement], by_id: dict[str, Passage]) -> dict[str, str]:
    pids: list[str] = [e.passage_id for r in results for e in r.evidence if e.valid]
    pids += [c for s in summary for c in s.citations]
    out: dict[str, str] = {}
    for p in dict.fromkeys(pids):
        psg = by_id.get(p)
        if psg is not None:
            out[p] = f"{psg.publisher} — {psg.title} ({psg.date}): {psg.text}"
    return out


def _uncertainty_note(results: list[ClaimResult], summary: list[VerifiedStatement], dropped: int, ext_flags: list[str]) -> str:
    n = len(results)
    unv = sum(r.verdict == "unverified" for r in results)
    low = sum(r.confidence == "low" for r in results)
    unsup = sum(r.verdict == "unsupported" for r in results)
    parts = [f"{unv} of {n} claims could not be verified against the source pack."]
    if low:
        parts.append(f"{low} verdict(s) were downgraded by code checks (low confidence).")
    if unsup:
        parts.append(f"{unsup} claim(s) are contradicted by a source; the contradiction is a model judgement shown with its evidence.")
    if dropped:
        parts.append(f"{dropped} summary sentence(s) were removed because they failed citation/number checks.")
    if not summary:
        parts.append("No summary sentence could be verified.")
    if ext_flags:
        parts.append("Extraction notes: " + ", ".join(ext_flags) + ".")
    parts.append("Verdicts are automated and must be reviewed by an editor before publication.")
    return " ".join(parts)
