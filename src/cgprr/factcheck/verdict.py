"""Stage 2: per-claim verdict with code-enforced downgrades.

Final verdict = model verdict, then corrected by code:
- gate 1: no passage above relevance thresholds -> "unverified" (model not called)
- evidence quote not verbatim / cites passage not shown -> evidence marked invalid
- supported|unsupported with no valid evidence -> "unverified" (no_valid_evidence)
- supported but a claim number is absent from the cited passages -> "unverified"
- supported but a claim proper noun is absent from the cited source documents -> "unverified"
- optional consistency pass (passages reversed); disagreement -> "unverified"

Confidence is computed by code from these signals, not taken from the model:
- high:   supported, evidence valid, every claim number found in the evidence quotes and
          >= 60% of the claim's content terms present in the quotes;
          or unsupported where evidence quotes carry numbers and a claim number is absent
- medium: other verdicts that survived the checks unchanged
- low:    any verdict produced by a code downgrade or consistency disagreement

Edge cases NOT handled: claims whose evidence spans >top_k passages; correct numbers with a wrong
direction/relation pass the number check (model judgement only); temporal reasoning
("last week") relative to article date.
"""

from __future__ import annotations

from cgprr import obs, prompts
from cgprr.llm import LLM
from cgprr.retrieve import MultiRetriever, Retriever
from cgprr.schemas import ClaimResult, Confidence, EvidenceSpan, LLMVerdict, Passage, Verdict
from cgprr.textnorm import content_terms, ungrounded_numbers
from cgprr.verify import quote_valid, resolve_label, ungrounded_entities


async def _judge(claim: str, passages: list[Passage], llm: LLM) -> LLMVerdict:
    return await llm.complete(LLMVerdict, prompts.VERDICT_SYSTEM, prompts.verdict_user(claim, passages), name="verdict")


async def check_claim(claim_id: str, claim: str, article_quote: str, retriever: Retriever | MultiRetriever, llm: LLM,
                      *, consistency: bool = False) -> ClaimResult:
    hits, rflags = await retriever.search(claim)
    flags = list(rflags)
    ids = [h.passage.pid for h in hits]

    if not retriever.relevant(hits):
        flags.append("no_passage_above_threshold")
        res = ClaimResult(claim_id=claim_id, text=claim, article_quote=article_quote, verdict="unverified",
                          model_verdict=None, evidence=[], rationale="No source passage addresses this claim.",
                          confidence="medium", model_confidence=None, flags=flags, retrieved=ids)
        obs.log("claim.verdict", claim_id=claim_id, verdict="unverified", gate="gate1", retrieved=ids)
        return res

    passages = [h.passage for h in hits]
    by_id = {p.pid: p for p in passages}
    out = await _judge(claim, passages, llm)
    label_map = prompts.label_map(passages)

    evidence = []
    for e in out.evidence:
        pid = resolve_label(e.passage_id, label_map)
        evidence.append(EvidenceSpan(passage_id=pid, quote=e.quote, valid=quote_valid(e.quote, by_id.get(pid))))
    if any(not e.valid for e in evidence):
        flags.append("invalid_evidence_quote")
    valid_ev = [e for e in evidence if e.valid]
    verdict: Verdict = out.verdict
    downgraded = False

    if verdict in ("supported", "unsupported") and not valid_ev:
        verdict, downgraded = "unverified", True
        flags.append("no_valid_evidence")

    if verdict == "supported":
        cited_texts = [by_id[e.passage_id].text for e in valid_ev]
        missing = ungrounded_numbers(claim, cited_texts)
        if missing:
            verdict, downgraded = "unverified", True
            flags.append("number_not_in_evidence:" + ",".join(missing))
        else:
            ents = ungrounded_entities(claim, [by_id[e.passage_id].doc_id for e in valid_ev], retriever.doc_vocab)
            if ents:
                verdict, downgraded = "unverified", True
                flags.append("entity_not_in_evidence:" + ",".join(ents))

    if consistency and not downgraded:
        second = await _judge(claim, list(reversed(passages)), llm)
        if second.verdict != out.verdict:
            verdict, downgraded = "unverified", True
            flags.append(f"inconsistent_verdict:{out.verdict}/{second.verdict}")

    confidence = _confidence(verdict, downgraded, claim, valid_ev)
    obs.log("claim.verdict", claim_id=claim_id, verdict=verdict, model_verdict=out.verdict,
            downgraded=downgraded, retrieved=ids, flags=flags)
    labels = list(dict.fromkeys(lb for e in valid_ev for lb in by_id[e.passage_id].labels))
    return ClaimResult(claim_id=claim_id, text=claim, article_quote=article_quote, verdict=verdict,
                       model_verdict=out.verdict, evidence=evidence, rationale=out.rationale,
                       confidence=confidence, model_confidence=out.confidence, flags=flags, retrieved=ids,
                       labels=labels)


def _confidence(verdict: Verdict, downgraded: bool, claim: str, valid_ev: list[EvidenceSpan]) -> Confidence:
    if downgraded:
        return "low"
    quotes = [e.quote for e in valid_ev]
    if verdict == "supported":
        terms = content_terms(claim)
        quote_terms = content_terms(" ".join(quotes))
        overlap = len(terms & quote_terms) / len(terms) if terms else 0.0
        return "high" if quotes and not ungrounded_numbers(claim, quotes) and overlap >= 0.6 else "medium"
    if verdict == "unsupported":
        ev_has_numbers = any(ungrounded_numbers(q, []) for q in quotes)
        claim_number_absent = bool(ungrounded_numbers(claim, quotes))
        return "high" if ev_has_numbers and claim_number_absent else "medium"
    return "medium"
