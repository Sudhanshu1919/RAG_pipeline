"""Core task: cited Q&A over the source pack with code-enforced refusal.

Flow: scan -> retrieve -> GATE 1 (unknown entity / low relevance; no model call) ->
      constrained generation -> GATE 2 (citations, numbers) -> answer | refusal.
Fails closed: any statement failing gate 2 refuses the whole answer.

No-LLM mode (`use_llm=False`, `cgprr ask --no-llm`): the same retrieval and gate 1, then the
best-matching source sentences are returned verbatim with their passage ids. Zero model calls, so it
runs with no GPU and no model server; nothing is paraphrased, so nothing can be hallucinated. It finds
where the answer is; it does not compose one (status "extracted").

Edge cases NOT handled: multi-turn context, questions needing aggregation across >top_k
passages, questions in languages other than English.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from cgprr import guard, obs, prompts
from cgprr.chunking import sentences
from cgprr.llm import _CHARS_PER_TOKEN, LLM, LLMError
from cgprr.retrieve import MultiRetriever, Retriever
from cgprr.schemas import AskResult, LLMAnswer, Passage
from cgprr.textnorm import content_terms
from cgprr.verify import check_statement, planted_numbers, resolve_labels, to_verified

REFUSAL = "Not in the sources."
_MAX_QUESTION_CHARS = 1_000


async def ask(question: str, retriever: Retriever | MultiRetriever, llm: LLM | None, *, log_text: bool = False,
              on_event: Callable[[str, dict[str, object]], Awaitable[None]] | None = None,
              use_llm: bool = True) -> AskResult:
    tid = obs.new_trace()
    t0 = time.perf_counter()
    question = question.strip()
    flags: list[str] = []
    timings: dict[str, float] = {}

    def refuse(reason: str, retrieved: list[str], **detail: object) -> AskResult:
        res = AskResult(trace_id=tid, status="refused", answer=REFUSAL, refusal_reason=reason,
                        retrieved=retrieved, flags=flags, latency_ms=_ms(t0), timings_ms=timings)
        obs.log("ask.result", status="refused", reason=reason, retrieved=retrieved, flags=flags,
                latency_ms=res.latency_ms, **detail)
        return res

    if not question:
        return refuse("empty_question", [])
    if len(question) > _MAX_QUESTION_CHARS:
        return refuse("question_too_long", [])

    obs.log("ask.start", q_fp=obs.text_fingerprint(question), q_len=len(question),
            **({"q_text": question} if log_text else {}))

    inj = guard.scan(question)
    if inj.suspected:
        flags.append("injection_suspected")
        obs.log("guard.injection", rules=list(inj.rules), where="question")

    unknown = retriever.unknown_entities(question)
    t_r = time.perf_counter()
    with obs.span("retrieve") as sp:
        hits, rflags = await retriever.search(question)
        sp["retrieved"] = [h.passage.pid for h in hits]
    timings["retrieve"] = _ms(t_r)
    if on_event:
        await on_event("retrieved", {"passages": [h.passage.pid for h in hits], "ms": timings["retrieve"]})
    flags.extend(rflags)
    ids = [h.passage.pid for h in hits]
    passages = [h.passage for h in hits]
    if planted_numbers(question, passages):
        flags.append("planted_number_in_question")

    # ---- GATE 1 (no model call) ------------------------------------------------------------
    if unknown:
        return refuse("gate1_unknown_entity", ids, entities=unknown)
    if not retriever.relevant(hits):
        return refuse("gate1_low_relevance", ids,
                      best_coverage=max((h.coverage for h in hits), default=0.0),
                      best_dense=max((h.dense or 0.0 for h in hits), default=0.0))

    if not use_llm or llm is None:
        picked = extract_sentences(question, passages)
        if not picked:
            return refuse("extractive_no_match", ids)
        by_pid = {p.pid: p for p in passages}
        cited = list(dict.fromkeys(pid for pid, _ in picked))
        res = AskResult(trace_id=tid, status="extracted", answer=" ".join(f"“{t}” [{pid}]" for pid, t in picked),
                        sources={c: by_pid[c].text for c in cited}, retrieved=ids, flags=flags + ["no_llm"],
                        latency_ms=_ms(t0), labels=list(dict.fromkeys(lb for c in cited for lb in by_pid[c].labels)),
                        timings_ms=timings)
        obs.log("ask.result", status="extracted", cited=cited, retrieved=ids, flags=res.flags, latency_ms=res.latency_ms)
        return res

    # ---- generation -------------------------------------------------------------------------
    # Only the passages shown to the model may be cited (gate 2 checks against `passages`).
    passages = budget_passages(passages, retriever.s.qa_passage_tokens)
    t_g = time.perf_counter()
    if on_event:
        await on_event("generating", {})
    try:
        out = await llm.complete(LLMAnswer, prompts.QA_SYSTEM, prompts.qa_user(question, passages), name="qa",
                                 max_tokens=retriever.s.qa_max_tokens)
    except LLMError as exc:
        # Critical path: surface loudly to the caller; do not fabricate a fallback answer.
        obs.log("ask.error", error=type(exc).__name__)
        raise

    timings["generate"] = _ms(t_g)
    if not out.answerable or not out.statements:
        return refuse("model_not_in_sources", ids)

    # ---- GATE 2 -----------------------------------------------------------------------------
    labels = prompts.label_map(passages)
    statements = [resolve_labels(s, labels) for s in out.statements]
    allowed = {p.pid: p for p in passages}
    for st in statements:
        chk = check_statement(st, allowed, retriever.doc_vocab)
        if not chk.ok:
            return refuse("gate2_" + chk.reasons[0], ids, reasons=chk.reasons, **chk.detail)

    verified = [to_verified(s) for s in statements]
    cited = list(dict.fromkeys(c for s in verified for c in s.citations))
    content_labels = list(dict.fromkeys(lb for c in cited for lb in allowed[c].labels))
    answer = " ".join(f"{s.text.rstrip()} [{', '.join(s.citations)}]" for s in verified)
    res = AskResult(trace_id=tid, status="answered", answer=answer, statements=verified,
                    sources={c: allowed[c].text for c in cited}, retrieved=ids, flags=flags,
                    latency_ms=_ms(t0), labels=content_labels, timings_ms=timings)
    obs.log("ask.result", status="answered", cited=cited, retrieved=ids, flags=flags, latency_ms=res.latency_ms)
    return res


def extract_sentences(question: str, passages: list[Passage], *, max_sentences: int = 2,
                      max_passages: int = 3) -> list[tuple[str, str]]:
    """(passage id, verbatim sentence) pairs that share the most content terms with the question.
    Ties go to the higher-ranked passage. Sentences sharing no content term are never returned."""
    q = content_terms(question)
    if not q:
        return []
    scored: list[tuple[float, int, int, str, str]] = []
    for rank, p in enumerate(passages[:max_passages]):
        for j, sent in enumerate(sentences(p.text)):
            overlap = len(q & content_terms(sent))
            if overlap and len(sent) >= 20:
                scored.append((overlap / len(q), -rank, -j, p.pid, sent))
    scored.sort(reverse=True)
    if not scored:
        return []
    floor = 0.6 * scored[0][0]  # a second sentence must be nearly as relevant as the best one
    return [(pid, sent) for score, _, _, pid, sent in scored[:max_sentences] if score >= floor]


def budget_passages(passages: list[Passage], max_tokens: int) -> list[Passage]:
    """Keep passages in rank order while their estimated size fits `max_tokens`; the top passage is always kept."""
    if max_tokens <= 0:
        return passages
    out: list[Passage] = []
    used = 0
    for p in passages:
        cost = int(len(p.text) / _CHARS_PER_TOKEN) + 12  # + label / heading overhead
        if out and used + cost > max_tokens:
            break
        out.append(p)
        used += cost
    return out


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 2)
