"""Stage 3: short summary built only from SUPPORTED claims and their evidence passages.

Each sentence passes gate 2 (citations exist in the allowed set, numbers grounded in cited
passages) or is dropped. Dropping is a graceful degradation: the summary is non-critical, the
claim table is the primary output. If nothing survives, the summary is empty and the report
says so.

Edge cases NOT handled: an article with zero supported claims yields no summary (by design:
there is nothing verified to summarise).
"""

from __future__ import annotations

from cgprr import obs, prompts
from cgprr.llm import LLM
from cgprr.schemas import ClaimResult, LLMSummary, Passage, VerifiedStatement
from cgprr.verify import check_statement, resolve_labels, to_verified


async def summarise(title: str, results: list[ClaimResult], index_by_id: dict[str, Passage],
                    llm: LLM, doc_vocab: dict[str, set[str]] | None = None) -> tuple[list[VerifiedStatement], int]:
    supported = [r for r in results if r.verdict == "supported"]
    if not supported:
        return [], 0
    pids = list(dict.fromkeys(e.passage_id for r in supported for e in r.evidence if e.valid))
    passages = [index_by_id[p] for p in pids if p in index_by_id]
    if not passages:
        return [], 0
    out = await llm.complete(LLMSummary, prompts.SUMMARY_SYSTEM,
                             prompts.summary_user(title, [r.text for r in supported], passages), name="summary")
    allowed = {p.pid: p for p in passages}
    labels = prompts.label_map(passages)
    kept: list[VerifiedStatement] = []
    dropped = 0
    for raw in out.sentences:
        s = resolve_labels(raw, labels)
        chk = check_statement(s, allowed, doc_vocab)
        if chk.ok:
            kept.append(to_verified(s))
        else:
            dropped += 1
            obs.log("summary.sentence_dropped", reasons=chk.reasons, **chk.detail)
    return kept, dropped
