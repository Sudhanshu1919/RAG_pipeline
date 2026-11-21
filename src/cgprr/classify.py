"""Optional build-time content classifier: Llama Guard 3 (open weights, local via Ollama).

Enabled with CGPRR_GUARD_MODEL (e.g. `llama-guard3:1b`, ~1 GB). It runs once per chunk at build
time only, so it adds nothing to answer latency. Its verdicts can only make outcomes stricter or
add labels; the deterministic rules in `security.py` stay authoritative:

  S4  child sexual exploitation  -> the whole file is dropped (nothing stored), reported
  S9  indiscriminate weapons     -> label; quarantine only if the rules also see instruction form
  other S-codes                  -> content_warning label (facts are kept)

Llama Guard is trained on chat turns; a document chunk is presented as a user message. On news
text it over-flags reporting of crime and violence, which is why its output is mapped to labels,
not deletions.

Edge cases NOT handled: chunks longer than the guard model's context (truncated to 4,000 chars).
"""

from __future__ import annotations

import asyncio
import re

from cgprr import obs
from cgprr.llm import LLMError, LocalLLM
from cgprr.schemas import Document
from cgprr.security import chunk_hash, operational_harm

S_CODES = {
    "S1": "violence", "S2": "crime", "S3": "sexual_violence", "S4": "child_sexual_exploitation",
    "S5": "defamation", "S6": "specialized_advice", "S7": "privacy", "S8": "intellectual_property",
    "S9": "indiscriminate_weapons", "S10": "hate", "S11": "self_harm", "S12": "sexual_content",
    "S13": "elections", "S14": "code_abuse",
}


def parse_guard(text: str) -> list[str]:
    t = text.strip().lower()
    if not t.startswith("unsafe"):
        return []
    return [S_CODES[c] for c in dict.fromkeys(re.findall(r"\bS\d{1,2}\b", text.upper())) if c in S_CODES]


async def classify_docs(docs: list[Document], llm: LocalLLM, model: str) -> dict[str, list[str]]:
    """chunk hash -> categories (empty list = safe). Errors leave a chunk unclassified."""
    out: dict[str, list[str]] = {}
    errors = 0

    async def one(text: str) -> None:
        nonlocal errors
        h = chunk_hash(text)
        if h in out:
            return
        try:
            reply = await llm.chat_text(model, [{"role": "user", "content": text[:4000]}], name="guard", max_tokens=16)
            out[h] = parse_guard(reply)
        except LLMError:
            errors += 1

    await asyncio.gather(*[one(t) for d in docs for i, t in enumerate(d.paragraphs, 1) if i not in d.flagged_paragraphs])
    obs.log("guard.done", model=model, n=len(out), unsafe=sum(1 for v in out.values() if v), errors=errors)
    return out


def apply_verdicts(docs: list[Document], verdicts: dict[str, list[str]], report: object) -> list[Document]:
    """Apply classifier verdicts to already-screened documents (stricter only)."""
    kept: list[Document] = []
    for d in docs:
        drop = False
        for i, text in enumerate(d.paragraphs, 1):
            cats = verdicts.get(chunk_hash(text), [])
            if not cats:
                continue
            if "child_sexual_exploitation" in cats:
                drop = True
                report.add(file=d.source, action="rejected", rule="classifier:child_sexual_exploitation", doc=d.doc_id, chunk=i)  # type: ignore[attr-defined]
                break
            if "indiscriminate_weapons" in cats and operational_harm(text) and i not in d.flagged_paragraphs:
                d.flagged_paragraphs.append(i)
                report.add(file=d.source, action="quarantined", rule="classifier:weapons_operational", doc=d.doc_id, chunk=i)  # type: ignore[attr-defined]
            labels = d.labels.setdefault(str(i), [])
            for c in cats:
                lb = f"content_warning:{c}"
                if lb not in labels:
                    labels.append(lb)
                    report.add(file=d.source, action="labelled", rule=f"classifier:{c}", doc=d.doc_id, chunk=i)  # type: ignore[attr-defined]
        if not drop:
            kept.append(d)
    return kept
