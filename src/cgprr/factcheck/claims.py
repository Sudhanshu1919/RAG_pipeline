"""Stage 1: claim extraction with a verbatim-quote hallucination boundary.

Every extracted claim must point at an exact span of the article. Claims whose
`article_quote` is not a substring are dropped (the model invented or paraphrased them).

Edge cases NOT handled: claims that need two separate spans; articles longer than the
context window (no chunking; shipped articles are < 1k tokens).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cgprr import obs, prompts
from cgprr.ingest import Article
from cgprr.llm import LLM
from cgprr.schemas import LLMClaim, LLMClaims
from cgprr.textnorm import is_substring_quote, normalise

MIN_CLAIMS = 5
MAX_CLAIMS = 8


@dataclass
class Extraction:
    claims: list[LLMClaim]
    dropped_invalid_quote: int = 0
    flags: list[str] = field(default_factory=list)


async def extract_claims(article: Article, llm: LLM) -> Extraction:
    body = article.body
    kept: list[LLMClaim] = []
    seen: set[str] = set()
    dropped = 0
    feedback: str | None = None
    for attempt in range(2):
        out = await llm.complete(LLMClaims, prompts.CLAIMS_SYSTEM,
                                 prompts.claims_user(article.title, body, feedback), name="extract_claims")
        for c in out.claims:
            key = normalise(c.article_quote)
            if key in seen:
                continue
            if not is_substring_quote(c.article_quote, body):
                dropped += 1
                continue
            seen.add(key)
            kept.append(c)
        if len(kept) >= MIN_CLAIMS:
            break
        feedback = (f"Only {len(kept)} claims had verbatim article_quote spans. Return {MIN_CLAIMS}-{MAX_CLAIMS} "
                    "claims and copy article_quote exactly from the article.")
        obs.log("claims.retry", attempt=attempt, kept=len(kept), dropped=dropped)

    flags: list[str] = []
    if dropped:
        flags.append(f"claims_dropped_invalid_quote:{dropped}")
    if len(kept) < MIN_CLAIMS:
        flags.append(f"claims_shortfall:{len(kept)}")
    obs.log("claims.extracted", article=article.article_id, kept=min(len(kept), MAX_CLAIMS), dropped=dropped)
    return Extraction(claims=kept[:MAX_CLAIMS], dropped_invalid_quote=dropped, flags=flags)
