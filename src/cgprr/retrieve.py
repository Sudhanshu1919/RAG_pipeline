"""Hybrid retrieval (BM25 + dense, reciprocal-rank fusion), over one or several collections.

Per collection (`Retriever`):
- BM25 over retrieval tokens (raw text + heading + read-pass context/questions/keywords).
- Dense: cosine to the chunk vector and to each of its read-pass question vectors; a chunk's dense
  score is the max of those (a question phrased like the user's question pulls its chunk up).
- RRF fuses the two rank lists (k=60). A per-document cap keeps one long document from filling
  every slot (diversity), applied after fusion.
- Gate-1 signals on every hit: lexical coverage of the query terms and the dense score, compared
  with this collection's thresholds (calibrated at build time, see calibrate.py; env overrides).

Several collections (`MultiRetriever`): each is searched with its own index and thresholds, the
per-collection rank lists are fused again by RRF, and passage ids are namespaced
"<collection>/<doc>:p<n>" so ids from different collections can never collide. Collections not
selected are never loaded.

The unknown-entity check uses RAW text vocabulary only (never read-pass text).

Edge cases NOT handled: metadata filters by date/publisher, multilingual queries, queries longer
than the embedding model's context (truncated by the server).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from cgprr import obs
from cgprr.config import Settings
from cgprr.index import Index
from cgprr.llm import LLM, LLMError
from cgprr.schemas import Hit, Passage
from cgprr.textnorm import entity_candidates, tokenize

_RRF_K = 60


class _BM25(BM25Okapi):
    """BM25 with the Lucene/Robertson-Sparck-Jones idf, log(1 + (N - n + 0.5) / (n + 0.5)), which is always
    positive. rank_bm25's Okapi idf is exactly 0 for a term in half of a tiny collection (e.g. 1 of 2
    chunks), which would make every score 0 and fall back to file order."""

    def _calc_idf(self, nd: dict[str, int]) -> None:  # type: ignore[override]
        n = self.corpus_size
        for word, freq in nd.items():
            self.idf[word] = math.log(1.0 + (n - freq + 0.5) / (freq + 0.5))
        self.average_idf = sum(self.idf.values()) / len(self.idf) if self.idf else 0.0


def _rrf_ranks(scores: np.ndarray) -> np.ndarray:
    n = len(scores)
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(1, n + 1)
    return 1.0 / (_RRF_K + ranks)


def _cap_per_doc(order: list[int], doc_of: list[str], k: int, cap: int) -> list[int]:
    if cap <= 0:
        return order[:k]
    picked: list[int] = []
    per: dict[str, int] = {}
    overflow: list[int] = []
    for i in order:
        d = doc_of[i]
        if per.get(d, 0) < cap:
            picked.append(i)
            per[d] = per.get(d, 0) + 1
        else:
            overflow.append(i)
        if len(picked) == k:
            return picked
    return (picked + overflow)[:k]  # fewer documents than slots: fill with the rest in rank order


class Retriever:
    def __init__(self, index: Index, settings: Settings, llm: LLM | None, *, name: str | None = None) -> None:
        self.idx = index
        self.s = settings
        self.llm = llm
        self.name = name or settings.collection
        rt = index.retrieval_tokens
        self._bm25 = _BM25(rt) if any(rt) else None
        self._token_sets = [set(t) for t in rt]
        raw_sets = [set(t) for t in index.tokens]
        self._raw_sets = raw_sets
        self._bm25_raw: _BM25 | None = None  # built lazily, for the with/without read-pass ablation
        self.doc_vocab: dict[str, set[str]] = index.doc_vocab()
        self.vocab: set[str] = set().union(*self.doc_vocab.values()) if self.doc_vocab else set()
        self._doc_of = [p.doc_id for p in index.passages]
        t = index.thresholds()
        self.min_coverage = settings.min_coverage if "CGPRR_MIN_COVERAGE" in os.environ or "min_coverage" not in t else t["min_coverage"]
        self.min_dense = settings.min_dense if "CGPRR_MIN_DENSE" in os.environ or "min_dense" not in t else t["min_dense"]
        self.max_per_doc = int(os.environ.get("CGPRR_MAX_PER_DOC", "3"))

    def unknown_entities(self, text: str) -> list[str]:
        return sorted(e for e in entity_candidates(text) if e not in self.vocab)

    async def _dense(self, query: str, use_questions: bool = True) -> tuple[np.ndarray | None, list[str]]:
        if self.idx.embeddings is None or self.llm is None:
            return None, []
        try:
            qv = np.asarray((await self.llm.embed([query], query=True))[0], dtype=np.float32)
        except LLMError:
            obs.log("retrieve.dense_degraded", collection=self.name)
            return None, ["dense_unavailable"]  # degrade to BM25-only for this query
        qv /= np.linalg.norm(qv) + 1e-12
        dense = self.idx.embeddings @ qv
        if use_questions and self.idx.q_embeddings is not None and self.idx.q_owner is not None and len(self.idx.q_owner):
            qs = self.idx.q_embeddings @ qv
            best = np.full(len(dense), -1.0, dtype=np.float32)
            np.maximum.at(best, self.idx.q_owner, qs.astype(np.float32))
            dense = np.maximum(dense, best)
        return dense, []

    async def search(self, query: str, k: int | None = None, *, use_questions: bool = True) -> tuple[list[Hit], list[str]]:
        k = k or self.s.top_k
        q_tokens = tokenize(query)
        n = len(self.idx.passages)
        bm, sets = self._bm25, self._token_sets
        if not use_questions and self.idx.aug_tokens is not None:
            if self._bm25_raw is None:
                self._bm25_raw = _BM25(self.idx.tokens)
            bm, sets = self._bm25_raw, self._raw_sets
        bm25 = np.asarray(bm.get_scores(q_tokens), dtype=np.float64) if (q_tokens and bm) else np.zeros(n)
        dense, flags = await self._dense(query, use_questions)
        rrf = _rrf_ranks(bm25) + (_rrf_ranks(dense) if dense is not None else 0.0)
        order = [int(i) for i in np.argsort(-rrf, kind="stable")]
        top = _cap_per_doc(order, self._doc_of, k, self.max_per_doc)
        q_terms = set(q_tokens)
        hits: list[Hit] = []
        for rank, i in enumerate(top, 1):
            cov = (len(q_terms & sets[i]) / len(q_terms)) if q_terms else 0.0
            dv = float(dense[i]) if dense is not None else None
            rel = cov >= self.min_coverage or (dv is not None and dv >= self.min_dense)
            hits.append(Hit(passage=self.idx.passages[i], rank=rank, rrf=float(rrf[i]), bm25=float(bm25[i]),
                            dense=dv, coverage=round(cov, 4), relevant=rel, collection=self.name))
        return hits, flags

    def relevant(self, hits: list[Hit]) -> bool:
        """Gate-1 relevance: any hit clears its collection's lexical OR dense threshold."""
        return any(h.relevant for h in hits)


@dataclass
class _MergedIndex:
    """Read-only view with the attributes callers use on `Retriever.idx`."""
    passages: list[Passage]
    embeddings: object | None
    manifest: dict[str, object]

    @property
    def by_id(self) -> dict[str, Passage]:
        return {p.pid: p for p in self.passages}


def _ns(p: Passage, coll: str) -> Passage:
    return p.model_copy(update={"pid": f"{coll}/{p.pid}", "doc_id": f"{coll}/{p.doc_id}"})


class MultiRetriever:
    """Same interface as Retriever, over several collections. Ids are namespaced '<coll>/<doc>:p<n>'."""

    def __init__(self, retrievers: list[Retriever], settings: Settings) -> None:
        if len(retrievers) < 2:
            raise ValueError("MultiRetriever needs at least two collections; use Retriever for one")
        self.parts = retrievers
        self.s = settings
        self.name = ",".join(r.name for r in retrievers)
        self.vocab: set[str] = set().union(*(r.vocab for r in retrievers))
        self.doc_vocab: dict[str, set[str]] = {f"{r.name}/{d}": v for r in retrievers for d, v in r.doc_vocab.items()}
        passages = [_ns(p, r.name) for r in retrievers for p in r.idx.passages]
        dense_any = any(r.idx.embeddings is not None for r in retrievers)
        self.idx = _MergedIndex(passages, True if dense_any else None,
                                {"collections": [r.name for r in retrievers]})

    def unknown_entities(self, text: str) -> list[str]:
        return sorted(e for e in entity_candidates(text) if e not in self.vocab)

    async def search(self, query: str, k: int | None = None, *, use_questions: bool = True) -> tuple[list[Hit], list[str]]:
        k = k or self.s.top_k
        flags: list[str] = []
        pooled: dict[str, tuple[float, Hit]] = {}
        for r in self.parts:
            hits, fl = await r.search(query, k=max(k, 8), use_questions=use_questions)
            flags += fl
            for h in hits:
                key = f"{r.name}/{h.passage.pid}"
                score = 1.0 / (_RRF_K + h.rank)
                nh = h.model_copy(update={"passage": _ns(h.passage, r.name), "rrf": score})
                pooled[key] = (score, nh)
        ranked = sorted(pooled.values(), key=lambda t: (-t[0], -t[1].bm25))
        order = list(range(len(ranked)))
        doc_of = [h.passage.doc_id for _, h in ranked]
        top = _cap_per_doc(order, doc_of, k, int(os.environ.get("CGPRR_MAX_PER_DOC", "3")))
        out = [ranked[i][1].model_copy(update={"rank": n}) for n, i in enumerate(top, 1)]
        return out, list(dict.fromkeys(flags))

    def relevant(self, hits: list[Hit]) -> bool:
        return any(h.relevant for h in hits)

