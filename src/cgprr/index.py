"""Build and load a collection's on-disk index.

Files in indexes/<collection>/:
  passages.jsonl     passage + raw tokens + augmented tokens (raw + heading + read-pass context/questions)
  reading.jsonl      read-pass output per chunk (+ per-document summaries)
  emb_text.npy       one L2-normalised vector per chunk ("title. heading. context. text")
  emb_q.npy, q_owner.npy   one vector per read-pass question + the index of the chunk it belongs to
  manifest.json      counts, models, read-pass status, calibrated gate thresholds
  ingest_report.jsonl  every security/ingest decision (ids and rules only, never text)
  quarantine.jsonl     quarantined chunks kept for human review (never searched, never shown to the model)

Two-step build (latency: a collection is searchable within minutes, enriched later):
  step A  parse -> screen -> chunk -> BM25 -> embed                 (no generator calls)
  step B  read pass -> augmented BM25 tokens + question vectors     (one LLM call per chunk, cached)
Each step writes a complete index and swaps it in atomically.

What the read pass may influence: ranking (BM25 tokens, dense vectors) and gate-1 coverage.
What it may never influence: `doc_vocab`/`vocab` (unknown-entity check), gate 2, evidence quotes —
those use raw chunk text only, so an entity or figure invented by the reader cannot ground an answer.

Vector search is exact (one matrix product): for <= ~50k vectors that is faster and simpler than ANN.

Edge cases NOT handled: incremental updates (full rebuild; the LLM cache makes unchanged chunks free),
concurrent writers to the same index dir, embedding-model change without rebuild (detected, rejected).
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from cgprr import ingest, obs, security
from cgprr.config import Settings
from cgprr.llm import LLM, LLMError
from cgprr.schemas import Document, Passage
from cgprr.textnorm import tokenize

INDEX_VERSION = 3  # v3: read-pass augmentation, question vectors, calibrated thresholds


@dataclass
class Index:
    passages: list[Passage]
    tokens: list[list[str]]          # raw text tokens (vocabulary / gates)
    embeddings: NDArray[np.float32] | None
    manifest: dict[str, object]
    aug_tokens: list[list[str]] | None = None   # retrieval tokens incl. read-pass keys
    q_embeddings: NDArray[np.float32] | None = None
    q_owner: NDArray[np.int64] | None = None
    readings: dict[str, dict[str, object]] = field(default_factory=dict)

    @property
    def by_id(self) -> dict[str, Passage]:
        return {p.pid: p for p in self.passages}

    @property
    def retrieval_tokens(self) -> list[list[str]]:
        return self.aug_tokens if self.aug_tokens is not None else self.tokens

    def doc_vocab(self) -> dict[str, set[str]]:
        """doc_id -> stemmed tokens of the whole document (title + headings + all indexed paragraphs).
        RAW document text only: headings come from the file itself (e.g. a PDF table's row label), never
        from the read pass, so a term that appears only in a heading still counts as known."""
        out: dict[str, set[str]] = {}
        for p, toks in zip(self.passages, self.tokens, strict=True):
            s = out.setdefault(p.doc_id, set())
            s.update(toks)
            if p.heading:
                s.update(tokenize(p.heading))
        return out

    def thresholds(self) -> dict[str, float]:
        t = self.manifest.get("thresholds")
        if not isinstance(t, dict):
            return {}
        # only the numeric gate thresholds; "source", "tpr", counts etc. are reporting fields
        return {k: float(t[k]) for k in ("min_coverage", "min_dense") if isinstance(t.get(k), (int, float))}


def _raw_tokens(p: Passage) -> list[str]:
    return tokenize(f"{p.title} {p.text}")


def _aug_tokens(p: Passage, r: dict[str, object] | None) -> list[str]:
    extra = [p.heading]
    if r:
        extra += [str(r.get("context", ""))] + [str(q) for q in r.get("questions", []) or []] + [str(k) for k in r.get("keywords", []) or []]  # type: ignore[union-attr]
    return _raw_tokens(p) + tokenize(" ".join(x for x in extra if x))


def _embed_text(p: Passage, r: dict[str, object] | None) -> str:
    ctx = str(r.get("context", "")) if r else ""
    return ". ".join(x for x in (p.title, p.heading, ctx, p.text) if x)


async def _embed_all(settings: Settings, llm: LLM, passages: list[Passage], readings: dict[str, dict[str, object]]
                     ) -> tuple[NDArray[np.float32] | None, NDArray[np.float32] | None, NDArray[np.int64] | None, str | None]:
    try:
        vecs = await llm.embed([_embed_text(p, readings.get(p.pid)) for p in passages])
        emb = np.asarray(vecs, dtype=np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        qs, owner = [], []
        for i, p in enumerate(passages):
            for q in (readings.get(p.pid) or {}).get("questions", []) or []:  # type: ignore[union-attr]
                qs.append(str(q))
                owner.append(i)
        qemb = qown = None
        if qs:
            qv = np.asarray(await llm.embed(qs), dtype=np.float32)
            qv /= np.linalg.norm(qv, axis=1, keepdims=True) + 1e-12
            qemb, qown = qv, np.asarray(owner, dtype=np.int64)
        return emb, qemb, qown, None
    except LLMError as exc:
        obs.log("index.dense_unavailable", error=type(exc).__name__)
        return None, None, None, str(exc)


def _write(settings: Settings, idx: Index, report_entries: list[dict[str, object]] | None,
           quarantined: list[dict[str, object]] | None, summaries: dict[str, str] | None) -> None:
    out = settings.index_dir
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    aug = idx.aug_tokens or [None] * len(idx.passages)  # type: ignore[list-item]
    with (tmp / "passages.jsonl").open("w", encoding="utf-8") as fh:
        for p, toks, at in zip(idx.passages, idx.tokens, aug, strict=True):
            rec: dict[str, object] = {"passage": p.model_dump(), "tokens": toks}
            if at is not None:
                rec["aug_tokens"] = at
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if idx.readings or summaries:
        with (tmp / "reading.jsonl").open("w", encoding="utf-8") as fh:
            for doc_id, s in (summaries or {}).items():
                fh.write(json.dumps({"doc_id": doc_id, "summary": s}, ensure_ascii=False) + "\n")
            for pid, r in idx.readings.items():
                fh.write(json.dumps({"pid": pid, **r}, ensure_ascii=False) + "\n")
    if idx.embeddings is not None:
        np.save(tmp / "emb_text.npy", idx.embeddings)
    if idx.q_embeddings is not None and idx.q_owner is not None:
        np.save(tmp / "emb_q.npy", idx.q_embeddings)
        np.save(tmp / "q_owner.npy", idx.q_owner)
    # Carry over files that the current step does not regenerate.
    for name in ("ingest_report.jsonl", "quarantine.jsonl"):
        if (out / name).is_file():
            shutil.copy2(out / name, tmp / name)
    if report_entries is not None:
        with (tmp / "ingest_report.jsonl").open("w", encoding="utf-8") as fh:
            for e in report_entries:
                fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    if quarantined is not None:
        (tmp / "quarantine.jsonl").unlink(missing_ok=True)
        for q in quarantined:
            security.quarantine_record(tmp / "quarantine.jsonl", q)
    (tmp / "manifest.json").write_text(json.dumps(idx.manifest, indent=2), encoding="utf-8")
    if out.exists():
        shutil.rmtree(out)
    tmp.replace(out)  # atomic swap of the directory on the same filesystem


def _quarantine_records(docs: list[Document]) -> list[dict[str, object]]:
    return [{"pid": f"{d.doc_id}:p{i}", "file": d.source, "hash": security.chunk_hash(d.paragraphs[i - 1]),
             "text": d.paragraphs[i - 1]} for d in docs for i in d.flagged_paragraphs]


async def build(settings: Settings, llm: LLM | None, *, read: bool | None = None,
                on_progress: Callable[[str, int, int], None] | None = None) -> Index:
    """Step A always; step B (read pass) when `read` (default: settings.read_pass) and a model is given."""
    t0 = time.perf_counter()
    coll = settings.coll()
    docs, report = ingest.load_sources_with_report(coll)

    if settings.guard_model and llm is not None and hasattr(llm, "chat_text"):
        from cgprr.classify import apply_verdicts, classify_docs
        try:
            verdicts = await classify_docs(docs, llm, settings.guard_model)  # type: ignore[arg-type]
            docs = apply_verdicts(docs, verdicts, report)
            if hasattr(llm, "unload"):
                await llm.unload(settings.guard_model)  # small GPUs: free it before the generator loads
        except LLMError as exc:
            report.add(file="*", action="skipped", rule="classifier_unavailable", detail=type(exc).__name__)

    passages = ingest.to_passages(docs)
    if not passages:
        raise ingest.IngestError(f"collection {coll.name!r}: no indexable text after screening")
    tokens = [_raw_tokens(p) for p in passages]
    counts = {a: report.count(a) for a in ("rejected", "quarantined", "redacted", "labelled", "removed", "skipped")}
    base_manifest: dict[str, object] = {
        "version": INDEX_VERSION, "collection": settings.collection, "sources": list(coll.sources),
        "chunking": coll.chunking, "chunk_tokens": coll.chunk_tokens,
        "n_docs": len(docs), "n_passages": len(passages),
        "quarantined": sum(len(d.flagged_paragraphs) for d in docs), "ingest": counts,
        "guard_model": settings.guard_model or None,
    }

    # ---- step A: searchable now ---------------------------------------------------------------
    emb = qemb = qown = None
    dense_error: str | None = None
    if settings.use_dense and llm is not None:
        emb, qemb, qown, dense_error = await _embed_all(settings, llm, passages, {})
    manifest = {**base_manifest, "dense": emb is not None, "embed_model": settings.embed_model if emb is not None else None,
                "embed_dim": int(emb.shape[1]) if emb is not None else None, "dense_error": dense_error,
                "read_pass": {"status": "pending" if (read if read is not None else settings.read_pass) and llm is not None else "off"},
                "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    idx = Index(passages, tokens, emb, manifest)
    _write(settings, idx, report.entries, _quarantine_records(docs), None)
    obs.log("index.build", step="A", n_passages=len(passages), dense=emb is not None,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2))
    if on_progress:
        on_progress("indexed", len(passages), len(passages))

    do_read = (read if read is not None else settings.read_pass) and llm is not None
    if not do_read:
        return idx

    # ---- step B: read pass, then re-index ------------------------------------------------------
    from cgprr.reader import READ_PROMPT_VERSION, read_collection

    t1 = time.perf_counter()
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    if emb is not None and hasattr(llm, "unload"):
        await llm.unload(settings.embed_model)  # small GPUs: the generator gets the whole card for the read pass
    summaries, readings_obj, errors = await read_collection(
        docs, passages, llm, progress_path=settings.index_dir / "read_progress.json",
        on_progress=(lambda d, t: on_progress("reading", d, t)) if on_progress else None)
    readings = {pid: r.model_dump(exclude={"pid"}) for pid, r in readings_obj.items()}
    for pid, r in readings.items():
        if r.get("risk", "none") not in ("none", "violence", "hate"):
            report.add(file=pid.rsplit(":p", 1)[0], action="review", rule=f"reader_risk:{r['risk']}", doc=pid.rsplit(":p", 1)[0],
                       chunk=int(pid.rsplit(":p", 1)[1]))
    aug = [_aug_tokens(p, readings.get(p.pid)) for p in passages]
    if settings.use_dense and llm is not None and emb is not None:
        if hasattr(llm, "unload"):
            await llm.unload(llm.model)  # small GPUs: free the generator before loading the embedder
        emb, qemb, qown, dense_error = await _embed_all(settings, llm, passages, readings)
    manifest = {**manifest, "dense": emb is not None, "dense_error": dense_error,
                "read_pass": {"status": "done" if not errors else "partial", "model": llm.model,
                              "prompt_version": READ_PROMPT_VERSION, "chunks_read": len(readings),
                              "errors": len(errors), "questions": int(qemb.shape[0]) if qemb is not None else
                              sum(len(r.get("questions", [])) for r in readings.values()),  # type: ignore[arg-type]
                              "seconds": round(time.perf_counter() - t1, 1)},
                "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    idx = Index(passages, tokens, emb, manifest, aug, qemb, qown, readings)
    _write(settings, idx, report.entries, None, summaries)
    obs.log("index.build", step="B", n_passages=len(passages), read=len(readings), errors=len(errors),
            latency_ms=round((time.perf_counter() - t1) * 1000, 2))
    return idx


def save_manifest(settings: Settings, manifest: dict[str, object]) -> None:
    p = settings.index_dir / "manifest.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp.replace(p)


def load(settings: Settings) -> Index:
    d = settings.index_dir
    mf = d / "manifest.json"
    if not mf.is_file():
        raise FileNotFoundError(f"index for collection {settings.collection!r} not found at {d}; run `cgprr build-index`")
    manifest = json.loads(mf.read_text(encoding="utf-8"))
    if manifest.get("version") != INDEX_VERSION:
        raise RuntimeError("index version mismatch; rebuild with `cgprr build-index`")
    if manifest.get("collection") != settings.collection:
        raise RuntimeError("index belongs to a different collection; refusing to load (data separation)")
    passages: list[Passage] = []
    tokens: list[list[str]] = []
    aug: list[list[str]] = []
    with (d / "passages.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            passages.append(Passage.model_validate(rec["passage"]))
            tokens.append(list(rec["tokens"]))
            if "aug_tokens" in rec:
                aug.append(list(rec["aug_tokens"]))
    readings: dict[str, dict[str, object]] = {}
    if (d / "reading.jsonl").is_file():
        for line in (d / "reading.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if "pid" in r:
                readings[r.pop("pid")] = r
    emb = qemb = qown = None
    if manifest.get("dense"):
        if manifest.get("embed_model") != settings.embed_model:
            raise RuntimeError(
                f"index embedded with {manifest.get('embed_model')!r} but CGPRR_EMBED_MODEL={settings.embed_model!r}; rebuild")
        emb = np.load(d / "emb_text.npy").astype(np.float32)
        if emb.shape[0] != len(passages):
            raise RuntimeError("embeddings/passages length mismatch; rebuild the index")
        if (d / "emb_q.npy").is_file():
            qemb = np.load(d / "emb_q.npy").astype(np.float32)
            qown = np.load(d / "q_owner.npy").astype(np.int64)
    return Index(passages, tokens, emb, manifest, aug if len(aug) == len(passages) else None, qemb, qown, readings)
