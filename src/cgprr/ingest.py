"""Ingest: collection source files -> validated Documents (chunked) -> Passages.

Any supported file (Markdown/TXT, PDF, DOCX, HTML; see `cgprr.loaders`) is parsed in an isolated
worker, split into chunks (`cgprr.chunking`), and screened chunk by chunk (`cgprr.security`).
Chunk ids are 1-based in file order -> citation id "<doc_id>:p<n>".

Doc ids: the front-matter `id` when present and valid, otherwise filename slug + 6 hex of the
content hash (stable across rebuilds, changes when the content changes).

Strict collections (`require_synthetic = true`) refuse the whole build on any bad file, as the
original loader did. Other collections skip a rejected file, record why in the ingest report, and
continue: one bad upload must not block a newsroom's whole collection.

Edge cases NOT handled: nested YAML front matter, multi-line header values, incremental ingest
(every build re-reads all files; the read-pass cache makes unchanged chunks cheap).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from cgprr import obs, security
from cgprr.chunking import chunk_blocks
from cgprr.collections import Collection
from cgprr.loaders import Block, Limits, Loaded, LoaderError, supported_suffixes
from cgprr.loaders.sandbox import load_isolated, ok
from cgprr.loaders.text import FRONT_MATTER_RE, split_front_matter
from cgprr.schemas import DOC_ID_RX, Document, Passage
from cgprr.textnorm import clean_text

_FM_RE = FRONT_MATTER_RE  # kept for backward compatibility of imports
_MAX_PARAGRAPH_CHARS = 4_000
_DOC_ID_RE = re.compile(rf"^{DOC_ID_RX}$")


class Article(BaseModel):
    model_config = ConfigDict(extra="forbid")
    article_id: str = Field(pattern=rf"^{DOC_ID_RX}$")
    title: str
    outlet: str
    date: str
    paragraphs: list[str]

    @property
    def body(self) -> str:
        return "\n\n".join(self.paragraphs)


class IngestError(ValueError):
    pass


@dataclass
class IngestReport:
    """One entry per decision. Never holds document text: only ids, rules and actions."""
    entries: list[dict[str, object]] = field(default_factory=list)

    def add(self, *, file: str, action: str, rule: str, doc: str | None = None, chunk: int | None = None,
            detail: str = "") -> None:
        e: dict[str, object] = {"file": file, "action": action, "rule": rule}
        if doc:
            e["doc"] = doc
        if chunk is not None:
            e["chunk"] = chunk
        if detail:
            e["detail"] = detail[:300]
        self.entries.append(e)
        obs.log("ingest.decision", **e)

    def count(self, action: str) -> int:
        return sum(1 for e in self.entries if e["action"] == action)


def doc_id_for(path: Path, raw: bytes) -> str:
    """Stable id for a file without an explicit `id:`: filename slug + 6 hex of its content hash."""
    slug = re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")[:48] or "doc"
    return f"{slug}-{hashlib.sha256(raw).hexdigest()[:6]}"


def _check_synthetic(meta: dict[str, str], name: str) -> None:
    if meta.get("synthetic", "").lower() != "true":
        # Brief §5: public or synthetic data only. Refuse unlabelled files loudly.
        raise IngestError(f"{name}: front matter must declare 'synthetic: true' (or adapt for a named public dataset)")


def _parse(path: Path, *, require_front_matter: bool = True, require_synthetic: bool = True) -> tuple[dict[str, str], list[str]]:
    """Paragraph-level parse of a Markdown file (used for articles in strict collections)."""
    raw_bytes = path.read_bytes()
    if len(raw_bytes) > 2_000_000:
        raise IngestError(f"{path.name}: exceeds 2000000 bytes")
    try:
        header, body = split_front_matter(clean_text(raw_bytes.decode("utf-8")), path.name)
    except (LoaderError, UnicodeDecodeError) as exc:
        raise IngestError(str(exc)) from exc
    if not header and (require_front_matter or require_synthetic):
        raise IngestError(f"{path.name}: missing front matter")
    if require_synthetic:
        _check_synthetic(header, path.name)
    header.setdefault("id", doc_id_for(path, raw_bytes))
    header.setdefault("title", path.stem)
    paras = [p for p in (re.sub(r"\s+", " ", x).strip() for x in re.split(r"\n\s*\n", body)) if p]
    for i, p in enumerate(paras, 1):
        if len(p) > _MAX_PARAGRAPH_CHARS:
            raise IngestError(f"{path.name}: paragraph {i} exceeds {_MAX_PARAGRAPH_CHARS} chars")
    return header, paras


def _to_document(coll: Collection, path: Path, loaded: Loaded, report: IngestReport, seen: set[str],
                 screener: security.Screener) -> Document | None:
    meta = loaded.meta
    if coll.require_synthetic:
        _check_synthetic(meta, path.name)
    raw = path.read_bytes()
    doc_id = meta.get("id", "")
    if not _DOC_ID_RE.fullmatch(doc_id):
        doc_id = doc_id_for(path, raw)
    if doc_id in seen:
        if coll.require_synthetic:
            raise IngestError(f"duplicate doc id {doc_id}")
        report.add(file=path.name, action="rejected", rule="duplicate_doc_id", doc=doc_id)
        return None
    for note in loaded.notes:
        if note.startswith("hidden_text_removed"):
            report.add(file=path.name, action="removed", rule="hidden_text", doc=doc_id, detail=note)

    # Screen block by block BEFORE packing: redaction and labels apply in place; a quarantined block is
    # held apart so the chunker gives it a chunk of its own (benign neighbours stay searchable).
    screened: list[Block] = []
    for bno, b in enumerate(loaded.blocks, 1):
        v = screener.screen(b.text)
        for rule in v.rules:
            report.add(file=path.name, action=v.action_for(rule), rule=rule, doc=doc_id, chunk=bno)
        if v.action == "reject_file":
            report.add(file=path.name, action="rejected", rule="file_rejected_by_content_policy", doc=doc_id)
            return None  # nothing from this file is stored
        screened.append(Block(v.text, b.heading_path, b.page, b.kind, v.action == "quarantine", tuple(v.labels)))

    chunks = chunk_blocks(screened, mode=coll.chunking, chunk_tokens=coll.chunk_tokens)
    texts = [c.text for c in chunks]
    headings = [c.heading for c in chunks]
    pages = [c.page_start for c in chunks]
    flagged = [i for i, c in enumerate(chunks, 1) if c.held]
    labels = {i: list(c.labels) for i, c in enumerate(chunks, 1) if c.labels}
    for i in flagged:
        obs.log("ingest.quarantine", doc=doc_id, para=i)
    if not texts:
        report.add(file=path.name, action="rejected", rule="no_text", doc=doc_id)
        return None
    seen.add(doc_id)
    return Document(
        doc_id=doc_id, title=meta.get("title") or path.stem, publisher=meta.get("publisher", ""),
        date=meta.get("date", ""), doc_type=meta.get("doc_type", loaded.kind), paragraphs=texts,
        flagged_paragraphs=flagged, headings=headings, pages=pages, source=path.name,
        labels={str(k): v for k, v in labels.items()},
    )


def load_sources_with_report(coll: Collection, *, limits: Limits | None = None,
                             screener: security.Screener | None = None) -> tuple[list[Document], IngestReport]:
    report = IngestReport()
    sup = supported_suffixes()
    files: list[Path] = []
    for f in coll.source_files():
        rel_parts = f.relative_to(coll.root.resolve()).parts
        if any(part.startswith(".") for part in rel_parts) or f.name == "collection.toml":  # hidden files and folders (.git, .upload-*)
            continue
        if f.suffix.lower() not in sup:
            if coll.require_synthetic:
                continue  # strict corpora: unrelated files (e.g. a README) are simply not sources
            report.add(file=f.name, action="skipped", rule="unsupported_type", detail=f.suffix)
            continue
        files.append(f)
    if not files:
        raise IngestError(f"no source files for collection {coll.name!r} (sources: {list(coll.sources)})")

    results = load_isolated(files, limits)
    scr = screener or security.Screener(coll)
    docs: list[Document] = []
    seen: set[str] = set()
    for f in files:
        res = results[f]
        loaded = ok(res)
        if loaded is None:
            if coll.require_synthetic:
                raise IngestError(str(res[2]))
            report.add(file=f.name, action="rejected", rule=str(res[1]), detail=str(res[2]))
            continue
        doc = _to_document(coll, f, loaded, report, seen, scr)  # IngestError propagates (strict collections)
        if doc is not None:
            docs.append(doc)
    if not docs:
        raise IngestError(f"collection {coll.name!r}: every source file was rejected; see the ingest report")
    return docs, report


def load_sources(coll: Collection) -> list[Document]:
    return load_sources_with_report(coll)[0]


def to_passages(docs: list[Document]) -> list[Passage]:
    out: list[Passage] = []
    for d in docs:
        for i, text in enumerate(d.paragraphs, 1):
            if i in d.flagged_paragraphs:
                continue  # quarantined: never indexed, never shown to the model
            out.append(
                Passage(
                    pid=f"{d.doc_id}:p{i}", doc_id=d.doc_id, para_no=i, title=d.title,
                    publisher=d.publisher, date=d.date, doc_type=d.doc_type, text=text,
                    heading=d.headings[i - 1] if i <= len(d.headings) else "",
                    page=d.pages[i - 1] if i <= len(d.pages) else None,
                    source=d.source, labels=d.labels.get(str(i), []),
                )
            )
    return out


# --------------------------------------------------------------------------- articles (Track C input)


def _article_path(article_dir: Path, article_id: str) -> Path:
    if not _DOC_ID_RE.fullmatch(article_id):
        raise IngestError(f"invalid article id {article_id!r}")
    for suffix in sorted(supported_suffixes()):
        p = article_dir / f"{article_id}{suffix}"
        if p.is_file():
            return p
    raise IngestError(f"article {article_id} not found in {article_dir}")


def load_article_file(path: Path, *, require_synthetic: bool = False, article_id: str | None = None) -> Article:
    """Any supported file -> Article. Strict (synthetic) Markdown articles keep the original checks."""
    aid = article_id or re.sub(r"[^A-Za-z0-9_.-]+", "-", path.stem).strip("-")[:80] or "article"
    if require_synthetic:
        h, paras = _parse(path)
        if h.get("id") != aid:
            raise IngestError(f"{path.name}: id mismatch")
        return Article(article_id=aid, title=h.get("title", ""), outlet=h.get("outlet", ""),
                       date=h.get("date", ""), paragraphs=paras)
    # Same isolation as collection files: an article handed over with --file is untrusted too
    # (separate worker, timeout, memory cap), not parsed in the CLI process.
    res = load_isolated([path])[path]
    loaded = ok(res)
    if loaded is None:
        raise IngestError(str(res[2]))
    paras = [b.text for b in loaded.blocks]
    return Article(article_id=aid, title=loaded.meta.get("title") or path.stem,
                   outlet=loaded.meta.get("outlet", loaded.meta.get("publisher", "")),
                   date=loaded.meta.get("date", ""), paragraphs=paras)


def load_article(article_dir: Path, article_id: str, *, require_synthetic: bool = True) -> Article:
    return load_article_file(_article_path(article_dir, article_id), require_synthetic=require_synthetic,
                             article_id=article_id)


def list_articles(article_dir: Path) -> list[str]:
    if not article_dir.is_dir():
        return []
    sup = supported_suffixes()
    return sorted({p.stem for p in article_dir.iterdir() if p.is_file() and p.suffix.lower() in sup and not p.name.startswith(".")})
