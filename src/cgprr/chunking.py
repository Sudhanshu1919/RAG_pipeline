"""Structure-aware chunking: Blocks -> retrieval chunks.

Modes (per collection, `chunking = "structure" | "paragraph"` in collection.toml):
- structure (default): never crosses a heading boundary; consecutive blocks under the same heading
  are packed up to a token budget (`chunk_tokens`, at 3.2 chars/token, the same conservative ratio
  the LLM client uses); a block larger than the budget is split at sentence ends with one sentence
  of overlap; a short trailing piece of a section is merged into its predecessor.
- paragraph: one block = one chunk (the original behaviour; keeps the shipped gold passage ids).
  Oversized blocks are still split by sentence.

A block held by screening (quarantine) always becomes a chunk of its own, so quarantining one
paragraph never hides the benign text around it. Every chunk keeps its heading path, page range
and the union of its blocks' content labels.

Edge cases NOT handled: sentence splitting is regex-based (abbreviation list, not a parser);
tables larger than the budget are split by rows without repeating the header row.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from cgprr.loaders import Block

CHARS_PER_TOKEN = 3.2
_ABBREV = ("Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "No.", "St.", "Gen.", "Col.", "Lt.", "Sgt.", "Inc.", "Ltd.",
           "Co.", "Corp.", "vs.", "e.g.", "i.e.", "etc.", "U.S.", "U.K.", "U.N.", "Jan.", "Feb.", "Mar.", "Apr.",
           "Jun.", "Jul.", "Aug.", "Sep.", "Sept.", "Oct.", "Nov.", "Dec.", "approx.", "Fig.", "Rs.")
_SENT_END = re.compile(r"(?<=[.!?…])[\"”’)]?\s+(?=[A-Z0-9\"“‘(\[])")


@dataclass
class Chunk:
    text: str
    heading_path: tuple[str, ...] = ()
    page_start: int | None = None
    page_end: int | None = None
    held: bool = False
    labels: tuple[str, ...] = ()
    fragments: bool = False  # True once tiny table/row fragments were coalesced into this chunk

    @property
    def heading(self) -> str:
        return " > ".join(self.heading_path)


def sentences(text: str) -> list[str]:
    protected = text
    for i, ab in enumerate(_ABBREV):
        protected = protected.replace(ab, ab.replace(".", f"\x00{i}\x00"))
    out = []
    for p in _SENT_END.split(protected):
        for i, ab in enumerate(_ABBREV):
            p = p.replace(ab.replace(".", f"\x00{i}\x00"), ab)
        if p.strip():
            out.append(p.strip())
    return out


def _like(b: Block, text: str) -> Block:
    return Block(text, b.heading_path, b.page, b.kind, b.held, b.labels)


def _split_long(b: Block, budget: int) -> list[Block]:
    if len(b.text) <= budget:
        return [b]
    table = b.kind == "table"
    units = b.text.split("\n") if table else sentences(b.text)
    sep = "\n" if table else " "
    pieces: list[str] = []
    cur: list[str] = []
    for u in units:
        while len(u) > budget:  # one "sentence" longer than the budget: hard cut at a space
            cut = u.rfind(" ", 0, budget) + 1 or budget
            if cur:
                pieces.append(sep.join(cur))
                cur = []
            pieces.append(u[:cut].strip())
            u = u[cut:].strip()
        if cur and len(sep.join([*cur, u])) > budget:
            pieces.append(sep.join(cur))
            # one-sentence overlap keeps a fact that straddles the cut retrievable from both sides
            cur = cur[-1:] if not table and len(cur[-1]) + len(u) < budget else []
        if u:
            cur.append(u)
    if cur:
        pieces.append(sep.join(cur))
    return [_like(b, p) for p in pieces if p]


def chunk_blocks(blocks: list[Block], *, mode: str = "structure", chunk_tokens: int = 350) -> list[Chunk]:
    budget = int(chunk_tokens * CHARS_PER_TOKEN)
    if mode == "paragraph":
        return [Chunk(p.text, p.heading_path, p.page, p.page, p.held, p.labels)
                for b in blocks for p in _split_long(b, max(budget, 4000))]
    if mode != "structure":
        raise ValueError(f"unknown chunking mode {mode!r}")

    chunks: list[Chunk] = []
    cur: list[Block] = []

    def emit() -> None:
        if not cur:
            return
        pages = [b.page for b in cur if b.page is not None]
        labels = tuple(dict.fromkeys(lb for b in cur for lb in b.labels))
        chunks.append(Chunk("\n".join(b.text for b in cur), cur[0].heading_path,
                            min(pages) if pages else None, max(pages) if pages else None, cur[0].held, labels))
        cur.clear()

    for big in blocks:
        for b in _split_long(big, budget):
            if cur and (b.held or cur[0].held or b.heading_path != cur[0].heading_path
                        or sum(len(x.text) + 1 for x in cur) + len(b.text) > budget):
                emit()
            cur.append(b)
    emit()

    merged: list[Chunk] = []
    for c in chunks:
        prev = merged[-1] if merged else None
        if (prev is not None and not prev.held and not c.held and prev.heading_path == c.heading_path
                and len(c.text) < 0.25 * budget and len(prev.text) + len(c.text) <= 1.25 * budget):
            prev.text = f"{prev.text}\n{c.text}"
            prev.page_end = c.page_end if c.page_end is not None else prev.page_end
            prev.labels = tuple(dict.fromkeys(prev.labels + c.labels))
        else:
            merged.append(c)
    return _merge_fragments(merged, budget)


def _merge_fragments(chunks: list[Chunk], budget: int) -> list[Chunk]:
    """Coalesce runs of tiny, consecutive, same-page chunks that sit under *different* headings.

    Plain-text PDF extraction turns a table into one line per cell/row, and a Title-Case row label
    ("Tertiary Sector GVA") looks like a section heading — so a table explodes into many 1-line
    "sections" that are too small to rank. Each fragment keeps its heading inline ("heading: text"),
    so the row label stays searchable and citable. Only paged sources (PDF), whose headings are
    guessed from line shape, are affected; real sections (any chunk >= 15% of the budget) and
    Markdown/DOCX/HTML headings are never touched."""
    tiny = 0.15 * budget
    out: list[Chunk] = []
    for c in chunks:
        prev = out[-1] if out else None
        if (prev is not None and not prev.held and not c.held and len(c.text) < tiny
                and c.page_start is not None and prev.page_end == c.page_start
                and c.heading_path != prev.heading_path
                and (prev.fragments or len(prev.text) < tiny)
                and len(prev.text) + len(c.text) + len(c.heading) + 2 <= budget):
            if not prev.fragments:
                if prev.heading_path:
                    prev.text = f"{prev.heading_path[-1]}: {prev.text}"
                prev.fragments = True
            label = c.heading_path[-1] if c.heading_path else ""
            prev.text = f"{prev.text}\n{label}: {c.text}" if label else f"{prev.text}\n{c.text}"
            prev.labels = tuple(dict.fromkeys(prev.labels + c.labels))
        else:
            out.append(c)
    return out
