"""PDF loader (pypdf). Text PDFs only; page numbers are kept on every block.

Paragraphs: pypdf returns lines. Lines are joined into paragraphs, a paragraph ends at a blank
line or at a line that ends a sentence and is clearly shorter than the page's typical line.
Words hyphenated across a line break are re-joined. Headings are detected heuristically: a short
line (<= 90 chars) with no final full stop that is numbered ("2.1 Results"), ALL CAPS, or Title Case.

Edge cases NOT handled: multi-column layouts (columns may interleave), tables (cells come out as
lines of text), scanned pages (no OCR), text hidden by render mode/colour, running headers/footers
are removed only when the same line repeats on >= 60% of pages.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from cgprr.loaders import Block, Limits, Loaded, LoaderError

_NUMBERED_HEADING = re.compile(r"^(\d+(\.\d+){0,3}\.?|[IVXLC]+\.|[A-Z]\.)\s+\S")
_SMALL_WORDS = {"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the", "to", "with", "vs", "from"}


def load_pdf(path: Path, limits: Limits) -> Loaded:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(str(path), strict=False)
    except (PdfReadError, ValueError, OSError) as exc:
        raise LoaderError("unreadable", f"{path.name}: {type(exc).__name__}") from exc
    if reader.is_encrypted:
        raise LoaderError("encrypted_pdf", f"{path.name}: encrypted/password-protected PDF")
    n = len(reader.pages)
    if n > limits.max_pages:
        raise LoaderError("too_many_pages", f"{path.name}: {n} pages > limit {limits.max_pages}")

    pages: list[list[str]] = []
    for page in reader.pages:
        try:
            txt = page.extract_text() or ""
        except Exception:  # noqa: BLE001 - one bad page must not lose the document
            txt = ""
        pages.append([ln.rstrip() for ln in txt.splitlines()])

    repeated = _running_lines(pages)
    blocks: list[Block] = []
    path_stack: tuple[str, ...] = ()
    for pno, lines in enumerate(pages, 1):
        lines = [ln for ln in lines if ln.strip() and ln.strip() not in repeated and not _page_number(ln)]
        typical = _typical_len(lines)
        para: list[str] = []
        for ln in lines:
            s = ln.strip()
            if _is_heading(s, typical):
                _flush(blocks, para, path_stack, pno)
                path_stack = (s,)
                continue
            para.append(s)
            if s.endswith((".", "?", "!", ":", '."', ".”")) and len(s) < 0.75 * typical:
                _flush(blocks, para, path_stack, pno)
        _flush(blocks, para, path_stack, pno)

    meta: dict[str, str] = {}
    info = reader.metadata
    if info:
        if info.title:
            meta["title"] = str(info.title)
        if info.author:
            meta["publisher"] = str(info.author)
        if info.creation_date:
            meta["date"] = info.creation_date.date().isoformat()
    return Loaded(kind="pdf", blocks=blocks, meta=meta, notes=[f"pages:{n}"])


def _flush(blocks: list[Block], para: list[str], path: tuple[str, ...], page: int) -> None:
    if para:
        blocks.append(Block(_join(para), path, page, "para"))
        para.clear()


def _join(lines: list[str]) -> str:
    out = ""
    for ln in lines:
        if out.endswith("-") and ln[:1].islower():
            out = out[:-1] + ln  # re-join hyphenation at line break
        else:
            out = f"{out} {ln}" if out else ln
    return re.sub(r"\s+", " ", out).strip()


def _typical_len(lines: list[str]) -> int:
    lens = sorted(len(s.strip()) for s in lines)
    return max(40, lens[int(len(lens) * 0.75)]) if lens else 80


def _running_lines(pages: list[list[str]]) -> set[str]:
    if len(pages) < 3:
        return set()
    seen: Counter[str] = Counter()
    for lines in pages:
        seen.update({ln.strip() for ln in lines[:3] + lines[-3:] if ln.strip()})
    return {ln for ln, c in seen.items() if c >= 0.6 * len(pages)}


def _page_number(line: str) -> bool:
    return bool(re.fullmatch(r"\s*(page\s*)?\d{1,4}(\s*(/|of)\s*\d{1,4})?\s*", line, re.IGNORECASE))


def _is_heading(s: str, typical: int) -> bool:
    if not s or len(s) > 90 or s.endswith((".", ",", ";", ":")) or len(s) > 0.8 * typical:
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'\-]*", s)
    if not words or len(words) > 12:
        return False
    if _NUMBERED_HEADING.match(s):
        return True
    if s.isupper() and len(words) >= 1 and len(s) >= 4:
        return True
    caps = [w[0].isupper() for w in words if w.lower() not in _SMALL_WORDS]
    return len(words) >= 2 and bool(caps) and all(caps)
