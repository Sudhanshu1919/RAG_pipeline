"""Word .docx loader (python-docx).

Before parsing, the zip container is checked: total uncompressed size and compression ratio
(zip-bomb guard) and the presence of a VBA project (macros are rejected even inside a .docx).
Heading styles ("Heading 1..9", "Title") build the heading path; tables become one block per
table with rows joined by " | ". Hidden text — runs marked hidden, white font, or font size
below 2 pt — is removed and counted in `notes` (it is a common way to plant instructions).

Edge cases NOT handled: text boxes, footnotes, comments and tracked changes (not extracted);
nested tables (flattened); embedded objects/images (ignored).
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from cgprr.loaders import Block, Limits, Loaded, LoaderError


def check_zip(path: Path, limits: Limits) -> None:
    try:
        with zipfile.ZipFile(path) as z:
            infos = z.infolist()
    except zipfile.BadZipFile as exc:
        raise LoaderError("unreadable", f"{path.name}: corrupt zip container") from exc
    total = sum(i.file_size for i in infos)
    packed = sum(i.compress_size for i in infos) or 1
    if total > limits.max_zip_uncompressed or total / packed > limits.max_zip_ratio:
        raise LoaderError("zip_bomb", f"{path.name}: uncompressed {total} bytes, ratio {total / packed:.0f}")
    names = {i.filename.lower() for i in infos}
    if any(n.endswith("vbaproject.bin") for n in names):
        raise LoaderError("macro_document", f"{path.name}: contains a VBA macro project")


def load_docx(path: Path, limits: Limits) -> Loaded:
    check_zip(path, limits)
    import docx
    from docx.oxml.ns import qn

    try:
        d = docx.Document(str(path))
    except Exception as exc:
        raise LoaderError("unreadable", f"{path.name}: {type(exc).__name__}") from exc

    hidden = 0
    blocks: list[Block] = []
    stack: list[str] = []

    def visible_text(par: object) -> str:
        nonlocal hidden
        parts: list[str] = []
        for run in par.runs:  # type: ignore[attr-defined]
            f = run.font
            color = f.color.rgb if f.color is not None and f.color.type is not None else None
            tiny = f.size is not None and f.size.pt < 2
            if f.hidden or tiny or str(color).upper() == "FFFFFF":
                if run.text.strip():
                    hidden += 1
                continue
            parts.append(run.text)
        return "".join(parts)

    body = d.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            par = docx.text.paragraph.Paragraph(child, d)
            text = " ".join(visible_text(par).split())
            if not text:
                continue
            style = (par.style.name if par.style is not None else "") or ""
            level = _heading_level(style)
            if level:
                del stack[level - 1:]
                stack.append(text)
                continue
            kind = "list" if "list" in style.lower() else "para"
            blocks.append(Block(text, tuple(stack), None, kind))
        elif child.tag == qn("w:tbl"):
            table = docx.table.Table(child, d)
            rows = []
            for row in table.rows:
                cells = [" ".join(visible_text(p).split()) for c in row.cells for p in c.paragraphs]
                cells = [c for c in dict.fromkeys(cells) if c]  # merged cells repeat
                if cells:
                    rows.append(" | ".join(cells))
            if rows:
                blocks.append(Block("\n".join(rows), tuple(stack), None, "table"))

    cp = d.core_properties
    meta = {k: v for k, v in {
        "title": cp.title or "", "publisher": cp.author or "",
        "date": cp.created.date().isoformat() if cp.created else "",
    }.items() if v}
    notes = [f"hidden_text_removed:{hidden}"] if hidden else []
    return Loaded(kind="docx", blocks=blocks, meta=meta, notes=notes)


def _heading_level(style: str) -> int:
    s = style.strip().lower()
    if s == "title":
        return 1
    if s.startswith("heading "):
        tail = s.split(" ", 1)[1]
        return int(tail) if tail.isdigit() and 1 <= int(tail) <= 9 else 0
    return 0
