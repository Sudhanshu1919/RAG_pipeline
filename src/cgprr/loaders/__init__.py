"""Document loaders: any supported file -> a flat list of structural Blocks.

Supported: Markdown / plain text, PDF (pypdf), Word .docx (python-docx), HTML (BeautifulSoup,
stdlib parser). Every loader returns the same `Loaded` shape, so chunking and indexing never
depend on the source format.

Security layer 1 (file level) lives here: the type is detected from the file's bytes, never from
its extension; a mismatch, an encrypted PDF, a macro-enabled Office file, a zip bomb, or a file
over the size/page limits is rejected with a named rule. Parsing runs in a separate process
(`sandbox.load_isolated`) so a malformed file can only crash its own worker.

Edge cases NOT handled: scanned (image-only) PDFs — no OCR, they yield no text and are reported as
`no_text`; legacy .doc/.xls/.ppt (binary Office) — rejected as unsupported; PDF text hidden by
render mode or colour — not detected (DOCX/HTML hidden text is).
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from pathlib import Path


class LoaderError(ValueError):
    """A file was rejected. `rule` is a stable machine-readable reason for the ingest report."""

    def __init__(self, rule: str, message: str) -> None:
        super().__init__(f"{rule}: {message}")
        self.rule = rule


@dataclass(frozen=True)
class Limits:
    max_file_bytes: int = 50_000_000
    max_pages: int = 500
    max_zip_uncompressed: int = 200_000_000
    max_zip_ratio: float = 100.0
    max_block_chars: int = 20_000
    timeout_s: float = 60.0


@dataclass
class Block:
    text: str
    heading_path: tuple[str, ...] = ()
    page: int | None = None
    kind: str = "para"  # para | list | table | quote | code
    held: bool = False          # quarantined by screening: kept apart, never indexed
    labels: tuple[str, ...] = ()  # content labels from screening


@dataclass
class Loaded:
    kind: str  # markdown | text | pdf | docx | html
    blocks: list[Block]
    meta: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)  # e.g. "hidden_text_removed:3"


_SUFFIX_KIND = {
    ".md": "markdown", ".markdown": "markdown", ".txt": "text", ".text": "text",
    ".pdf": "pdf", ".docx": "docx", ".html": "html", ".htm": "html", ".xhtml": "html",
}
_MACRO_SUFFIXES = {".docm", ".dotm", ".xlsm", ".xltm", ".pptm", ".potm", ".ppsm"}
_LEGACY_OFFICE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"  # OLE2: .doc/.xls/.ppt


def supported_suffixes() -> set[str]:
    return set(_SUFFIX_KIND)


def sniff(path: Path, head: bytes) -> str:
    """Content-based type. Returns one of the loader kinds or raises LoaderError."""
    suffix = path.suffix.lower()
    if suffix in _MACRO_SUFFIXES:
        raise LoaderError("macro_document", f"{path.name}: macro-enabled Office files are not accepted")
    if head.startswith((b"MZ", b"\x7fELF", b"\xcf\xfa\xed\xfe", b"\xfe\xed\xfa", b"#!")):
        raise LoaderError("executable", f"{path.name}: executable or script content (whatever the extension says)")
    if head.startswith(b"%PDF-"):
        actual = "pdf"
    elif head.startswith(b"PK\x03\x04"):
        actual = "docx" if _is_docx(path) else "zip"
    elif head.startswith(_LEGACY_OFFICE):
        raise LoaderError("unsupported_type", f"{path.name}: legacy binary Office format; save as .docx or PDF")
    elif b"\x00" in head[:4096] and not head.startswith((b"\xff\xfe", b"\xfe\xff")):
        raise LoaderError("binary_file", f"{path.name}: binary content")
    else:
        lowered = head[:2048].lstrip().lower()
        actual = "html" if lowered.startswith((b"<!doctype html", b"<html")) or b"<body" in lowered else "text"
    expected = _SUFFIX_KIND.get(suffix)
    if expected is None:
        raise LoaderError("unsupported_type", f"{path.name}: extension {suffix!r} is not supported")
    if actual == "zip":
        raise LoaderError("type_mismatch", f"{path.name}: zip archive that is not a Word document")
    # Text formats are interchangeable (an HTML snippet saved as .txt is still text); binary must match.
    textual = {"markdown", "text", "html"}
    if expected in textual and actual in textual:
        return expected
    if expected != actual:
        raise LoaderError("type_mismatch", f"{path.name}: extension says {expected} but content is {actual}")
    return actual


def _is_docx(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as z:
            names = set(z.namelist())
    except zipfile.BadZipFile:
        return False
    return "word/document.xml" in names and "[Content_Types].xml" in names


def load_file(path: Path, limits: Limits | None = None) -> Loaded:
    """Parse one file in-process. Prefer `sandbox.load_isolated` for untrusted collections."""
    lim = limits or Limits()
    size = path.stat().st_size
    if size > lim.max_file_bytes:
        raise LoaderError("too_large", f"{path.name}: {size} bytes > limit {lim.max_file_bytes}")
    if size == 0:
        raise LoaderError("empty_file", f"{path.name}: empty")
    with path.open("rb") as fh:
        head = fh.read(8192)
    kind = sniff(path, head)
    if kind == "pdf":
        from cgprr.loaders.pdf import load_pdf
        out = load_pdf(path, lim)
    elif kind == "docx":
        from cgprr.loaders.docx_loader import load_docx
        out = load_docx(path, lim)
    elif kind == "html":
        from cgprr.loaders.html_loader import load_html
        out = load_html(path, lim)
    else:
        from cgprr.loaders.text import load_text
        out = load_text(path, lim, markdown=kind == "markdown")
    from cgprr.textnorm import clean_text

    for b in out.blocks:
        b.text = clean_text(b.text)
        b.heading_path = tuple(clean_text(h) for h in b.heading_path)
    out.meta = {k: clean_text(v) for k, v in out.meta.items()}
    out.blocks = [piece for b in out.blocks if b.text.strip() for piece in _split_oversize(b, lim)]
    if not out.blocks:
        raise LoaderError("no_text", f"{path.name}: no extractable text (scanned PDF? OCR is not supported)")
    return out


def _split_oversize(b: Block, lim: Limits) -> list[Block]:
    """Never truncate: a pathological block (e.g. a PDF page with no paragraph breaks) is cut at
    whitespace into pieces of at most max_block_chars; the chunker splits further by sentence."""
    if len(b.text) <= lim.max_block_chars:
        return [b]
    out: list[Block] = []
    rest = b.text
    while rest:
        cut = len(rest) if len(rest) <= lim.max_block_chars else (rest.rfind(" ", 0, lim.max_block_chars) + 1 or lim.max_block_chars)
        out.append(Block(rest[:cut].strip(), b.heading_path, b.page, b.kind, b.held, b.labels))
        rest = rest[cut:]
    return [x for x in out if x.text]
