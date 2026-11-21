"""Markdown and plain-text loader.

Front matter ('---' fenced `key: value` lines) is optional and becomes `Loaded.meta`.
Markdown headings (`#`..`######`) build the heading path; fenced code blocks are kept whole.
Paragraphs are separated by blank lines and whitespace-collapsed, exactly as the original
ingest did, so a heading-free Markdown file yields the same paragraphs (and passage ids) as before.
"""

from __future__ import annotations

import re
from pathlib import Path

from cgprr.loaders import Block, Limits, Loaded, LoaderError

FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")


def split_front_matter(raw: str, name: str) -> tuple[dict[str, str], str]:
    m = FRONT_MATTER_RE.match(raw)
    if not m:
        return {}, raw
    header: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if not line.strip():
            continue
        if ":" not in line:
            raise LoaderError("bad_front_matter", f"{name}: bad header line {line!r}")
        k, v = line.split(":", 1)
        header[k.strip()] = v.strip()
    return header, m.group(2)


def decode(raw: bytes, name: str) -> str:
    for enc in ("utf-8-sig", "utf-16"):
        try:
            text = raw.decode(enc)
            if enc == "utf-16" and not raw.startswith((b"\xff\xfe", b"\xfe\xff")):
                continue
            return text
        except UnicodeDecodeError:
            continue
    raise LoaderError("bad_encoding", f"{name}: not valid UTF-8/UTF-16 text")


def load_text(path: Path, limits: Limits, *, markdown: bool) -> Loaded:
    raw = decode(path.read_bytes(), path.name)
    meta, body = split_front_matter(raw, path.name)
    blocks = parse_markdown(body) if markdown else parse_plain(body)
    return Loaded(kind="markdown" if markdown else "text", blocks=blocks, meta=meta, notes=["front_matter"] if meta else [])


def _collapse(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def parse_plain(body: str) -> list[Block]:
    return [Block(_collapse(p)) for p in re.split(r"\n\s*\n", body) if _collapse(p)]


def parse_markdown(body: str) -> list[Block]:
    blocks: list[Block] = []
    path: list[str] = []
    buf: list[str] = []
    in_code = False

    def flush(kind: str | None = None) -> None:
        if not buf:
            return
        text = "\n".join(buf)
        buf.clear()
        if not text.strip():
            return
        k = kind or _kind(text)
        blocks.append(Block(text.strip("\n") if k == "code" else _collapse(text), tuple(path), None, k))

    for line in body.splitlines():
        if _FENCE_RE.match(line):
            if in_code:
                buf.append(line)
                flush("code")
                in_code = False
            else:
                flush()
                in_code = True
                buf.append(line)
            continue
        if in_code:
            buf.append(line)
            continue
        m = _HEADING_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            del path[level - 1:]
            path.extend([""] * (level - 1 - len(path)))
            path.append(_collapse(m.group(2)))
            continue
        if not line.strip():
            flush()
            continue
        buf.append(line)
    flush("code" if in_code else None)
    # Drop the empty placeholders used for skipped heading levels (e.g. "#" then "###").
    for b in blocks:
        b.heading_path = tuple(h for h in b.heading_path if h)
    return blocks


def _kind(text: str) -> str:
    first = text.lstrip().split("\n", 1)[0]
    if first.startswith("|"):
        return "table"
    if re.match(r"^([-*+]|\d+[.)])\s", first):
        return "list"
    if first.startswith(">"):
        return "quote"
    return "para"
