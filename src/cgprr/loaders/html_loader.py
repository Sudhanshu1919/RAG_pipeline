"""HTML loader (BeautifulSoup with the stdlib html.parser: no native dependency).

Removed before extraction: script, style, noscript, template, iframe, object, embed, svg, form,
nav, header, footer, aside; elements hidden with the `hidden` attribute, aria-hidden="true", or an
inline style of display:none / visibility:hidden / font-size:0 / opacity:0 (counted in notes).
Nothing is fetched: links, images and remote resources are never requested.
h1..h6 build the heading path; p, li, blockquote, pre, td/th rows, dd/dt become blocks.

Edge cases NOT handled: text hidden by external CSS classes (stylesheets are not evaluated),
content rendered by JavaScript (not executed), <br>-only layouts without block elements
(fallback: the body's remaining text becomes paragraphs split at blank lines).
"""

from __future__ import annotations

import re
from pathlib import Path

from cgprr.loaders import Block, Limits, Loaded
from cgprr.loaders.text import decode

_DROP = ["script", "style", "noscript", "template", "iframe", "object", "embed", "svg", "form",
         "nav", "header", "footer", "aside", "head"]
_HIDDEN_STYLE = re.compile(r"display\s*:\s*none|visibility\s*:\s*hidden|font-size\s*:\s*0(px|pt|em|%)?\s*(;|$)|opacity\s*:\s*0(\.0+)?\s*(;|$)", re.IGNORECASE)
_BLOCK_TAGS = {"p", "li", "blockquote", "pre", "dd", "dt", "figcaption", "caption"}
_HEAD_TAGS = {f"h{i}": i for i in range(1, 7)}


def load_html(path: Path, limits: Limits) -> Loaded:
    from bs4 import BeautifulSoup

    raw = decode(path.read_bytes(), path.name)
    soup = BeautifulSoup(raw, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    meta: dict[str, str] = {"title": title} if title else {}
    for name, key in (("author", "publisher"), ("date", "date"), ("article:published_time", "date")):
        tag = soup.find("meta", attrs={"name": name}) or soup.find("meta", attrs={"property": name})
        if tag and tag.get("content"):
            meta.setdefault(key, str(tag["content"])[:200])

    hidden = 0
    for el in soup.find_all(True):
        if el.decomposed:
            continue
        style = el.get("style") or ""
        if el.has_attr("hidden") or str(el.get("aria-hidden", "")).lower() == "true" or _HIDDEN_STYLE.search(str(style)):
            if el.get_text(strip=True):
                hidden += 1
            el.decompose()
    for tag in soup.find_all(_DROP):
        if tag.decomposed:
            continue
        # <header>/<footer> inside an <article>/<main> often hold the headline or byline: keep those.
        if tag.name in ("header", "footer") and tag.find_parent(["article", "main"]) is not None:
            continue
        tag.decompose()

    root = soup.body or soup
    blocks: list[Block] = []
    stack: list[str] = []
    for el in root.find_all(list(_BLOCK_TAGS) + list(_HEAD_TAGS) + ["tr"]):
        if el.find_parent(list(_BLOCK_TAGS - {"li"}) + ["tr"]) is not None and el.name != "li":
            continue  # nested inside another block we already take whole
        if el.name == "li" and el.find_parent("li") is not None:
            continue
        if el.name in _HEAD_TAGS:
            lvl = _HEAD_TAGS[el.name]
            text = el.get_text(" ", strip=True)
            if text:
                del stack[lvl - 1:]
                stack.append(text)
            continue
        if el.name == "tr":
            cells = [c.get_text(" ", strip=True) for c in el.find_all(["td", "th"])]
            text = " | ".join(c for c in cells if c)
            kind = "table"
        else:
            text = el.get_text("\n" if el.name == "pre" else " ", strip=True)
            kind = {"li": "list", "blockquote": "quote", "pre": "code"}.get(el.name, "para")
        if el.name != "pre":
            text = " ".join(text.split())
        if text:
            blocks.append(Block(text, tuple(stack), None, kind))

    if not blocks:  # <br>-only or bare-text pages
        for p in re.split(r"\n\s*\n", root.get_text("\n")):
            if " ".join(p.split()):
                blocks.append(Block(" ".join(p.split())))
    notes = [f"hidden_text_removed:{hidden}"] if hidden else []
    return Loaded(kind="html", blocks=blocks, meta=meta, notes=notes)
