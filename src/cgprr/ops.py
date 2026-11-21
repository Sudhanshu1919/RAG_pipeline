"""Collection operations shared by the CLI and the web UI (one implementation, two front ends).

Everything here works on the same files the pipeline reads: `collection.toml`, the collection's
`docs/` folder, `indexes/<name>/` and `review.json`. Nothing here talks to a browser or a terminal.

Upload safety: a file name is reduced to its base name and a safe character set (no path traversal,
no hidden files), its type must be supported, and its first bytes are sniffed immediately so an
executable renamed to .pdf is refused at upload, not only at build time. Full parsing still happens
in the sandboxed build.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import tomllib
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cgprr import obs
from cgprr.collections import (
    Collection,
    CollectionError,
    list_collections,
    load_collection,
    user_collections_dir,
    validate_name,
)
from cgprr.config import Settings
from cgprr.loaders import Limits, LoaderError, sniff, supported_suffixes

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._ ()-]+")
MAX_FILES_PER_UPLOAD = 50


@dataclass
class AddResult:
    name: str            # stored file name ("" when refused)
    original: str
    ok: bool
    rule: str = ""       # why it was refused
    detail: str = ""


# --------------------------------------------------------------------------- toml

def _toml_value(v: object) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int | float):
        return str(v)
    if isinstance(v, list | tuple):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    return json.dumps(str(v), ensure_ascii=False)  # a JSON string is a valid TOML basic string


def write_config(path: Path, cfg: dict[str, object]) -> None:
    """Write a flat collection.toml (the only shape collections use)."""
    lines = [f"{k} = {_toml_value(v)}" for k, v in cfg.items() if v is not None]
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def _read_config(coll: Collection) -> dict[str, Any]:
    with coll.config_path.open("rb") as fh:
        return tomllib.load(fh)


def _require_writable(coll: Collection) -> None:
    if not coll.writable:
        raise CollectionError(f"collection {coll.name!r} is a shipped sample and read-only; create your own collection")


# --------------------------------------------------------------------------- create / files

def create_collection(root: Path, name: str, description: str = "", *, redact: list[str] | None = None) -> Collection:
    validate_name(name)
    if name in list_collections(root):
        raise CollectionError(f"collection {name!r} already exists")
    folder = user_collections_dir(root) / name
    (folder / "docs").mkdir(parents=True, exist_ok=True)
    (folder / "eval").mkdir(exist_ok=True)
    cfg: dict[str, object] = {"name": name, "description": description.strip()[:300], "sources": ["docs/**/*"],
                              "require_synthetic": False, "chunking": "structure", "chunk_tokens": 350}
    if redact is not None:
        cfg["redact"] = [k for k in redact if k in {"secret", "card", "aadhaar", "email", "phone"}]
    write_config(folder / "collection.toml", cfg)
    obs.log("ops.collection_created", collection=name)
    return load_collection(root, name)


def safe_filename(raw: str) -> str:
    """Base name only, safe characters, supported suffix. Raises CollectionError otherwise."""
    base = raw.replace("\\", "/").rsplit("/", 1)[-1].strip()
    stem, dot, suffix = base.rpartition(".")
    if not dot or not stem:
        raise CollectionError(f"{raw!r}: file name needs an extension")
    suffix = "." + suffix.lower()
    if suffix not in supported_suffixes():
        raise CollectionError(f"{raw!r}: type {suffix} is not supported (PDF, DOCX, MD/TXT, HTML)")
    stem = _SAFE_NAME.sub("_", stem).strip(" ._")[:100]
    if not stem:
        raise CollectionError(f"{raw!r}: no usable file name")
    return stem + suffix


def _unique(folder: Path, name: str) -> Path:
    dest = folder / name
    stem, suffix = dest.stem, dest.suffix
    n = 1
    while dest.exists():
        dest = folder / f"{stem}-{n}{suffix}"
        n += 1
    return dest


def add_files(coll: Collection, items: Iterable[tuple[str, bytes]], *, limits: Limits | None = None) -> list[AddResult]:
    """Store uploaded files in the collection's docs/ folder after a quick type check."""
    _require_writable(coll)
    lim = limits or Limits()
    docs = coll.root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    out: list[AddResult] = []
    for i, (original, data) in enumerate(items):
        if i >= MAX_FILES_PER_UPLOAD:
            out.append(AddResult("", original, False, "too_many_files", f"at most {MAX_FILES_PER_UPLOAD} per upload"))
            continue
        try:
            name = safe_filename(original)
        except CollectionError as exc:
            out.append(AddResult("", original, False, "bad_name", str(exc)))
            continue
        if not data:
            out.append(AddResult("", original, False, "empty_file", "empty"))
            continue
        if len(data) > lim.max_file_bytes:
            out.append(AddResult("", original, False, "too_large", f"{len(data)} bytes > {lim.max_file_bytes}"))
            continue
        dest = _unique(docs, name)
        # Write under a hidden temporary name with the real suffix (sniff opens a .docx as a zip and
        # checks the suffix); hidden files are never treated as sources, even if a crash leaves one.
        tmp = docs / f".upload-{secrets.token_hex(6)}{dest.suffix}"
        tmp.write_bytes(data)
        try:
            sniff(tmp, data[:8192])
        except LoaderError as exc:
            tmp.unlink(missing_ok=True)
            out.append(AddResult("", original, False, exc.rule, str(exc).replace(tmp.name, name)))
            continue
        tmp.replace(dest)
        out.append(AddResult(dest.name, original, True))
    obs.log("ops.files_added", collection=coll.name, added=sum(r.ok for r in out), refused=sum(not r.ok for r in out))
    return out


def add_paths(coll: Collection, paths: Iterable[Path]) -> list[AddResult]:
    """CLI helper: add files or whole folders from disk.

    Inside a folder, hidden files/folders (.git, .DS_Store) and symlinks are skipped, so a folder upload
    cannot pull in files from elsewhere on disk. Size is checked before a file is read into memory.
    Folders larger than MAX_FILES_PER_UPLOAD are added in batches."""
    lim = Limits()
    out: list[AddResult] = []
    files: list[Path] = []
    for src in paths:
        if not src.exists():
            out.append(AddResult("", str(src), False, "not_found", "no such file or folder"))
        elif src.is_file():
            files.append(src)
        else:
            files += sorted(p for p in src.rglob("*") if p.is_file() and not p.is_symlink()
                            and not any(part.startswith(".") for part in p.relative_to(src).parts))
    for i in range(0, len(files), MAX_FILES_PER_UPLOAD):
        items: list[tuple[str, bytes]] = []
        for f in files[i:i + MAX_FILES_PER_UPLOAD]:
            size = f.stat().st_size
            if size > lim.max_file_bytes:
                out.append(AddResult("", f.name, False, "too_large", f"{size} bytes > {lim.max_file_bytes}"))
                continue
            items.append((f.name, f.read_bytes()))
        out += add_files(coll, items, limits=lim)
    return out


def _resolve_member(coll: Collection, rel: str) -> Path:
    p = (coll.root / rel).resolve()
    if coll.root.resolve() not in p.parents or not p.is_file():
        raise CollectionError(f"{rel!r} is not a file of collection {coll.name!r}")
    return p


def remove_file(coll: Collection, rel: str) -> None:
    _require_writable(coll)
    _resolve_member(coll, rel).unlink()
    if rel in coll.exclude:
        set_included(coll, rel, True)  # drop the stale exclude entry
    obs.log("ops.file_removed", collection=coll.name)


def set_included(coll: Collection, rel: str, included: bool) -> None:
    """Switch a file in or out of the index. Excluded files are never parsed, chunked, read or embedded."""
    _require_writable(coll)
    cfg = _read_config(coll)
    excl = [str(x) for x in cfg.get("exclude", [])]
    if included:
        excl = [x for x in excl if x != rel]
    else:
        _resolve_member(coll, rel)
        if rel not in excl:
            excl.append(rel)
    cfg["exclude"] = excl
    write_config(coll.config_path, cfg)


def delete_collection(root: Path, name: str) -> None:
    import shutil

    from cgprr.config import Settings as _S

    coll = load_collection(root, name)
    _require_writable(coll)
    s = _S(root=root, collection=name)
    for d in (coll.root, s.index_dir, s.cache_dir.parent, s.out_dir):
        if d.exists():
            shutil.rmtree(d)
    if s.log_path.exists():
        s.log_path.unlink()
    obs.log("ops.collection_deleted", collection=name)


# --------------------------------------------------------------------------- status / listing

def _file_hash(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def _manifest(s: Settings) -> dict[str, Any]:
    p = s.index_dir / "manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def _report(s: Settings) -> list[dict[str, Any]]:
    p = s.index_dir / "ingest_report.jsonl"
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()] if p.is_file() else []


def file_listing(s: Settings) -> list[dict[str, Any]]:
    """Every file with its include switch and its state relative to the last build."""
    coll = s.coll()
    built = _manifest(s).get("files", {})
    rejected = {e["file"]: e.get("rule", "") for e in _report(s) if e.get("action") == "rejected"}
    rows = []
    for p in coll.all_source_files():
        rel = coll.rel(p)
        st = p.stat()
        included = rel not in coll.exclude
        h = _file_hash(p)
        if not included:
            state = "excluded"
        elif p.name in rejected and rel in built:
            state = "rejected"
        elif rel not in built:
            state = "new"
        elif built[rel] != h:
            state = "changed"
        else:
            state = "indexed"
        rows.append({"file": rel, "name": p.name, "bytes": st.st_size, "type": p.suffix.lower().lstrip("."),
                     "included": included, "state": state, "rule": rejected.get(p.name, "")})
    return rows


def collection_status(s: Settings) -> dict[str, Any]:
    coll = s.coll()
    m = _manifest(s)
    prog_p = s.index_dir / "read_progress.json"
    prog = json.loads(prog_p.read_text(encoding="utf-8")) if prog_p.is_file() else None
    files = coll.all_source_files()
    th = m.get("thresholds") or {}
    return {
        "name": coll.name, "description": coll.description, "writable": coll.writable,
        "files_total": len(files), "files_included": len(coll.source_files()),
        "built": bool(m), "built_at": m.get("built_at"), "n_docs": m.get("n_docs", 0), "n_passages": m.get("n_passages", 0),
        "dense": m.get("dense", False), "read_pass": (m.get("read_pass") or {}).get("status", "none"),
        "read_progress": prog, "thresholds": {k: th.get(k) for k in ("min_coverage", "min_dense", "source")} if th else None,
        "ingest": m.get("ingest", {}),
    }


# --------------------------------------------------------------------------- build

async def build_collection(s: Settings, llm: Any, *, read: bool, calibrate: bool,
                           on_progress: Callable[[str, int, int], None] | None = None) -> dict[str, Any]:
    """index.build (+ calibration) and a per-file hash record for change detection. Returns the manifest."""
    from cgprr.index import build, load, save_manifest
    from cgprr.retrieve import Retriever

    coll = s.coll()
    before = _manifest(s).get("files", {})
    idx = await build(s, llm, read=read, on_progress=on_progress)
    files = {coll.rel(p): _file_hash(p) for p in coll.source_files()}
    idx.manifest["files"] = files
    idx.manifest["changes"] = {
        "added": sorted(set(files) - set(before)), "removed": sorted(set(before) - set(files)),
        "changed": sorted(k for k in files if k in before and before[k] != files[k]),
        "excluded": sorted(coll.exclude),
    }
    if calibrate:
        from cgprr.calibrate import calibrate as _calibrate
        from cgprr.evalgen import load_items

        if on_progress:
            on_progress("calibrating", 0, 1)
        other_qs: list[str] = []
        for o in [n for n in list_collections(s.root) if n != s.collection][:3]:
            try:
                other_qs += [str(q["question"]) for q in load_items(s.for_collection(o).eval_dir, "qa", split=None)
                             if q.get("expected") == "answer"][:10]
            except CollectionError:
                continue
        ret = Retriever(load(s), s, llm if s.use_dense else None)
        th = await _calibrate(ret, s.eval_dir, others=other_qs)
        idx.manifest["thresholds"] = {k: v for k, v in th.items() if not isinstance(v, list)}
    save_manifest(s, idx.manifest)
    if on_progress:
        on_progress("done", 1, 1)
    return idx.manifest


# --------------------------------------------------------------------------- review

def review_summary(s: Settings) -> dict[str, Any]:
    from cgprr.security import load_review

    decisions = load_review(s.coll())
    q = s.index_dir / "quarantine.jsonl"
    rows = [json.loads(line) for line in q.read_text(encoding="utf-8").splitlines() if line.strip()] if q.is_file() else []
    counts = Counter((e["action"], e["rule"]) for e in _report(s))
    return {
        "counts": [{"action": a, "rule": r, "n": n} for (a, r), n in sorted(counts.items())],
        "rejected": [{"file": e["file"], "rule": e["rule"], "detail": e.get("detail", "")}
                     for e in _report(s) if e.get("action") == "rejected"],
        "quarantined": [{**r, "state": decisions.get(str(r.get("hash")), "pending")} for r in rows],
    }


def set_review(s: Settings, release: list[str], keep: list[str]) -> Path:
    from cgprr.security import load_review, save_review

    c = s.coll()
    decisions = load_review(c)
    for h in release:
        decisions[h] = "release"
    for h in keep:
        decisions[h] = "keep_quarantined"
    return save_review(c, decisions)
