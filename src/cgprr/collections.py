"""Named document collections. A collection is the unit of data separation: its own source
files, index, log, cache, eval set and output directory.

Two roots are searched, in this order:
    collections/<name>/            shipped samples (versioned, read-only from the UI)
    $CGPRR_DATA_DIR/collections/   user uploads (default <root>/data, git-ignored); writable
A name may exist in only one of them.

Layout:
    collections/<name>/collection.toml
    collections/<name>/<sources...>      (paths/globs listed in the toml, relative to this folder)
    collections/<name>/eval/             (qa_questions.jsonl, claims_gold.jsonl, *_auto.jsonl)

collection.toml keys (all optional except `name`):
    name = "cgprr-news"
    description = "..."
    sources = ["source_pack/*.md"]     # globs, relative to the collection folder
    articles = "articles"              # folder of articles to fact-check (Track C)
    require_synthetic = false          # refuse files whose front matter lacks `synthetic: true`
    chunk_tokens = 350                 # target chunk size for structure-aware chunking
    chunking = "structure"             # or "paragraph": one blank-line paragraph = one chunk
    exclude = ["draft.pdf"]            # files kept on disk but never parsed or indexed

Edge cases NOT handled: collections nested inside each other, globs that escape the collection
folder (rejected), remote sources (URLs) — files must be on local disk.
"""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class CollectionError(ValueError):
    pass


@dataclass(frozen=True)
class Collection:
    name: str
    root: Path
    description: str = ""
    sources: tuple[str, ...] = ("docs/**/*",)
    articles: str | None = None
    require_synthetic: bool = False
    chunk_tokens: int = 350
    chunking: str = "structure"  # structure | paragraph
    extra: dict[str, object] = field(default_factory=dict)
    exclude: tuple[str, ...] = ()  # paths relative to the collection folder; never parsed
    writable: bool = False         # True for user collections under the data root

    @property
    def config_path(self) -> Path:
        return self.root / "collection.toml"

    @property
    def eval_dir(self) -> Path:
        return self.root / "eval"

    @property
    def article_dir(self) -> Path:
        return self.root / (self.articles or "articles")

    def rel(self, p: Path) -> str:
        return p.resolve().relative_to(self.root.resolve()).as_posix()

    def all_source_files(self) -> list[Path]:
        """Every file the source globs match, excluded ones included (for listings)."""
        return self._match(skip_excluded=False)

    def source_files(self) -> list[Path]:
        """Files to parse: the source globs minus `exclude`."""
        return self._match(skip_excluded=True)

    def _match(self, *, skip_excluded: bool) -> list[Path]:
        base = self.root.resolve()
        excluded = set(self.exclude)
        out: dict[Path, None] = {}
        for pattern in self.sources:
            for p in sorted(self.root.glob(pattern)):
                rp = p.resolve()
                if not rp.is_file():
                    continue
                if base not in rp.parents:
                    raise CollectionError(f"{self.name}: source {p} is outside the collection folder")
                if rp.name == "collection.toml" or (skip_excluded and rp.relative_to(base).as_posix() in excluded):
                    continue
                out[rp] = None
        return list(out)


def collections_dir(root: Path) -> Path:
    """Shipped (sample) collections."""
    return root / "collections"


def data_dir(root: Path) -> Path:
    """Where users' documents and UI state live; git-ignored. Override with CGPRR_DATA_DIR."""
    raw = os.environ.get("CGPRR_DATA_DIR", "")
    return Path(raw).resolve() if raw else root / "data"


def user_collections_dir(root: Path) -> Path:
    return data_dir(root) / "collections"


def _find(root: Path, name: str) -> tuple[Path, bool] | None:
    hits = [(d / name, w) for d, w in ((collections_dir(root), False), (user_collections_dir(root), True))
            if (d / name / "collection.toml").is_file()]
    if len(hits) > 1:
        raise CollectionError(f"collection {name!r} exists both as a sample and in the data folder; rename one")
    return hits[0] if hits else None


def validate_name(name: str) -> str:
    if not _NAME_RE.fullmatch(name):
        raise CollectionError(f"invalid collection name {name!r}: use lowercase letters, digits, '-' or '_'")
    return name


def load_collection(root: Path, name: str) -> Collection:
    validate_name(name)
    found = _find(root, name)
    if found is None:
        raise CollectionError(f"collection {name!r} not found; see `cgprr collection list`")
    folder, writable = found
    cfg_path = folder / "collection.toml"
    with cfg_path.open("rb") as fh:
        cfg = tomllib.load(fh)
    if cfg.get("name", name) != name:
        raise CollectionError(f"{cfg_path}: name {cfg.get('name')!r} does not match folder {name!r}")
    sources = cfg.get("sources", ["docs/**/*"])
    if isinstance(sources, str):
        sources = [sources]
    chunk_tokens = int(cfg.get("chunk_tokens", 350))
    if not 64 <= chunk_tokens <= 2000:
        raise CollectionError(f"{cfg_path}: chunk_tokens must be in [64, 2000]")
    chunking = str(cfg.get("chunking", "structure"))
    if chunking not in ("structure", "paragraph"):
        raise CollectionError(f"{cfg_path}: chunking must be 'structure' or 'paragraph'")
    exclude = cfg.get("exclude", [])
    if not isinstance(exclude, list):
        raise CollectionError(f"{cfg_path}: exclude must be a list of file paths")
    known = {"name", "description", "sources", "articles", "require_synthetic", "chunk_tokens", "chunking", "exclude"}
    return Collection(
        name=name, root=folder, description=str(cfg.get("description", "")),
        sources=tuple(str(s) for s in sources), articles=cfg.get("articles"),
        require_synthetic=bool(cfg.get("require_synthetic", False)), chunk_tokens=chunk_tokens, chunking=chunking,
        extra={k: v for k, v in cfg.items() if k not in known},
        exclude=tuple(str(x) for x in exclude), writable=writable,
    )


def list_collections(root: Path) -> list[str]:
    names: set[str] = set()
    for d in (collections_dir(root), user_collections_dir(root)):
        if d.is_dir():
            names |= {p.name for p in d.iterdir() if (p / "collection.toml").is_file()}
    return sorted(names)


def parse_selection(raw: str | None, default: str) -> list[str]:
    """'a,b' -> ['a', 'b'] (validated, de-duplicated, order kept)."""
    names = [n.strip() for n in (raw or default).split(",") if n.strip()]
    if not names:
        raise CollectionError("no collection selected")
    return [validate_name(n) for n in dict.fromkeys(names)]
