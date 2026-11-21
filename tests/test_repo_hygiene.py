"""Files that bash reads must have LF line endings (a CRLF made `set -o pipefail` fail under WSL)."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_bash_read_files_have_lf_endings() -> None:
    files = [*REPO.glob("scripts/*.sh"), *REPO.glob("profiles/*.env"), *REPO.glob("collections/*/collection.toml")]
    assert files
    bad = [str(f.relative_to(REPO)) for f in files if b"\r\n" in f.read_bytes()]
    assert bad == [], f"CRLF line endings in {bad}"


_SKIP = {".venv", "indexes", "logs", ".cache", "out", "data", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".git"}
_TEXT = {".py", ".md", ".toml", ".yml", ".yaml", ".ipynb", ".sh", ".env", ".jsonl", ".txt", ".html"}


def test_every_shipped_text_file_has_lf_endings() -> None:
    files = [p for p in REPO.rglob("*") if p.is_file() and not (set(p.relative_to(REPO).parts) & _SKIP)
             and not any(x.endswith(".egg-info") for x in p.relative_to(REPO).parts)
             and (p.suffix in _TEXT or p.name in {"Dockerfile", ".gitignore", ".gitattributes"})]
    assert len(files) > 50
    bad = [str(f.relative_to(REPO)) for f in files if b"\r\n" in f.read_bytes()]
    assert bad == [], f"CRLF line endings in {bad}"
