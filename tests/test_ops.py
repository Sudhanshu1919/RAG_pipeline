"""Collection operations used by the CLI: user collections live under the data root, uploads are
checked at once, excluded files are never parsed, shipped samples are read-only."""

from __future__ import annotations

from pathlib import Path

import pytest

from cgprr import ops
from cgprr.collections import CollectionError, list_collections, load_collection
from cgprr.config import Settings
from cgprr.index import build
from tests.fixtures_gen import make_docx

TEXT_A = b"# Port report\n\nThe Velmora port handled 2.4 million TEU in FY2025.\n"
TEXT_B = b"# Draft\n\nUnreviewed draft figures: 9.9 million TEU.\n"


def _settings(root: Path, name: str) -> Settings:
    return Settings(root=root, collection=name, use_dense=False, read_pass=False, guard_model="")


def test_user_collection_lives_in_data_root_and_is_listed(tmp_path: Path) -> None:
    coll = ops.create_collection(tmp_path, "desk", "my documents")
    assert coll.writable and coll.root == tmp_path / "data" / "collections" / "desk"
    assert "desk" in list_collections(tmp_path)
    with pytest.raises(CollectionError):
        ops.create_collection(tmp_path, "desk")  # names are unique


def test_upload_checks_name_type_and_content(tmp_path: Path) -> None:
    coll = ops.create_collection(tmp_path, "desk")
    docx = make_docx(tmp_path / "src.docx").read_bytes()
    res = ops.add_files(coll, [
        ("../../etc/report.md", TEXT_A),           # path parts are dropped, the base name is kept
        ("notes.md", TEXT_A),
        ("notes.md", TEXT_B),                      # same name again: stored as notes-1.md
        ("brief.docx", docx),
        ("invoice.pdf", b"MZ\x90\x00 not a pdf"),  # executable renamed to .pdf
        ("script.exe", b"MZ"),                     # unsupported type
        ("empty.txt", b""),
    ])
    stored = [r.name for r in res if r.ok]
    assert stored == ["report.md", "notes.md", "notes-1.md", "brief.docx"]
    refused = {r.original: r.rule for r in res if not r.ok}
    assert refused == {"invoice.pdf": "executable", "script.exe": "bad_name", "empty.txt": "empty_file"}
    files = sorted(p.name for p in (coll.root / "docs").iterdir())
    assert files == ["brief.docx", "notes-1.md", "notes.md", "report.md"]  # no temporary files left behind


async def test_excluded_file_is_never_parsed(tmp_path: Path) -> None:
    coll = ops.create_collection(tmp_path, "desk")
    ops.add_files(coll, [("report.md", TEXT_A), ("draft.md", TEXT_B)])
    ops.set_included(coll, "docs/draft.md", False)
    s = _settings(tmp_path, "desk")
    idx = await build(s, None, read=False)
    assert all("9.9 million" not in p.text for p in idx.passages)
    listing = {r["name"]: r for r in ops.file_listing(s)}
    assert listing["draft.md"]["state"] == "excluded" and not listing["draft.md"]["included"]
    ops.set_included(load_collection(tmp_path, "desk"), "docs/draft.md", True)
    idx2 = await build(s, None, read=False)
    assert any("9.9 million" in p.text for p in idx2.passages)


async def test_build_records_file_changes(tmp_path: Path) -> None:
    coll = ops.create_collection(tmp_path, "desk")
    ops.add_files(coll, [("report.md", TEXT_A)])
    s = _settings(tmp_path, "desk")
    m1 = await ops.build_collection(s, None, read=False, calibrate=False)
    assert m1["changes"]["added"] == ["docs/report.md"]
    ops.add_files(load_collection(tmp_path, "desk"), [("draft.md", TEXT_B)])
    m2 = await ops.build_collection(s, None, read=False, calibrate=False)
    assert m2["changes"]["added"] == ["docs/draft.md"] and m2["changes"]["changed"] == []
    assert {r["state"] for r in ops.file_listing(s)} == {"indexed"}


def test_shipped_sample_is_read_only() -> None:
    sample = load_collection(Path(__file__).resolve().parents[1], "cgprr-news")
    assert not sample.writable
    with pytest.raises(CollectionError):
        ops.add_files(sample, [("x.md", TEXT_A)])
    with pytest.raises(CollectionError):
        ops.set_included(sample, "source_pack/S01.md", False)
