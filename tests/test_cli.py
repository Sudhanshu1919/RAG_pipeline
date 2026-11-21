"""CLI smoke test without a model server: create -> build (step A) -> list/info/review -> eval review."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cgprr.cli import main


def test_cli_collection_lifecycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    src = tmp_path / "incoming"
    src.mkdir()
    (src / "brief.md").write_text("# Budget\nThe council approved a budget of 4.5 million crowns.\n\nContact: desk@council.example\n", encoding="utf-8")
    (src / "notes.xyz").write_text("ignored", encoding="utf-8")
    monkeypatch.setenv("CGPRR_ROOT", str(tmp_path))
    monkeypatch.setenv("CGPRR_USE_DENSE", "0")
    monkeypatch.setenv("CGPRR_READ_PASS", "0")
    monkeypatch.setenv("CGPRR_ISOLATE_PARSING", "0")
    assert main(["collection", "create", "council", str(src), "--description", "test"]) == 0
    assert main(["-c", "council", "collection", "build"]) == 0
    manifest = json.loads((tmp_path / "indexes" / "council" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_docs"] == 1 and manifest["ingest"]["redacted"] == 1
    assert manifest["thresholds"]["source"].startswith("default")
    capsys.readouterr()
    assert main(["collection", "list"]) == 0
    assert "council" in capsys.readouterr().out
    assert main(["-c", "council", "collection", "review"]) == 0
    assert "pii:email" in capsys.readouterr().out
    assert main(["-c", "council", "eval", "retrieval"]) == 0
    assert main(["-c", "council", "eval", "review"]) == 0
    assert main(["-c", "nope", "collection", "info"]) == 1      # unknown collection: clean error, exit 1
    assert main(["-c", "../x", "collection", "info"]) == 1      # path traversal in a name is rejected


def test_path_args_survive_terminal_line_wrap() -> None:
    from cgprr.cli import _clean_path_arg

    assert _clean_path_arg("data\collections\x\docs\r\n\report.pdf ") == "data\collections\x\docs\report.pdf"
