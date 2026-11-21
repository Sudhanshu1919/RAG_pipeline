"""Phase 0: collections, generic ids, prompt-local citation labels."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import BaseModel

from cgprr import prompts
from cgprr.collections import CollectionError, load_collection, parse_selection
from cgprr.config import Settings
from cgprr.index import build, load
from cgprr.ingest import load_sources, to_passages
from cgprr.rag import ask
from cgprr.retrieve import Retriever
from cgprr.schemas import LLMAnswer, LLMStatement
from cgprr.textnorm import number_values
from cgprr.verify import resolve_labels
from tests.conftest import FakeLLM


def _make(root: Path, name: str, files: dict[str, str], toml_extra: str = "") -> None:
    d = root / "collections" / name
    (d / "docs").mkdir(parents=True)
    (d / "collection.toml").write_text(f'name = "{name}"\nsources = ["docs/*"]\n{toml_extra}', encoding="utf-8")
    for fn, text in files.items():
        (d / "docs" / fn).write_text(text, encoding="utf-8")


def test_shipped_collection_loads() -> None:
    root = Path(__file__).resolve().parents[1]
    c = load_collection(root, "cgprr-news")
    assert c.require_synthetic and len(c.source_files()) == 10 and c.eval_dir.is_dir()


def test_plain_files_get_stable_slug_ids(tmp_path: Path) -> None:
    _make(tmp_path, "plain", {"Annual Report 2025.txt": "Revenue was 4.2 billion.\n\nStaff numbered 310."})
    docs = load_sources(load_collection(tmp_path, "plain"))
    assert docs[0].doc_id.startswith("annual-report-2025-") and len(docs[0].doc_id) == len("annual-report-2025-") + 6
    # structure chunking packs the two short paragraphs into one chunk
    assert [p.pid for p in to_passages(docs)] == [f"{docs[0].doc_id}:p1"]
    assert "Revenue was 4.2 billion." in to_passages(docs)[0].text and "Staff numbered 310." in to_passages(docs)[0].text
    again = load_sources(load_collection(tmp_path, "plain"))
    assert again[0].doc_id == docs[0].doc_id  # same content -> same id


def test_require_synthetic_is_per_collection(tmp_path: Path) -> None:
    _make(tmp_path, "strict", {"a.md": "No front matter here."}, "require_synthetic = true\n")
    with pytest.raises(ValueError):
        load_sources(load_collection(tmp_path, "strict"))


def test_bad_names_and_selection() -> None:
    with pytest.raises(CollectionError):
        parse_selection("../etc", "x")
    assert parse_selection("a, b,a", "x") == ["a", "b"]
    assert parse_selection(None, "news") == ["news"]


async def test_index_of_other_collection_is_refused(tmp_path: Path) -> None:
    _make(tmp_path, "one", {"a.txt": "Alpha fact."})
    _make(tmp_path, "two", {"b.txt": "Beta fact."})
    s1 = Settings(root=tmp_path, collection="one", use_dense=False, cache_enabled=False)
    await build(s1, None)
    # Point collection "two" at collection one's index directory: must refuse (data separation).
    (tmp_path / "indexes" / "one").rename(tmp_path / "indexes" / "two")
    with pytest.raises(RuntimeError, match="different collection"):
        load(replace(s1, collection="two"))


def test_label_resolution_and_citation_stripping() -> None:
    st = resolve_labels(LLMStatement(text="Tariffs fall 25% [P2].", citations=["P2", "S01:p1", "P9"]),
                        {"P1": "S02:p1", "P2": "S01:p2"})
    assert st.text == "Tariffs fall 25%."
    assert st.citations == ["S01:p2", "S01:p1", "P9"]  # unknown label kept -> gate 2 rejects it
    assert number_values("see P3 and [P12] and kotp-3fa21c:p4") == set()


async def test_model_citing_labels_is_answered_with_real_ids(retriever: Retriever) -> None:
    def h(name: str, schema: type[BaseModel], user: str) -> BaseModel:
        # Find which label the prompt gave S01:p2, exactly as a real model would see it.
        lab = next(lb for lb, pid in prompts.label_map(_shown(retriever, user)).items() if pid == "S01:p2")
        return LLMAnswer(answerable=True, statements=[LLMStatement(text="The agreement covers 1,200 tariff lines.", citations=[lab])])
    r = await ask("How many tariff lines does KOTP cover?", retriever, FakeLLM(h))
    assert r.status == "answered" and r.answer.endswith("[S01:p2]") and "[P" not in r.answer


async def test_unknown_label_is_refused(retriever: Retriever) -> None:
    def h(name: str, schema: type[BaseModel], user: str) -> BaseModel:
        return LLMAnswer(answerable=True, statements=[LLMStatement(text="The agreement covers 1,200 tariff lines.", citations=["P9"])])
    r = await ask("How many tariff lines does KOTP cover?", retriever, FakeLLM(h))
    assert r.refusal_reason == "gate2_citation_not_in_context"


def _shown(retriever: Retriever, user: str) -> list:  # type: ignore[type-arg]
    """Passages in prompt order, recovered from the prompt text (labels P1..Pk in order)."""
    by_text = {p.text: p for p in retriever.idx.passages}
    out = []
    for chunk in user.split("<passage ")[1:]:
        body = chunk.split(">", 1)[1].split("</passage>")[0]
        out.append(by_text[body.replace("‹", "<").replace("›", ">")])
    return out
