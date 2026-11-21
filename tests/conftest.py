"""Test fixtures. FakeLLM replaces the model server so gates and orchestration can be tested
deterministically. These tests verify the CODE PATHS; they say nothing about model quality,
which is measured only by `cgprr eval` against a real local model."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import pytest
from pydantic import BaseModel

from cgprr import obs
from cgprr.config import Settings
from cgprr.index import build
from cgprr.retrieve import Retriever

REPO = Path(__file__).resolve().parents[1]
T = TypeVar("T", bound=BaseModel)
Handler = Callable[[str, type[BaseModel], str], BaseModel]


class FakeLLM:
    model = "fake-llm"

    def __init__(self, handler: Handler) -> None:
        self.handler = handler
        self.calls: list[str] = []

    async def complete(self, schema: type[T], system: str, user: str, *, name: str, max_tokens: int | None = None) -> T:
        self.calls.append(name)
        out = self.handler(name, schema, user)
        assert isinstance(out, schema)
        return out

    async def embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        raise AssertionError("dense disabled in tests")

    async def aclose(self) -> None:
        return None


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    try:
        (tmp_path / "collections").symlink_to(REPO / "collections")
    except OSError:  # Windows without symlink privilege (WinError 1314): copy instead
        shutil.copytree(REPO / "collections", tmp_path / "collections")
    obs.configure(tmp_path / "logs" / "test.jsonl")
    return Settings(root=tmp_path, collection="cgprr-news", use_dense=False, cache_enabled=False)


@pytest.fixture()
async def retriever(settings: Settings) -> Retriever:
    idx = await build(settings, None)
    return Retriever(idx, settings, None)
