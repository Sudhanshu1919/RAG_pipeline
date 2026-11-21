from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from cgprr import obs
from cgprr.config import Settings
from cgprr.llm import CircuitOpenError, ContextOverflowError, LLMError, LLMOutputError, LocalLLM, decode_schema
from cgprr.schemas import LLMAnswer


def _client(tmp_path: Path, handler, **kw) -> LocalLLM:  # type: ignore[no-untyped-def]
    obs.configure(tmp_path / "l.jsonl")
    s = Settings(root=tmp_path, cache_enabled=kw.pop("cache", False), llm_max_retries=kw.pop("retries", 2),
                 breaker_threshold=kw.pop("threshold", 5), **kw)
    llm = LocalLLM(s)
    llm._client = httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(handler))
    return llm


def _ok(content: dict[str, object]) -> httpx.Response:
    return httpx.Response(200, json={"message": {"content": json.dumps(content)}, "prompt_eval_count": 10, "eval_count": 5})


async def test_retry_then_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("asyncio.sleep", _nosleep)
    n = {"i": 0}

    def h(req: httpx.Request) -> httpx.Response:
        n["i"] += 1
        return httpx.Response(503) if n["i"] == 1 else _ok({"answerable": False, "statements": []})

    llm = _client(tmp_path, h)
    out = await llm.complete(LLMAnswer, "s", "u", name="t")
    assert out.answerable is False and n["i"] == 2


async def test_non_retryable_4xx_fails_fast(tmp_path: Path) -> None:
    n = {"i": 0}

    def h(req: httpx.Request) -> httpx.Response:
        n["i"] += 1
        return httpx.Response(400)

    with pytest.raises(LLMError):
        await _client(tmp_path, h).complete(LLMAnswer, "s", "u", name="t")
    assert n["i"] == 1


async def test_breaker_opens(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("asyncio.sleep", _nosleep)
    llm = _client(tmp_path, lambda r: httpx.Response(503), retries=0, threshold=2)
    for _ in range(2):
        with pytest.raises(LLMError):
            await llm.complete(LLMAnswer, "s", "u", name="t")
    with pytest.raises(CircuitOpenError):
        await llm.complete(LLMAnswer, "s", "u", name="t")


async def test_embed_failures_do_not_block_chat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing embedding model (e.g. bge-m3 not pulled) must not open the breaker for chat calls."""
    monkeypatch.setattr("asyncio.sleep", _nosleep)

    def h(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/embed"):
            return httpx.Response(500)
        return _ok({"answerable": False, "statements": []})

    llm = _client(tmp_path, h, retries=0, threshold=2)
    for _ in range(3):
        with pytest.raises(LLMError):
            await llm.embed(["x"])
    out = await llm.complete(LLMAnswer, "s", "u", name="t")
    assert out.answerable is False


async def test_has_model(tmp_path: Path) -> None:
    def h(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "qwen3:4b-instruct-2507-q4_K_M"}, {"name": "bge-m3:latest"}]})

    llm = _client(tmp_path, h)
    assert await llm.has_model("bge-m3") is True
    assert await llm.has_model("qwen3:4b-instruct-2507-q4_K_M") is True
    assert await llm.has_model("llama-guard3:1b") is False


async def test_schema_violation_repair_then_fail(tmp_path: Path) -> None:
    llm = _client(tmp_path, lambda r: _ok({"wrong": 1}))
    with pytest.raises(LLMOutputError):
        await llm.complete(LLMAnswer, "s", "u", name="t")


async def test_cache_hit_skips_server(tmp_path: Path) -> None:
    n = {"i": 0}

    def h(req: httpx.Request) -> httpx.Response:
        n["i"] += 1
        return _ok({"answerable": False, "statements": []})

    llm = _client(tmp_path, h, cache=True)
    await llm.complete(LLMAnswer, "s", "u", name="t")
    await llm.complete(LLMAnswer, "s", "u", name="t")
    assert n["i"] == 1


def test_decode_schema_has_no_regex() -> None:
    assert '"pattern"' not in json.dumps(decode_schema(LLMAnswer))


async def _nosleep(_: float) -> None:
    return None


async def test_context_overflow_fails_before_call(tmp_path: Path) -> None:
    n = {"i": 0}

    def h(req: httpx.Request) -> httpx.Response:
        n["i"] += 1
        return _ok({"answerable": False, "statements": []})

    llm = _client(tmp_path, h, num_ctx=1024, num_predict=256)
    with pytest.raises(ContextOverflowError):
        await llm.complete(LLMAnswer, "s", "x" * 4000, name="t")
    assert n["i"] == 0


async def test_options_sent_to_ollama(tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def h(req: httpx.Request) -> httpx.Response:
        seen.update(json.loads(req.content))
        return _ok({"answerable": False, "statements": []})

    await _client(tmp_path, h, num_ctx=2048, num_predict=512).complete(LLMAnswer, "s", "u", name="t")
    assert seen["options"] == {"temperature": 0, "seed": 7, "num_ctx": 2048, "num_predict": 512}
    assert isinstance(seen["format"], dict)
