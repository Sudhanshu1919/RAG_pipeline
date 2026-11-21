"""Local LLM client: structured JSON output, retries with exponential backoff + jitter,
circuit breaker, concurrency limit, exact-match disk cache, token/latency logging.

Backends (both local; no paid cloud API on any path):
- ollama:        POST /api/chat with `format=<JSON Schema>`; POST /api/embed
- openai_compat: POST /v1/chat/completions with response_format json_schema (vLLM, llama.cpp
                 server, LM Studio); POST /v1/embeddings

Edge cases NOT handled:
- Cache is exact-match per (model, prompt, schema). No semantic cache: for a fact-check
  workload, near-duplicate prompts with different numbers must NOT share answers.
- Cache is local disk, single host. Horizontal scale -> swap for Redis keyed identically.
- Breaker state is per-process; multiple workers do not share it.
- Streaming is not used; a slow generation holds the semaphore until timeout.
- Model output that is schema-valid but semantically empty is caught downstream by gates.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from cgprr import obs
from cgprr.config import Settings

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


class CircuitOpenError(LLMError):
    pass


class LLMOutputError(LLMError):
    pass


class ContextOverflowError(LLMError):
    """Prompt would not fit in num_ctx. Raised BEFORE the call: Ollama otherwise truncates the
    oldest tokens silently, which would drop the system prompt (the injection/grounding rules)."""


# Conservative chars-per-token for English + JSON schema text (Qwen/Llama BPE average ~3.5-4.2).
_CHARS_PER_TOKEN = 3.2


class LLM(Protocol):
    model: str

    async def complete(self, schema: type[T], system: str, user: str, *, name: str, max_tokens: int | None = None) -> T: ...

    async def embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]: ...

    async def aclose(self) -> None: ...


def _as_int(v: object) -> int:
    return int(v) if isinstance(v, (int, float)) else 0


def _strip(node: object) -> object:
    if isinstance(node, dict):
        return {k: _strip(v) for k, v in node.items() if not (k in {"pattern", "title"} and isinstance(v, str))}
    if isinstance(node, list):
        return [_strip(v) for v in node]
    return node


def decode_schema(schema: type[BaseModel]) -> dict[str, object]:
    """Schema sent to the decoder. Regex `pattern`s are removed because grammar converters
    support them inconsistently; Pydantic still enforces them on receipt."""
    out = _strip(schema.model_json_schema())
    assert isinstance(out, dict)
    return out


class CircuitBreaker:
    def __init__(self, threshold: int, cooldown_s: float) -> None:
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self.failures = 0
        self.opened_at: float | None = None

    def before(self) -> None:
        if self.opened_at is None:
            return
        if time.monotonic() - self.opened_at >= self.cooldown_s:
            self.opened_at = None  # half-open: allow one trial call
            self.failures = self.threshold - 1
            return
        raise CircuitOpenError("LLM circuit open; local model server unhealthy")

    def success(self) -> None:
        self.failures = 0
        self.opened_at = None

    def failure(self) -> None:
        self.failures += 1
        if self.failures >= self.threshold:
            self.opened_at = time.monotonic()
            obs.log("llm.breaker_open", failures=self.failures)


_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


class LocalLLM:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.model = settings.llm_model
        self._client = httpx.AsyncClient(
            base_url=settings.llm_url,
            timeout=httpx.Timeout(settings.llm_timeout_s, connect=5.0),
            # Ignore HTTP(S)_PROXY from the environment: prompts carry document text and must go straight
            # to the local model server, never through a proxy.
            trust_env=False,
        )
        self._sem = asyncio.Semaphore(settings.llm_concurrency)
        self._breaker = CircuitBreaker(settings.breaker_threshold, settings.breaker_cooldown_s)
        # Embeddings get their own breaker: a missing/broken embedding model must not block chat calls.
        self._embed_breaker = CircuitBreaker(settings.breaker_threshold, settings.breaker_cooldown_s)
        self._qcache: dict[str, list[float]] = {}
        self._cache_dir: Path | None = settings.cache_dir if settings.cache_enabled else None
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    async def aclose(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------ transport

    async def _post(self, path: str, payload: dict[str, object], *, name: str) -> dict[str, object]:
        attempt = 0
        breaker = self._embed_breaker if name == "embed" else self._breaker
        while True:
            breaker.before()
            try:
                async with self._sem:
                    r = await self._client.post(path, json=payload)
                if r.status_code in _RETRYABLE_STATUS:
                    raise httpx.HTTPStatusError(f"retryable {r.status_code}", request=r.request, response=r)
                r.raise_for_status()
                data = r.json()
                if not isinstance(data, dict):
                    raise LLMError("non-object response from model server")
                breaker.success()
                return data
            except (httpx.TransportError, httpx.HTTPStatusError, json.JSONDecodeError) as exc:
                retryable = not isinstance(exc, httpx.HTTPStatusError) or (
                    exc.response is not None and exc.response.status_code in _RETRYABLE_STATUS
                )
                breaker.failure()
                if not retryable or attempt >= self.s.llm_max_retries:
                    obs.log("llm.call_failed", call=name, attempt=attempt, error=type(exc).__name__)
                    raise LLMError(f"{name}: model server call failed ({type(exc).__name__})") from exc
                delay = min(8.0, 0.5 * (2**attempt)) * (0.5 + random.random())
                obs.log("llm.retry", call=name, attempt=attempt, delay_s=round(delay, 2), error=type(exc).__name__)
                attempt += 1
                await asyncio.sleep(delay)

    # ------------------------------------------------------------------ cache

    def _cache_key(self, schema: type[BaseModel], system: str, user: str) -> str:
        blob = json.dumps(
            [self.s.backend, self.model, system, user, schema.model_json_schema()], sort_keys=True
        )
        return hashlib.sha256(blob.encode()).hexdigest()

    def _cache_get(self, key: str, schema: type[T]) -> T | None:
        if not self._cache_dir:
            return None
        p = self._cache_dir / f"{key}.json"
        if not p.is_file():
            return None
        try:
            return schema.model_validate_json(p.read_text(encoding="utf-8"))
        except (ValidationError, OSError):
            return None  # stale/corrupt cache is non-critical

    def _cache_put(self, key: str, value: BaseModel) -> None:
        if not self._cache_dir:
            return
        tmp = self._cache_dir / f"{key}.tmp"
        tmp.write_text(value.model_dump_json(), encoding="utf-8")
        tmp.replace(self._cache_dir / f"{key}.json")

    # ------------------------------------------------------------------ public API

    async def complete(self, schema: type[T], system: str, user: str, *, name: str, max_tokens: int | None = None) -> T:
        num_predict = min(self.s.num_predict, max_tokens) if max_tokens else self.s.num_predict
        key = self._cache_key(schema, system, user)
        cached = self._cache_get(key, schema)
        if cached is not None:
            obs.log("llm.call", call=name, model=self.model, cache="hit", tokens_in=0, tokens_out=0, latency_ms=0.0)
            return cached

        est = int((len(system) + len(user) + len(json.dumps(decode_schema(schema)))) / _CHARS_PER_TOKEN)
        if est + num_predict > self.s.num_ctx:
            obs.log("llm.context_overflow", call=name, est_prompt_tokens=est, num_ctx=self.s.num_ctx,
                    num_predict=num_predict)
            raise ContextOverflowError(
                f"{name}: ~{est} prompt tokens + {num_predict} output > num_ctx {self.s.num_ctx}; "
                "raise CGPRR_NUM_CTX or lower CGPRR_TOP_K"
            )
        messages: list[dict[str, str]] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        last_err: str | None = None
        for repair in range(2):  # one repair attempt on schema violation
            if last_err:
                messages = messages[:2] + [
                    {"role": "user", "content": f"Your previous output failed validation: {last_err[:300]}. Return valid JSON only."}
                ]
            t0 = time.perf_counter()
            content, tin, tout = await self._chat(schema, messages, name=name, num_predict=num_predict)
            latency = round((time.perf_counter() - t0) * 1000, 2)
            obs.log("llm.call", call=name, model=self.model, cache="miss", repair=repair,
                    tokens_in=tin, tokens_out=tout, latency_ms=latency,
                    tokens_per_s=round(tout / (latency / 1000), 2) if latency and tout else None)
            if tout >= num_predict:
                obs.log("llm.output_truncated", call=name, tokens_out=tout)  # JSON will fail validation -> repair
            try:
                value = schema.model_validate_json(content)
            except ValidationError as exc:
                last_err = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:5])
                obs.log("llm.schema_violation", call=name, repair=repair, errors=len(exc.errors()))
                continue
            self._cache_put(key, value)
            return value
        raise LLMOutputError(f"{name}: model output failed schema validation twice")

    async def _chat(self, schema: type[BaseModel], messages: list[dict[str, str]], *, name: str, num_predict: int) -> tuple[str, int, int]:
        js = decode_schema(schema)
        if self.s.backend == "ollama":
            data = await self._post(
                "/api/chat",
                {"model": self.model, "messages": messages, "format": js, "stream": False, "keep_alive": self.s.keep_alive,
                 "options": {"temperature": 0, "seed": 7, "num_ctx": self.s.num_ctx,
                             "num_predict": num_predict}},
                name=name,
            )
            msg = data.get("message")
            content = msg.get("content", "") if isinstance(msg, dict) else ""
            return str(content), _as_int(data.get("prompt_eval_count")), _as_int(data.get("eval_count"))
        data = await self._post(
            "/v1/chat/completions",
            {"model": self.model, "messages": messages, "temperature": 0, "seed": 7,
             "max_tokens": num_predict,
             "response_format": {"type": "json_schema", "json_schema": {"name": schema.__name__, "schema": js, "strict": False}}},
            name=name,
        )
        choices = data.get("choices") or []
        content = ""
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            content = str((choices[0].get("message") or {}).get("content", ""))
        usage = data.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        return content, _as_int(usage.get("prompt_tokens")), _as_int(usage.get("completion_tokens"))

    async def embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        """Embeddings. With CGPRR_EMBED_ON_CPU=1, query embeddings run on the CPU (num_gpu=0) so the
        embedding model never competes with the generator for a small GPU; bulk (index) embedding
        still uses the GPU when it is free. Single-query results are memoised in-process."""
        if not texts:
            return []
        if query and len(texts) == 1 and texts[0] in self._qcache:
            return [self._qcache[texts[0]]]
        out: list[list[float]] = []
        for i in range(0, len(texts), 32):  # batch to bound request size
            batch = texts[i : i + 32]
            if self.s.backend == "ollama":
                payload: dict[str, object] = {"model": self.s.embed_model, "input": batch, "keep_alive": self.s.keep_alive}
                if query and self.s.embed_on_cpu:
                    payload["options"] = {"num_gpu": 0}
                data = await self._post("/api/embed", payload, name="embed")
                vecs = data.get("embeddings")
            else:
                data = await self._post("/v1/embeddings", {"model": self.s.embed_model, "input": batch}, name="embed")
                rows = data.get("data")
                vecs = [d.get("embedding") for d in rows if isinstance(d, dict)] if isinstance(rows, list) else None
            if not isinstance(vecs, list) or len(vecs) != len(batch):
                raise LLMError("embedding response shape mismatch")
            out.extend([[float(x) for x in v] for v in vecs])
        if query and len(texts) == 1:
            if len(self._qcache) > 2048:
                self._qcache.clear()
            self._qcache[texts[0]] = out[0]
        return out

    async def chat_text(self, model: str, messages: list[dict[str, str]], *, name: str, max_tokens: int = 32) -> str:
        """Plain-text chat for classifier models (e.g. Llama Guard) that do not emit JSON. Cached."""
        key = hashlib.sha256(json.dumps(["text", model, messages], sort_keys=True).encode()).hexdigest()
        if self._cache_dir and (self._cache_dir / f"{key}.txt").is_file():
            return (self._cache_dir / f"{key}.txt").read_text(encoding="utf-8")
        t0 = time.perf_counter()
        data = await self._post("/api/chat", {"model": model, "messages": messages, "stream": False, "keep_alive": self.s.keep_alive,
                                             "options": {"temperature": 0, "num_predict": max_tokens}}, name=name)
        msg = data.get("message")
        text = str(msg.get("content", "")) if isinstance(msg, dict) else ""
        obs.log("llm.call", call=name, model=model, cache="miss", tokens_in=_as_int(data.get("prompt_eval_count")),
                tokens_out=_as_int(data.get("eval_count")), latency_ms=round((time.perf_counter() - t0) * 1000, 2))
        if self._cache_dir:
            (self._cache_dir / f"{key}.txt").write_text(text, encoding="utf-8")
        return text

    async def warm(self, model: str) -> bool:
        """Load a model into memory now and keep it there (keep_alive), so the first user call is not slow."""
        if self.s.backend != "ollama":
            return True
        try:
            r = await self._client.post("/api/generate", json={"model": model, "keep_alive": self.s.keep_alive}, timeout=self.s.llm_timeout_s)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def unload(self, model: str) -> None:
        """Ask Ollama to free a model's memory now (keep_alive=0). Best-effort."""
        if self.s.backend != "ollama":
            return
        try:
            await self._client.post("/api/generate", json={"model": model, "keep_alive": 0}, timeout=30.0)
        except httpx.HTTPError:
            pass

    async def health(self) -> bool:
        try:
            path = "/api/tags" if self.s.backend == "ollama" else "/v1/models"
            r = await self._client.get(path, timeout=5.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def has_model(self, model: str) -> bool | None:
        """True/False if the Ollama server lists `model`; None when it cannot be checked."""
        if self.s.backend != "ollama":
            return None
        try:
            r = await self._client.get("/api/tags", timeout=5.0)
            models = r.json().get("models", []) if r.status_code == 200 else None
        except (httpx.HTTPError, ValueError):
            return None
        if not isinstance(models, list):
            return None
        want = model if ":" in model else f"{model}:latest"
        names = {str(m.get("name") or m.get("model") or "") for m in models if isinstance(m, dict)}
        return want in names
