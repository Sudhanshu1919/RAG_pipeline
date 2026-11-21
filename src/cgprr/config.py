"""Runtime configuration from environment variables. No secrets are read or stored."""

from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from cgprr.collections import Collection, load_collection, validate_name

Backend = Literal["ollama", "openai_compat"]


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw not in (None, "") else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _backend() -> Backend:
    raw = _env("CGPRR_BACKEND", "ollama")
    if raw == "openai_compat":
        return "openai_compat"
    if raw == "ollama":
        return "ollama"
    raise ValueError(f"CGPRR_BACKEND must be 'ollama' or 'openai_compat', got {raw!r}")


@dataclass(frozen=True)
class Settings:
    # Paths. One collection = one source set, one index dir, one log file, one cache (brief §2.5).
    # CGPRR_BUSINESS is accepted as a legacy alias for CGPRR_COLLECTION.
    root: Path = field(default_factory=lambda: Path(_env("CGPRR_ROOT", ".")).resolve())
    collection: str = field(default_factory=lambda: _env("CGPRR_COLLECTION", _env("CGPRR_BUSINESS", "cgprr-news")))

    # LLM serving. Main path is local (Ollama or a local OpenAI-compatible server such as
    # vLLM / llama.cpp). No paid cloud API is used anywhere in this codebase.
    backend: Backend = field(default_factory=lambda: _backend())
    llm_url: str = field(default_factory=lambda: _env("CGPRR_LLM_URL", "http://127.0.0.1:11434"))
    llm_model: str = field(default_factory=lambda: _env("CGPRR_LLM_MODEL", "qwen2.5:7b-instruct-q4_K_M"))
    embed_model: str = field(default_factory=lambda: _env("CGPRR_EMBED_MODEL", "bge-m3"))
    use_dense: bool = field(default_factory=lambda: _env_bool("CGPRR_USE_DENSE", True))
    num_ctx: int = field(default_factory=lambda: _env_int("CGPRR_NUM_CTX", 8192))
    # Hard cap on generated tokens per call: bounds latency and KV growth, stops runaway output.
    num_predict: int = field(default_factory=lambda: _env_int("CGPRR_NUM_PREDICT", 768))
    llm_timeout_s: float = field(default_factory=lambda: _env_float("CGPRR_LLM_TIMEOUT_S", 120.0))
    llm_max_retries: int = field(default_factory=lambda: _env_int("CGPRR_LLM_MAX_RETRIES", 3))
    llm_concurrency: int = field(default_factory=lambda: _env_int("CGPRR_LLM_CONCURRENCY", 2))
    breaker_threshold: int = field(default_factory=lambda: _env_int("CGPRR_BREAKER_THRESHOLD", 5))
    breaker_cooldown_s: float = field(default_factory=lambda: _env_float("CGPRR_BREAKER_COOLDOWN_S", 30.0))
    cache_enabled: bool = field(default_factory=lambda: _env_bool("CGPRR_CACHE", True))
    keep_alive: str = field(default_factory=lambda: _env("CGPRR_KEEP_ALIVE", "30m"))  # keep models warm (latency)
    embed_on_cpu: bool = field(default_factory=lambda: _env_bool("CGPRR_EMBED_ON_CPU", False))
    qa_max_tokens: int = field(default_factory=lambda: _env_int("CGPRR_QA_MAX_TOKENS", 384))
    # Prompt budget for Q&A passages (estimated tokens; 0 = no limit). Prefill dominates latency when
    # layers sit on the CPU: the top-ranked passages are kept, in order, until the budget is spent.
    qa_passage_tokens: int = field(default_factory=lambda: _env_int("CGPRR_QA_PASSAGE_TOKENS", 0))
    read_pass: bool = field(default_factory=lambda: _env_bool("CGPRR_READ_PASS", True))
    guard_model: str = field(default_factory=lambda: _env("CGPRR_GUARD_MODEL", ""))  # e.g. llama-guard3:1b; empty = off

    # Retrieval and gate thresholds. Calibrated on the shipped eval set; disclose in README.
    top_k: int = field(default_factory=lambda: _env_int("CGPRR_TOP_K", 5))
    min_coverage: float = field(default_factory=lambda: _env_float("CGPRR_MIN_COVERAGE", 0.5))
    min_dense: float = field(default_factory=lambda: _env_float("CGPRR_MIN_DENSE", 0.60))

    # Logging. Query text is hashed by default (zero-trust: queries may contain PII).
    log_query_text: bool = field(default_factory=lambda: _env_bool("CGPRR_LOG_QUERY_TEXT", False))

    def coll(self, name: str | None = None) -> Collection:
        return load_collection(self.root, name or self.collection)

    def for_collection(self, name: str) -> Settings:
        return replace(self, collection=name)

    @property
    def article_dir(self) -> Path:
        return self.coll().article_dir

    @property
    def eval_dir(self) -> Path:
        return self.coll().eval_dir

    @property
    def index_dir(self) -> Path:
        return self.root / "indexes" / self.collection

    @property
    def log_path(self) -> Path:
        return self.root / "logs" / f"{self.collection}.jsonl"

    @property
    def cache_dir(self) -> Path:
        return self.root / ".cache" / self.collection / "llm"  # no store is shared across collections

    @property
    def out_dir(self) -> Path:
        return self.root / "out" / self.collection


def is_local_url(url: str) -> bool:
    """True for a model server on this machine or the local network: loopback / private IPs,
    `localhost`, or a single-label host name (a Docker service such as `ollama`)."""
    from urllib.parse import urlsplit

    host = (urlsplit(url).hostname or "").lower()
    if not host:
        return False
    if host == "localhost" or "." not in host:
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return host.endswith((".local", ".internal", ".lan"))
    return ip.is_loopback or ip.is_private or ip.is_link_local


def load_settings() -> Settings:
    s = Settings()
    validate_name(s.collection)
    if s.backend not in ("ollama", "openai_compat"):
        raise ValueError(f"CGPRR_BACKEND must be 'ollama' or 'openai_compat', got {s.backend!r}")
    if not 0.0 <= s.min_coverage <= 1.0:
        raise ValueError("CGPRR_MIN_COVERAGE must be in [0, 1]")
    if s.num_predict < 64 or s.num_predict >= s.num_ctx:
        raise ValueError("CGPRR_NUM_PREDICT must be in [64, CGPRR_NUM_CTX)")
    if s.top_k < 1 or s.top_k > 50:
        raise ValueError("CGPRR_TOP_K must be in [1, 50]")
    # Every prompt carries document text: a typo'd or hostile URL must not send it off the machine.
    if not is_local_url(s.llm_url) and os.environ.get("CGPRR_ALLOW_REMOTE_LLM") != "1":
        raise ValueError(f"CGPRR_LLM_URL {s.llm_url!r} is not a local address; documents would leave this machine. "
                         "Set CGPRR_ALLOW_REMOTE_LLM=1 only if that server is yours.")
    return s
