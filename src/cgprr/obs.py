"""Observability: trace ids, timed spans, structured JSONL logs, optional OpenTelemetry.

Log policy: no secrets, no prompt/context text, no raw query text unless
CGPRR_LOG_QUERY_TEXT=1. Logged: trace id, span name, latency, token counts, retrieved
passage ids, gate outcomes.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:  # Optional: real distributed tracing when opentelemetry-sdk is installed and configured.
    from opentelemetry import trace as _otel_trace

    _TRACER: Any = _otel_trace.get_tracer("cgprr")
except Exception:  # noqa: BLE001 - OTel is non-critical; degrade to JSON logs only.
    _TRACER = None

_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="")
_lock = threading.Lock()
_log_path: Path | None = None
_echo_stderr = False

_FORBIDDEN_KEYS = {"prompt", "messages", "context", "api_key", "authorization", "query_text_raw"}


def configure(log_path: Path, echo_stderr: bool = False) -> None:
    global _log_path, _echo_stderr
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _log_path = log_path
    _echo_stderr = echo_stderr


def new_trace() -> str:
    tid = uuid.uuid4().hex
    _trace_id.set(tid)
    return tid


def trace_id() -> str:
    tid = _trace_id.get()
    return tid or new_trace()


def text_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def log(event: str, **fields: Any) -> None:
    for k in list(fields):
        if k in _FORBIDDEN_KEYS:
            fields.pop(k)  # defence in depth: never persist prompts/secrets even if passed by mistake
    rec = {
        "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
        "trace_id": trace_id(),
        "event": event,
        **fields,
    }
    line = json.dumps(rec, ensure_ascii=False, default=str)
    if _log_path is not None:
        with _lock, _log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    if _echo_stderr:
        print(line, file=sys.stderr)


@contextlib.contextmanager
def span(name: str, **attrs: Any) -> Iterator[dict[str, Any]]:
    """Timed span. Callers may add fields to the yielded dict; they are logged on exit."""
    extra: dict[str, Any] = {}
    t0 = time.perf_counter()
    otel_cm = _TRACER.start_as_current_span(name) if _TRACER is not None else contextlib.nullcontext()
    with otel_cm as otel_span:
        try:
            yield extra
        except Exception as exc:
            log(name, status="error", error=type(exc).__name__, latency_ms=_ms(t0), **attrs, **extra)
            raise
        else:
            if otel_span is not None and hasattr(otel_span, "set_attribute"):
                for k, v in {**attrs, **extra}.items():
                    if isinstance(v, (str, int, float, bool)):
                        otel_span.set_attribute(f"cgprr.{k}", v)
            log(name, status="ok", latency_ms=_ms(t0), **attrs, **extra)


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 2)
