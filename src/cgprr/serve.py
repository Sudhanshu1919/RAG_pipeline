"""`cgprr serve`: a small local HTTP server that keeps indexes, BM25 and models warm.

Why: a CLI call pays Python start-up, index load and BM25 rebuild every time, and the model server
may have unloaded the model. The server pays them once. Standard library only (no web framework).

Endpoints (JSON in, JSON out):
  GET  /health                     {"ok": true, "collections": [...], "model": ...}
  POST /ask        {"question": "...", "collections": ["a", "b"]?, "stream": false?, "llm": true?}
  POST /factcheck  {"text": "...", "title": "..."?, "collections": [...]?}

With "stream": true, /ask answers as server-sent events: `retrieved`, `generating`, then `final`
(the gated AskResult). Raw model tokens are deliberately NOT streamed: the answer is only shown
after gate 2 has checked it, so an unverified sentence never reaches the reader.

Security: binds to 127.0.0.1 by default and refuses a non-loopback host unless
CGPRR_ALLOW_REMOTE=1 (there is no authentication); request bodies are capped at 64 KB.
Browser attacks on a localhost server are blocked in code:
- DNS rebinding: the Host header must name a loopback host (localhost, 127.0.0.1, [::1]);
- cross-site requests: a request carrying an Origin header that is not loopback is refused, and a
  POST must be `Content-Type: application/json` (a web page cannot send that cross-site without a
  CORS preflight, which this server never approves).
CGPRR_ALLOW_REMOTE=1 disables the Host/Origin checks (the operator then owns network security).
An index rebuilt while the server runs (e.g. the read pass finishing) is picked up on the next
request (manifest modification time is checked per request).

Edge cases NOT handled: HTTP keep-alive / pipelining (one request per connection), TLS, auth.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import time
from typing import Any

from cgprr import obs
from cgprr.collections import list_collections, parse_selection
from cgprr.config import Settings
from cgprr.index import load
from cgprr.llm import LLMError, LocalLLM
from cgprr.retrieve import MultiRetriever, Retriever
from cgprr.textnorm import clean_text

_MAX_BODY = 64 * 1024
_LOOPBACK_NAMES = {"localhost", "127.0.0.1", "::1"}


def _host_only(value: str) -> str:
    """'localhost:8765' -> 'localhost'; '[::1]:8765' -> '::1'; 'http://a.b:1' -> 'a.b'."""
    v = value.strip().lower()
    if "://" in v:
        v = v.split("://", 1)[1]
    v = v.split("/", 1)[0]
    if v.startswith("["):
        return v[1:].split("]", 1)[0]
    return v.rsplit(":", 1)[0] if v.count(":") == 1 else v


def _is_loopback_name(name: str) -> bool:
    if name in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def request_refusal(method: str, headers: dict[str, str]) -> str | None:
    """Why a request must be refused (DNS rebinding / cross-site), or None. Skipped with CGPRR_ALLOW_REMOTE=1."""
    if os.environ.get("CGPRR_ALLOW_REMOTE") == "1":
        return None
    if not _is_loopback_name(_host_only(headers.get("host", ""))):
        return "Host header must be localhost / 127.0.0.1 (DNS-rebinding protection)"
    origin = headers.get("origin")
    if origin is not None and not _is_loopback_name(_host_only(origin)):
        return "cross-origin request refused"
    if method == "POST" and headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
        return "Content-Type must be application/json"
    return None


class App:
    def __init__(self, s: Settings, default: list[str]) -> None:
        self.s = s
        self.default = default
        self.llm = LocalLLM(s)
        self._ret: dict[str, tuple[float, Retriever]] = {}

    def _one(self, name: str) -> Retriever:
        st = self.s.for_collection(name)
        mtime = (st.index_dir / "manifest.json").stat().st_mtime
        cached = self._ret.get(name)
        if cached is None or cached[0] != mtime:
            self._ret[name] = (mtime, Retriever(load(st), st, self.llm, name=name))
            obs.log("serve.index_loaded", collection=name)
        return self._ret[name][1]

    def retriever(self, names: list[str] | None) -> Retriever | MultiRetriever:
        sel = parse_selection(",".join(names) if names else None, ",".join(self.default))
        known = set(list_collections(self.s.root))
        missing = [n for n in sel if n not in known]
        if missing:
            raise ValueError(f"unknown collection(s): {missing}")
        parts = [self._one(n) for n in sel]
        return parts[0] if len(parts) == 1 else MultiRetriever(parts, self.s)

    async def warm(self) -> None:
        for n in self.default:
            self._one(n)
        ok = await self.llm.warm(self.s.llm_model)
        if self.s.use_dense:
            await self.llm.warm(self.s.embed_model)
        obs.log("serve.warm", model_loaded=ok)


async def _read_request(reader: asyncio.StreamReader) -> tuple[str, str, dict[str, str], bytes]:
    head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=10)
    lines = head.decode("latin-1").split("\r\n")
    method, path, _ = lines[0].split(" ", 2)
    headers = {k.strip().lower(): v.strip() for k, v in (ln.split(":", 1) for ln in lines[1:] if ":" in ln)}
    n = int(headers.get("content-length", "0") or 0)
    if n > _MAX_BODY:
        raise ValueError("body too large")
    body = await asyncio.wait_for(reader.readexactly(n), timeout=10) if n else b""
    return method, path.split("?", 1)[0], headers, body


def _response(status: int, payload: Any, ctype: str = "application/json") -> bytes:
    body = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode()
    reason = {200: "OK", 400: "Bad Request", 403: "Forbidden", 404: "Not Found", 413: "Payload Too Large", 500: "Internal Server Error",
              503: "Service Unavailable"}.get(status, "OK")
    return (f"HTTP/1.1 {status} {reason}\r\nContent-Type: {ctype}; charset=utf-8\r\nContent-Length: {len(body)}\r\n"
            f"Connection: close\r\n\r\n").encode() + body


async def handle(app: App, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    from cgprr.factcheck.pipeline import factcheck_article
    from cgprr.ingest import Article
    from cgprr.rag import ask

    t0 = time.perf_counter()
    try:
        try:
            method, path, headers, body = await _read_request(reader)
        except ValueError:
            writer.write(_response(413, {"error": "request too large"}))
            return
        refusal = request_refusal(method, headers)
        if refusal:
            obs.log("serve.refused", reason=refusal)
            writer.write(_response(403, {"error": refusal}))
            return
        if method == "GET" and path == "/health":
            writer.write(_response(200, {"ok": True, "collections": list_collections(app.s.root),
                                         "default": app.default, "model": app.s.llm_model}))
            return
        if method != "POST" or path not in ("/ask", "/factcheck"):
            writer.write(_response(404, {"error": "not found"}))
            return
        req = json.loads(body or b"{}")
        if not isinstance(req, dict):
            raise ValueError("JSON object expected")  # noqa: TRY004 - mapped to HTTP 400
        ret = app.retriever(req.get("collections"))
        if path == "/ask":
            question = str(req.get("question", ""))
            if req.get("llm") is False:  # {"llm": false}: no model call, verbatim source sentences
                r = await ask(question, ret, None, use_llm=False)
                writer.write(_response(200, json.loads(r.model_dump_json())))
                return
            if req.get("stream"):
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n")

                async def emit(event: str, data: dict[str, object]) -> None:
                    writer.write(f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode())
                    await writer.drain()

                r = await ask(question, ret, app.llm, on_event=emit)
                await emit("final", json.loads(r.model_dump_json()))
                return
            r = await ask(question, ret, app.llm)
            writer.write(_response(200, json.loads(r.model_dump_json())))
            return
        text = clean_text(str(req.get("text", ""))).strip()
        if not text or len(text) > 60_000:
            raise ValueError("'text' must be 1..60000 characters")
        art = Article(article_id="request", title=str(req.get("title", ""))[:300], outlet="", date="",
                      paragraphs=[p.strip() for p in text.split("\n\n") if p.strip()])
        rep = await factcheck_article(art, ret, app.llm, consistency=bool(req.get("consistency", False)))
        writer.write(_response(200, json.loads(rep.model_dump_json())))
    except (ValueError, KeyError) as exc:  # json.JSONDecodeError is a ValueError
        writer.write(_response(400, {"error": str(exc)[:300]}))
    except LLMError as exc:
        writer.write(_response(503, {"error": f"local model unavailable: {type(exc).__name__}"}))
    except Exception as exc:  # noqa: BLE001 - never crash the server on one request
        obs.log("serve.error", error=type(exc).__name__)
        writer.write(_response(500, {"error": "internal error"}))
    finally:
        obs.log("serve.request", latency_ms=round((time.perf_counter() - t0) * 1000, 2))
        try:
            await writer.drain()
            writer.close()
        except ConnectionError:
            pass


async def serve(s: Settings, names: list[str], host: str, port: int) -> int:
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host == "localhost"
    if not loopback and os.environ.get("CGPRR_ALLOW_REMOTE") != "1":
        raise ValueError(f"refusing to bind to non-loopback host {host!r} (no authentication); set CGPRR_ALLOW_REMOTE=1 to override")
    app = App(s, names)
    await app.warm()
    server = await asyncio.start_server(lambda r, w: handle(app, r, w), host, port)
    print(f"cgprr serve: http://{host}:{port}  collections={names}  model={s.llm_model}", flush=True)
    async with server:
        try:
            await server.serve_forever()
        finally:
            await app.llm.aclose()
    return 0
