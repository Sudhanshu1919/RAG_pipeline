"""Command-line entry point. Global option: -c/--collection NAME[,NAME...] (default $CGPRR_COLLECTION).

  cgprr health                                 check the local model server
  cgprr collection list | info | status        collections, index manifest, read-pass progress
  cgprr collection create NAME FILE_OR_DIR...  new collection from local files (copied into data/collections/NAME/docs)
  cgprr collection add FILE_OR_DIR...          add files to the selected collection
  cgprr collection files                       files with include switch and state since the last build
  cgprr collection include|exclude FILE...     switch files in or out of the index
  cgprr collection build [--no-read] [--no-calibrate]   ingest -> screen -> chunk -> index -> read pass -> calibrate
  cgprr collection review [--release HASH ...] [--keep HASH ...]   quarantined chunks: list / decide
  cgprr build-index                            alias of `collection build`
  cgprr ask "question" [--json] [--no-llm]     cited answer or refusal (several collections: -c a,b);
                                               --no-llm: zero model calls, verbatim source sentences
  cgprr factcheck --article ID | --all | --file PATH [--consistency]
  cgprr eval {retrieval,qa,claims,e2e,all} [--consistency]
  cgprr eval generate [--questions N] [--claims N]     draft eval set from the collection's own documents
  cgprr eval review [--accept ID ...] [--reject ID ...]  mark generated items reviewed / rejected
  cgprr serve [--port 8765]                    long-running local server (indexes and models stay warm)
  cgprr stats                                  measured tokens, latency, tokens/s from the log

Exit codes: 0 ok, 1 input/config error, 2 model server unreachable (health), 3 model failure on a
critical path, 4 latency target missed in eval.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from cgprr import obs
from cgprr.collections import CollectionError, list_collections, parse_selection
from cgprr.config import Settings, load_settings
from cgprr.llm import LLMError, LocalLLM
from cgprr.retrieve import MultiRetriever, Retriever


def make_retriever(s: Settings, names: list[str], llm: LocalLLM | None) -> Retriever | MultiRetriever:
    from cgprr.index import load

    parts = [Retriever(load(s.for_collection(n)), s.for_collection(n), llm, name=n) for n in names]
    return parts[0] if len(parts) == 1 else MultiRetriever(parts, s)


async def cmd_health(s: Settings) -> int:
    llm = LocalLLM(s)
    try:
        ok = await llm.health()
        print(json.dumps({"server": s.llm_url, "backend": s.backend, "reachable": ok, "model": s.llm_model,
                          "embed_model": s.embed_model, "guard_model": s.guard_model or None}))
        return 0 if ok else 2
    finally:
        await llm.aclose()


# --------------------------------------------------------------------------- collections


def cmd_collection_list(s: Settings) -> int:
    for name in list_collections(s.root):
        mf = s.root / "indexes" / name / "manifest.json"
        if mf.is_file():
            m = json.loads(mf.read_text(encoding="utf-8"))
            rp = (m.get("read_pass") or {}).get("status", "?")
            print(f"{name:24s} docs={m.get('n_docs')} chunks={m.get('n_passages')} dense={m.get('dense')} read_pass={rp}")
        else:
            print(f"{name:24s} (not built)")
    return 0


def cmd_collection_info(s: Settings) -> int:
    c = s.coll()
    mf = s.index_dir / "manifest.json"
    print(json.dumps({"name": c.name, "description": c.description, "root": str(c.root), "sources": list(c.sources),
                      "n_source_files": len(c.source_files()), "require_synthetic": c.require_synthetic,
                      "chunking": c.chunking, "chunk_tokens": c.chunk_tokens,
                      "index": json.loads(mf.read_text(encoding="utf-8")) if mf.is_file() else None}, indent=2))
    return 0


def cmd_collection_status(s: Settings) -> int:
    p = s.index_dir / "read_progress.json"
    mf = s.index_dir / "manifest.json"
    out: dict[str, object] = {"collection": s.collection}
    if mf.is_file():
        out["read_pass"] = json.loads(mf.read_text(encoding="utf-8")).get("read_pass")
    if p.is_file():
        out["progress"] = json.loads(p.read_text(encoding="utf-8"))
    print(json.dumps(out, indent=2))
    return 0


def cmd_collection_create(s: Settings, name: str, paths: list[str], description: str) -> int:
    from cgprr import ops

    coll = ops.create_collection(s.root, name, description)
    res = ops.add_paths(coll, [Path(p) for p in paths])
    _print_add(res)
    print(f"created collection {name!r} at {coll.root}  ({sum(r.ok for r in res)} added, {sum(not r.ok for r in res)} refused)")
    print(f"next: cgprr -c {name} collection build")
    return 0 if any(r.ok for r in res) else 1


def _print_add(results: list[Any]) -> None:
    for r in results:
        if r.ok:
            print(f"  added   {r.name}")
        else:
            print(f"  refused {r.original}: {r.rule} ({r.detail})", file=sys.stderr)


def cmd_collection_add(s: Settings, paths: list[str]) -> int:
    from cgprr import ops

    res = ops.add_paths(s.coll(), [Path(p) for p in paths])
    _print_add(res)
    return 0 if any(r.ok for r in res) else 1


def cmd_collection_files(s: Settings) -> int:
    from cgprr import ops

    for r in ops.file_listing(s):
        print(f"{'+' if r['included'] else '-'} {r['state']:9s} {r['bytes']:>10,d}  {r['file']}  {r['rule']}")
    return 0


def cmd_collection_include(s: Settings, files: list[str], included: bool) -> int:
    from cgprr import ops

    for f in files:
        ops.set_included(s.coll(), f, included)
        print(f"{'included' if included else 'excluded'}: {f}")
    print(f"rebuild to apply: cgprr -c {s.collection} collection build")
    return 0


def _clean_path_arg(p: str) -> str:
    """Drop line breaks and surrounding blanks that a wrapped terminal line puts inside a quoted path."""
    return p.replace("\r", "").replace("\n", "").strip()


async def _missing_models(s: Settings) -> list[str]:
    """Configured models the Ollama server does not list (empty when unknown or all present)."""
    llm = LocalLLM(s)
    try:
        wanted = [s.llm_model] + ([s.embed_model] if s.use_dense else [])
        return [m for m in wanted if await llm.has_model(m) is False]
    finally:
        await llm.aclose()


async def cmd_collection_build(s: Settings, *, read: bool, calibrate_: bool) -> int:
    from cgprr import ops

    llm = LocalLLM(s)
    try:
        need_llm = read or s.use_dense or bool(s.guard_model)
        if need_llm and (read or s.guard_model) and not await llm.health():
            print("ERROR: model server unreachable; build without it using --no-read and CGPRR_USE_DENSE=0", file=sys.stderr)
            return 3
        if read and await llm.has_model(s.llm_model) is False:
            print(f"ERROR: model {s.llm_model!r} is not installed. Run: ollama pull {s.llm_model}\n"
                  "  or build without the read pass: collection build --no-read", file=sys.stderr)
            return 3
        if s.use_dense and await llm.has_model(s.embed_model) is False:
            print(f"WARNING: embedding model {s.embed_model!r} is not installed (ollama pull {s.embed_model}); "
                  "building a BM25-only index. Load profiles\\lite.env or set CGPRR_USE_DENSE=0 to silence this.",
                  file=sys.stderr)
            s = replace(s, use_dense=False)
            need_llm = read or bool(s.guard_model)
        last = {"phase": ""}

        def progress(phase: str, done: int, total: int) -> None:
            if phase != last["phase"] or done == total or done % 10 == 0:
                print(f"  {phase}: {done}/{total}", file=sys.stderr)
                last["phase"] = phase

        manifest = await ops.build_collection(s, llm if need_llm else None, read=read, calibrate=calibrate_,
                                              on_progress=progress)
        print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2))
        if s.use_dense and not manifest.get("dense"):
            print("WARNING: dense embeddings unavailable; index is BM25-only.", file=sys.stderr)
        return 0
    finally:
        await llm.aclose()


def cmd_collection_review(s: Settings, release: list[str], keep: list[str]) -> int:
    from cgprr import ops

    if release or keep:
        print(f"saved {ops.set_review(s, release, keep)}; rebuild to apply: cgprr -c {s.collection} collection build")
        return 0
    r = ops.review_summary(s)
    if r["counts"]:
        print("Ingest decisions:")
        for c in r["counts"]:
            print(f"  {c['action']:18s} {c['rule']:45s} {c['n']}")
    print(f"\nQuarantined chunks ({len(r['quarantined'])}):")
    for q in r["quarantined"]:
        print(f"- [{q['state']}] hash={q.get('hash')} {q.get('pid')} ({q.get('file')})\n    {str(q.get('text'))[:300]}")
    if r["quarantined"]:
        print("\nRelease a false positive:  cgprr -c NAME collection review --release HASH")
    return 0


# --------------------------------------------------------------------------- ask / factcheck


async def cmd_ask(s: Settings, names: list[str], question: str, as_json: bool, no_llm: bool = False) -> int:
    from cgprr.rag import ask

    if no_llm:  # zero model calls: BM25 + gate 1 + verbatim source sentences
        r = await ask(question, make_retriever(s, names, None), None, log_text=s.log_query_text, use_llm=False)
    else:
        llm = LocalLLM(s)
        try:
            r = await ask(question, make_retriever(s, names, llm), llm, log_text=s.log_query_text)
        finally:
            await llm.aclose()
    if as_json:
        print(r.model_dump_json(indent=2))
    else:
        print(r.answer)
        if r.status in ("answered", "extracted"):
            if r.status == "extracted":
                print("(no-LLM mode: verbatim source sentences, not a composed answer)")
            print("\nSources:")
            for pid, txt in r.sources.items():
                print(f"  [{pid}] {txt}")
            if r.labels:
                print(f"content warnings: {', '.join(lb.split(':', 1)[-1] for lb in r.labels)}")
        else:
            print(f"(reason: {r.refusal_reason})")
        if r.flags:
            print(f"flags: {', '.join(r.flags)}")
        print(f"trace: {r.trace_id}  {r.latency_ms:.0f} ms  {r.timings_ms}")
    return 0


async def cmd_factcheck(s: Settings, names: list[str], article: str | None, all_: bool, file: str | None, consistency: bool) -> int:
    from cgprr.factcheck.pipeline import factcheck_article
    from cgprr.factcheck.render import to_html, to_markdown
    from cgprr.ingest import list_articles, load_article, load_article_file

    strict = s.coll().require_synthetic
    if file:
        articles = [load_article_file(Path(file))]
    else:
        ids = list_articles(s.article_dir) if all_ else [article or ""]
        articles = [load_article(s.article_dir, aid, require_synthetic=strict) for aid in ids]
    out_dir = s.root / "out" / "+".join(names)
    out_dir.mkdir(parents=True, exist_ok=True)
    llm = LocalLLM(s)
    try:
        ret = make_retriever(s, names, llm)
        for art in articles:
            rep = await factcheck_article(art, ret, llm, consistency=consistency)
            base = out_dir / f"factcheck_{art.article_id}"
            base.with_suffix(".json").write_text(rep.model_dump_json(indent=2), encoding="utf-8")
            md = to_markdown(rep)
            base.with_suffix(".md").write_text(md, encoding="utf-8")
            base.with_suffix(".html").write_text(to_html(rep), encoding="utf-8")
            print(md)
            print(f"-> {base}.json / .md / .html\n")
    finally:
        await llm.aclose()
    return 0


# --------------------------------------------------------------------------- eval


async def cmd_eval(s: Settings, suite: str, consistency: bool) -> int:
    from cgprr.evaluate import run_and_write

    llm = LocalLLM(s)
    try:
        report, results = await run_and_write(s, llm, suite, consistency=consistency)
    finally:
        await llm.aclose()
    print(report)
    print(f"-> {s.out_dir / f'eval_{suite}.md'}")
    qa = results.get("qa")
    if isinstance(qa, dict) and qa.get("slo_met") is False:
        print("ERROR: latency target missed (CGPRR_SLO_ASK_P95_MS)", file=sys.stderr)
        return 4
    return 0


async def cmd_eval_generate(s: Settings, n_q: int, n_c: int) -> int:
    from cgprr.evalgen import generate
    from cgprr.index import load

    llm = LocalLLM(s)
    try:
        idx = load(s)
        ret = Retriever(idx, s, None)
        stats = await generate(idx, llm, s.eval_dir, n_questions=n_q, n_claims=n_c, vocab=ret.vocab)
    finally:
        await llm.aclose()
    print(json.dumps(stats, indent=2))
    print(f"drafts written to {s.eval_dir}; review: cgprr -c {s.collection} eval review")
    print(f"then rebuild to calibrate thresholds: cgprr -c {s.collection} collection build")
    return 0


def cmd_eval_review(s: Settings, accept: list[str], reject: list[str]) -> int:
    from cgprr.evalgen import load_items, set_status

    if accept or reject:
        n = set_status(s.eval_dir, accept, "reviewed") + set_status(s.eval_dir, reject, "rejected")
        print(f"updated {n} item(s)")
        return 0
    for kind in ("qa", "claims"):
        items = [i for i in load_items(s.eval_dir, kind, split=None) if i.get("status") == "draft"]
        print(f"\n{kind}: {len(items)} draft item(s)")
        for i in items:
            if kind == "qa":
                print(f"  {i['qid']} [{i['split']}] {i['expected']:18s} {i['question']}  must_contain={i.get('must_contain')}")
            else:
                print(f"  {i['claim_id']} [{i['split']}] {i['gold_verdict']:11s} {i['claim']}")
    print("\naccept: cgprr eval review --accept G001 G002 ...   (or --accept all)   reject: --reject G003")
    return 0


def cmd_stats(s: Settings) -> int:
    """Throughput/token figures from the JSONL log (measured, not estimated)."""
    if not s.log_path.is_file():
        print("no log yet", file=sys.stderr)
        return 1
    calls = [json.loads(line) for line in s.log_path.read_text(encoding="utf-8").splitlines() if '"llm.call"' in line]
    miss = [c for c in calls if c.get("cache") == "miss" and c.get("event") == "llm.call"]
    tin = sum(int(c.get("tokens_in", 0)) for c in miss)
    tout = sum(int(c.get("tokens_out", 0)) for c in miss)
    secs = sum(float(c.get("latency_ms", 0)) for c in miss) / 1000
    lat = sorted(float(c.get("latency_ms", 0)) for c in miss)
    p50 = lat[len(lat) // 2] if lat else 0.0
    p95 = lat[min(len(lat) - 1, int(len(lat) * 0.95))] if lat else 0.0
    by_call: dict[str, int] = {}
    for c in miss:
        by_call[str(c.get("call"))] = by_call.get(str(c.get("call")), 0) + 1
    print(json.dumps({"llm_calls": len(calls), "cache_hits": len(calls) - len(miss), "model_calls": len(miss),
                      "tokens_in": tin, "tokens_out": tout, "model_seconds": round(secs, 2),
                      "output_tokens_per_s": round(tout / secs, 2) if secs else None,
                      "latency_ms_p50": p50, "latency_ms_p95": p95, "calls_by_type": by_call}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cgprr", description="CGPRR local RAG and fact-check over document collections")
    ap.add_argument("-v", "--verbose", action="store_true", help="echo JSON logs to stderr")
    ap.add_argument("-c", "--collection", help="collection(s) to use, comma-separated (default: $CGPRR_COLLECTION or cgprr-news)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("health")
    p_bi = sub.add_parser("build-index", help="alias of `collection build`")
    p_bi.add_argument("--no-read", action="store_true")
    p_bi.add_argument("--no-calibrate", action="store_true")
    sub.add_parser("stats", help="token/latency figures from logs")

    p_col = sub.add_parser("collection", help="manage document collections")
    csub = p_col.add_subparsers(dest="action", required=True)
    csub.add_parser("list")
    csub.add_parser("info")
    csub.add_parser("status")
    p_cc = csub.add_parser("create")
    p_cc.add_argument("name")
    p_cc.add_argument("paths", nargs="+")
    p_cc.add_argument("--description", default="")
    p_cb = csub.add_parser("build")
    p_cb.add_argument("--no-read", action="store_true", help="skip the LLM read pass (step A only)")
    p_cb.add_argument("--no-calibrate", action="store_true")
    p_ca = csub.add_parser("add")
    p_ca.add_argument("paths", nargs="+")
    csub.add_parser("files")
    for verb in ("include", "exclude"):
        p_ci = csub.add_parser(verb)
        p_ci.add_argument("files", nargs="+", help="paths relative to the collection folder, e.g. docs/report.pdf")
    p_cr = csub.add_parser("review")
    p_cr.add_argument("--release", nargs="*", default=[])
    p_cr.add_argument("--keep", nargs="*", default=[])

    p_ask = sub.add_parser("ask")
    p_ask.add_argument("question")
    p_ask.add_argument("--json", action="store_true")
    p_ask.add_argument("--no-llm", action="store_true", help="no model call: return the best-matching source sentences verbatim")
    p_fc = sub.add_parser("factcheck")
    g = p_fc.add_mutually_exclusive_group(required=True)
    g.add_argument("--article")
    g.add_argument("--all", action="store_true")
    g.add_argument("--file", help="any supported file (md/txt/pdf/docx/html) to fact-check")
    p_fc.add_argument("--consistency", action="store_true", help="second verdict pass with reversed passages")

    p_ev = sub.add_parser("eval")
    p_ev.add_argument("suite", choices=["retrieval", "qa", "claims", "e2e", "all", "generate", "review"])
    p_ev.add_argument("--consistency", action="store_true")
    p_ev.add_argument("--questions", type=int, default=40)
    p_ev.add_argument("--claims", type=int, default=30)
    p_ev.add_argument("--accept", nargs="*", default=[])
    p_ev.add_argument("--reject", nargs="*", default=[])

    p_sv = sub.add_parser("serve", help="local HTTP server; indexes and models stay loaded")
    p_sv.add_argument("--host", default="127.0.0.1")
    p_sv.add_argument("--port", type=int, default=8765)
    a = ap.parse_args(argv)

    try:
        s = load_settings()
        names = parse_selection(a.collection, s.collection)
        s = s.for_collection(names[0])
        obs.configure(s.log_path, echo_stderr=a.verbose)
        if a.cmd == "health":
            return asyncio.run(cmd_health(s))
        if a.cmd == "stats":
            return cmd_stats(s)
        if a.cmd == "build-index":
            return asyncio.run(cmd_collection_build(s, read=not a.no_read and s.read_pass, calibrate_=not a.no_calibrate))
        if a.cmd == "collection":
            for attr in ("paths", "files"):  # a path pasted/wrapped across lines carries a newline; never valid
                if getattr(a, attr, None):
                    setattr(a, attr, [_clean_path_arg(p) for p in getattr(a, attr)])
            if a.action == "list":
                return cmd_collection_list(s)
            if a.action == "create":
                return cmd_collection_create(s, a.name, a.paths, a.description)
            if a.action == "info":
                return cmd_collection_info(s)
            if a.action == "status":
                return cmd_collection_status(s)
            if a.action == "build":
                return asyncio.run(cmd_collection_build(s, read=not a.no_read and s.read_pass, calibrate_=not a.no_calibrate))
            if a.action == "review":
                return cmd_collection_review(s, a.release, a.keep)
            if a.action == "add":
                return cmd_collection_add(s, a.paths)
            if a.action == "files":
                return cmd_collection_files(s)
            if a.action in ("include", "exclude"):
                return cmd_collection_include(s, a.files, a.action == "include")
        if a.cmd == "ask":
            return asyncio.run(cmd_ask(s, names, a.question, a.json, a.no_llm))
        if a.cmd == "factcheck":
            return asyncio.run(cmd_factcheck(s, names, a.article, a.all, a.file, a.consistency))
        if a.cmd == "eval":
            if a.suite == "generate":
                return asyncio.run(cmd_eval_generate(s, a.questions, a.claims))
            if a.suite == "review":
                return cmd_eval_review(s, a.accept, a.reject)
            return asyncio.run(cmd_eval(s, a.suite, a.consistency))
        if a.cmd == "serve":
            from cgprr.serve import serve

            return asyncio.run(serve(s, names, a.host, a.port))
    except LLMError as exc:
        print(f"ERROR: local model unavailable or invalid output: {exc}", file=sys.stderr)
        for m in asyncio.run(_missing_models(s)):
            print(f"  model {m!r} is not installed: run  ollama pull {m}\n"
                  "  or load a profile that names an installed model, e.g. profiles\\lite.env", file=sys.stderr)
        return 3
    except (OSError, RuntimeError, ValueError, CollectionError) as exc:  # OSError: missing file, permissions
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
