"""Phases 2-5 with a fake LLM: read pass, multi-collection retrieval, calibration, eval generation,
classifier mapping, one-to-one e2e matching, and the local server. These test CODE PATHS, not
model quality."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from pydantic import BaseModel

from cgprr.calibrate import calibrate, choose
from cgprr.classify import apply_verdicts, parse_guard
from cgprr.config import Settings
from cgprr.evalgen import LLMEvalQuestion, LLMUnverifiedClaim, generate, load_items, perturb_number, set_status, split_for
from cgprr.evaluate import match_claims
from cgprr.index import build, load
from cgprr.ingest import IngestReport, load_sources_with_report
from cgprr.rag import ask
from cgprr.reader import LLMChunkReading, LLMDocReading
from cgprr.retrieve import MultiRetriever, Retriever
from cgprr.schemas import LLMAnswer, LLMStatement
from cgprr.security import chunk_hash
from tests.conftest import FakeLLM

PORT = ("# Throughput\nThe Port of Velmora handled 2.4 million TEU in FY2025, up 12% from 2.14 million TEU.\n\n"
        "# Workforce\nThe port employs 3,100 staff across four terminals.\n")
HEALTH = ("# Vaccination\nThe district vaccinated 81,000 children in 2025 under the Orvel programme.\n\n"
          "# Clinics\nNine new clinics opened in Tarsk province.\n")


def _make(root: Path, name: str, files: dict[str, str]) -> Settings:
    d = root / "collections" / name / "docs"
    d.mkdir(parents=True)
    (d.parent / "collection.toml").write_text(f'name = "{name}"\nsources = ["docs/*"]\nchunk_tokens = 64\n', encoding="utf-8")
    for k, v in files.items():
        (d / k).write_text(v, encoding="utf-8")
    return Settings(root=root, collection=name, use_dense=False, cache_enabled=False)


def _reader_llm(extra: dict[str, list[str]] | None = None) -> FakeLLM:
    def h(name: str, schema: type[BaseModel], user: str) -> BaseModel:
        if name == "read_doc":
            return LLMDocReading(summary="A port authority annual report.", entities=["Port of Velmora"])
        if name == "read_chunk":
            if "employs" in user:
                # one grounded question, one with an invented figure (must be dropped), one invented entity
                return LLMChunkReading(context="Staffing at the port.", keywords=["headcount", "workforce"],
                                       questions=["How large is the port's headcount?", "Did the port hire 9,999 people?",
                                                  "Is the port run by Zandor Holdings?"])
            return LLMChunkReading(context="Container volumes.", questions=["What was container throughput?"], keywords=["volume"])
        raise AssertionError(name)
    return FakeLLM(h)


async def test_read_pass_augments_retrieval_but_not_vocabulary(tmp_path: Path) -> None:
    s = _make(tmp_path, "port", {"report.md": PORT})
    idx = await build(s, _reader_llm(), read=True)
    assert idx.manifest["read_pass"]["status"] == "done"  # type: ignore[index]
    assert (s.index_dir / "reading.jsonl").is_file()
    staff = next(p for p in idx.passages if "employs" in p.text)
    r = idx.readings[staff.pid]
    assert "Did the port hire 9,999 people?" not in r["questions"] and r["dropped_questions"] == 1  # type: ignore[operator]
    ret = Retriever(load(s), s, None)
    hits, _ = await ret.search("What is the headcount?")
    assert hits[0].passage.pid == staff.pid                      # found via a read-pass keyword/question
    raw_hits, _ = await ret.search("What is the headcount?", use_questions=False)
    assert raw_hits[0].bm25 == 0.0                               # ablation: raw text has no such term
    assert "zandor" not in ret.vocab                              # read-pass text never enters the vocabulary
    assert ret.unknown_entities("Who owns Zandor Holdings?") == ["hold", "zandor"]


async def test_gate2_rejects_entity_only_present_in_reading(tmp_path: Path) -> None:
    s = _make(tmp_path, "port", {"report.md": PORT})
    await build(s, _reader_llm(), read=True)
    ret = Retriever(load(s), s, None)

    def h(name: str, schema: type[BaseModel], user: str) -> BaseModel:
        return LLMAnswer(answerable=True, statements=[LLMStatement(text="The port is run by Zandor Holdings with 3,100 staff.", citations=["P1"])])
    r = await ask("How many staff does the port employ across its terminals?", ret, FakeLLM(h))
    assert r.status == "refused" and r.refusal_reason == "gate2_entity_not_in_cited_source"


async def test_two_collections_are_isolated_unless_combined(tmp_path: Path) -> None:
    sp = _make(tmp_path, "port", {"report.md": PORT})
    sh = _make(tmp_path, "health", {"health.md": HEALTH})
    await build(sp, None)
    await build(sh, None)
    port = Retriever(load(sp), sp, None, name="port")
    health = Retriever(load(sh), sh, None, name="health")
    q = "How many children did the district vaccinate in 2025 under the Orvel programme?"

    def h(name: str, schema: type[BaseModel], user: str) -> BaseModel:
        return LLMAnswer(answerable=True, statements=[LLMStatement(text="The district vaccinated 81,000 children in 2025.", citations=["P1"])])
    alone = await ask(q, port, FakeLLM(h))
    assert alone.status == "refused"  # the port collection knows nothing about Orvel
    both = MultiRetriever([port, health], sp)
    r = await ask(q, both, FakeLLM(h))
    assert r.status == "answered" and all(pid.startswith("health/") for pid in r.sources)
    assert any(h_.passage.pid.startswith("health/") for h_ in (await both.search(q))[0])


def test_threshold_choice_meets_tpr_and_separates() -> None:
    pos = [(0.8, None), (0.7, None), (0.6, None), (0.55, None), (0.9, None), (0.75, None)]
    neg = [(0.1, None), (0.2, None), (0.0, None), (0.3, None)]
    t = choose(pos, neg, (0.5, 0.6))
    assert str(t["source"]).startswith("calibrated") and 0.3 < float(t["min_coverage"]) <= 0.55  # type: ignore[arg-type]
    assert t["tpr"] == 1.0 and t["fpr"] == 0.0
    # Many positives just above 0.875, negatives at most 0.5: the maximum-margin threshold sits in
    # the gap (not on the lowest positive, which is what overfitted the public-sample run).
    many = [(0.875 + 0.005 * i, None) for i in range(25)]
    t = choose(many, [(0.5, None), (0.33, None)], (0.5, 0.6))
    assert t["source"] == "calibrated" and 0.6 < float(t["min_coverage"]) < 0.8  # type: ignore[arg-type]
    # Few positives: never stricter than the defaults.
    t = choose(many[:7], [(0.5, None), (0.33, None)], (0.5, 0.6))
    assert float(t["min_coverage"]) <= 0.5 and "capped" in str(t["source"])  # type: ignore[arg-type]
    assert choose([(0.1, None)] * 6, [(0.9, None)] * 4, (0.5, 0.6))["source"].startswith("default")  # type: ignore[union-attr]


def test_perturb_and_split_are_deterministic() -> None:
    assert perturb_number("25%") == "40%" and perturb_number("1,200") == "1,920"
    assert perturb_number("2.4 million") == "3.8 million" and perturb_number("2025") == "2027"
    assert split_for("G001abc") == split_for("G001abc")


async def test_eval_generation_review_and_calibration(tmp_path: Path) -> None:
    s = _make(tmp_path, "port", {"report.md": PORT, "staff.md": "# Staff\nThe Velmora port employs 3,100 staff in four terminals.\n"})
    idx = await build(s, None)

    def h(name: str, schema: type[BaseModel], user: str) -> BaseModel:
        if name == "evalgen_question":
            if "2.4 million" in user:
                return LLMEvalQuestion(question="How much container traffic did Velmora move in FY2025?", answer_key="2.4 million TEU")
            return LLMEvalQuestion(question="What is the size of the Velmora port workforce?", answer_key="3,100 staff")
        if name == "evalgen_unverified":
            return LLMUnverifiedClaim(claim="The port plans to open a fifth terminal next year.")
        raise AssertionError(name)
    stats = await generate(idx, FakeLLM(h), s.eval_dir, n_questions=4, n_claims=6)
    assert stats["qa_answer"] >= 2 and stats["qa_refuse"] >= 1
    qa = load_items(s.eval_dir, "qa", split=None)
    assert all(q["status"] == "draft" and q["split"] in ("calibration", "test") for q in qa)
    assert any(q["category"] == "auto_entity_swap" and q["expected"] == "refuse" for q in qa)
    planted = [q for q in qa if q["category"] == "auto_planted_figure"]
    assert planted and all(q["must_not_contain"] for q in planted)
    claims = load_items(s.eval_dir, "claims", split=None)
    unsup = [c for c in claims if c["gold_verdict"] == "unsupported"]
    assert unsup and all("TEU" in c["claim"] or "staff" in c["claim"] for c in unsup)
    assert set_status(s.eval_dir, [str(qa[0]["qid"])], "reviewed") == 1
    assert load_items(s.eval_dir, "qa", split=None)[0]["status"] == "reviewed"
    th = await calibrate(Retriever(idx, s, None), s.eval_dir)
    assert "min_coverage" in th and th["n_neg"] >= 3  # type: ignore[operator]


def test_guard_output_mapping_and_application(tmp_path: Path) -> None:
    assert parse_guard("safe") == [] and parse_guard("unsafe\nS1,S10") == ["violence", "hate"]
    s = _make(tmp_path, "c", {"a.md": "The blast killed two people.\n\nSecond file text here."})
    docs, rep = load_sources_with_report(s.coll())
    v = {chunk_hash(docs[0].paragraphs[0]): ["violence"]}
    kept = apply_verdicts(docs, v, rep)
    assert kept and "content_warning:violence" in kept[0].labels["1"]
    v2 = {chunk_hash(docs[0].paragraphs[0]): ["child_sexual_exploitation"]}
    assert apply_verdicts(docs, v2, IngestReport()) == []  # whole file dropped


def test_e2e_matching_is_one_to_one() -> None:
    extracted = ["The agreement covers 1,200 tariff lines and will cut tariffs by 40% over five years"]
    gold = ["The agreement covers 1,200 tariff lines", "will cut tariffs by 40% over five years"]
    assert len(match_claims(extracted, gold)) == 1


async def test_local_server_health_and_ask(tmp_path: Path) -> None:
    from cgprr import serve as srv

    s = _make(tmp_path, "port", {"report.md": PORT})
    await build(s, None)
    app = srv.App(s, ["port"])
    def h(name: str, schema: type[BaseModel], user: str) -> BaseModel:
        return LLMAnswer(answerable=True, statements=[LLMStatement(text="The port employs 3,100 staff.", citations=["P1"])])
    app.llm = FakeLLM(h)  # type: ignore[assignment]
    server = await asyncio.start_server(lambda r, w: srv.handle(app, r, w), "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]

    async def call(raw: bytes) -> tuple[str, str]:
        r, w = await asyncio.open_connection("127.0.0.1", port)
        w.write(raw)
        await w.drain()
        data = (await r.read()).decode()
        w.close()
        head, _, body = data.partition("\r\n\r\n")
        return head.split("\r\n")[0], body

    hdr = b"Host: 127.0.0.1:8765\r\nContent-Type: application/json\r\n"

    def post(payload: bytes, extra: bytes = hdr) -> bytes:
        return b"POST /ask HTTP/1.1\r\n" + extra + b"Content-Length: " + str(len(payload)).encode() + b"\r\n\r\n" + payload

    async with server:
        status, body = await call(b"GET /health HTTP/1.1\r\nHost: localhost:8765\r\n\r\n")
        assert "200" in status and json.loads(body)["ok"] is True
        payload = json.dumps({"question": "How many staff does the port employ?"}).encode()
        status, body = await call(post(payload))
        assert "200" in status and json.loads(body)["status"] == "answered"
        status, body = await call(post(json.dumps({"question": "How many staff does the port employ?", "llm": False}).encode()))
        assert "200" in status and json.loads(body)["status"] == "extracted"
        payload_s = json.dumps({"question": "x", "stream": True}).encode()
        status, body = await call(post(payload_s))
        assert "event: final" in body
        # Browser-borne attacks on a localhost server: DNS rebinding, cross-site form/fetch posts.
        status, _ = await call(post(payload, b"Host: evil.example:8765\r\nContent-Type: application/json\r\n"))
        assert "403" in status
        status, _ = await call(post(payload, hdr + b"Origin: http://evil.example\r\n"))
        assert "403" in status
        status, _ = await call(post(payload, b"Host: localhost:8765\r\nContent-Type: text/plain\r\n"))
        assert "403" in status
        status, _ = await call(post(payload, hdr + b"Origin: http://localhost:3000\r\n"))
        assert "200" in status
        status, _ = await call(b"POST /ask HTTP/1.1\r\nContent-Length: 999999\r\n\r\n")
        assert "413" in status
    try:
        await srv.serve(s, ["port"], "0.0.0.0", 0)
        raise AssertionError("must refuse non-loopback bind")
    except ValueError:
        pass


def test_passage_budget_keeps_rank_order_and_top_passage() -> None:
    from cgprr.rag import budget_passages
    from cgprr.schemas import Passage
    ps = [Passage(pid=f"d:p{i}", doc_id="d", para_no=i, title="t", publisher="p", date="", doc_type="report",
                  text="word " * 200) for i in range(1, 6)]  # ~312 tokens each
    assert budget_passages(ps, 0) == ps
    assert [p.pid for p in budget_passages(ps, 700)] == ["d:p1", "d:p2"]
    assert [p.pid for p in budget_passages(ps, 10)] == ["d:p1"]  # the top passage is always shown
