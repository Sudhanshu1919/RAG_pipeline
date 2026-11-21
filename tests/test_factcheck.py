from __future__ import annotations

from pydantic import BaseModel

from cgprr.factcheck.pipeline import factcheck_article
from cgprr.factcheck.render import to_html, to_markdown
from cgprr.factcheck.verdict import check_claim
from cgprr.ingest import load_article
from cgprr.retrieve import Retriever
from cgprr.schemas import LLMClaim, LLMClaims, LLMEvidence, LLMStatement, LLMSummary, LLMVerdict
from tests.conftest import FakeLLM


def _verdict(v: str, pid: str, quote: str) -> FakeLLM:
    def h(name: str, schema: type[BaseModel], user: str) -> BaseModel:
        ev = [LLMEvidence(passage_id=pid, quote=quote)] if pid else []
        return LLMVerdict(verdict=v, evidence=ev, rationale="r", confidence="high")  # type: ignore[arg-type]
    return FakeLLM(h)


async def test_supported_with_valid_quote_is_high(retriever: Retriever) -> None:
    llm = _verdict("supported", "S03:p1", "The Port of Velmora handled 2.4 million TEU in FY2025")
    r = await check_claim("c", "The Port of Velmora handled 2.4 million TEU in FY2025.", "q", retriever, llm)
    assert (r.verdict, r.confidence) == ("supported", "high")


async def test_fabricated_quote_downgrades(retriever: Retriever) -> None:
    llm = _verdict("supported", "S03:p1", "The port handled a record 2.4 million TEU")
    r = await check_claim("c", "The Port of Velmora handled 2.4 million TEU in FY2025.", "q", retriever, llm)
    assert r.verdict == "unverified" and r.confidence == "low"
    assert "no_valid_evidence" in r.flags and "invalid_evidence_quote" in r.flags


async def test_supported_with_wrong_number_downgrades(retriever: Retriever) -> None:
    # Model wrongly says "supported" for 18% growth; evidence quote is real but says 12%.
    llm = _verdict("supported", "S03:p1", "up 12% from 2.14 million TEU in FY2024")
    r = await check_claim("c", "The Port of Velmora throughput grew 18% in FY2025 to 2.4 million TEU.", "q", retriever, llm)
    assert r.verdict == "unverified"
    assert any(f.startswith("number_not_in_evidence") for f in r.flags)


async def test_unsupported_numeric_conflict_is_high(retriever: Retriever) -> None:
    llm = _verdict("unsupported", "S01:p2", "Tariffs on covered lines will be reduced by 25% in phases over five years")
    r = await check_claim("c", "The agreement will cut tariffs by 40% over five years.", "q", retriever, llm)
    assert (r.verdict, r.confidence) == ("unsupported", "high")


async def test_gate1_unverified_without_model_call(retriever: Retriever) -> None:
    llm = _verdict("supported", "S05:p1", "x")
    r = await check_claim("c", "Defence analysts said the vessel cost about USD 120 million.", "q", retriever, llm)
    assert r.verdict == "unverified" and "no_passage_above_threshold" in r.flags
    assert llm.calls == []


async def test_pipeline_end_to_end(settings, retriever: Retriever) -> None:  # type: ignore[no-untyped-def]
    art = load_article(settings.article_dir, "A02")

    def h(name: str, schema: type[BaseModel], user: str) -> BaseModel:
        if name == "extract_claims":
            return LLMClaims(claims=[
                LLMClaim(text="The Port of Velmora handled 2.4 million TEU in FY2025.", article_quote="The Port of Velmora handled 2.4 million TEU in FY2025"),
                LLMClaim(text="Throughput grew 18% on the previous year.", article_quote="a jump of 18% on the previous year"),
                LLMClaim(text="Turnaround fell to 1.9 days.", article_quote="Turnaround time for vessels fell to 1.9 days"),
                LLMClaim(text="Port employs more than 5,000 people.", article_quote="The port employs more than 5,000 people"),
                LLMClaim(text="CDB approved a USD 450 million loan for Berth 7.", article_quote="approved a USD 450 million loan"),
                LLMClaim(text="Invented claim.", article_quote="this text is not in the article"),
            ])
        if name == "verdict":
            if "2.4 million" in user.split("<claim>")[1]:
                return LLMVerdict(verdict="supported", rationale="r", confidence="high",
                                  evidence=[LLMEvidence(passage_id="S03:p1", quote="handled 2.4 million TEU in FY2025")])
            return LLMVerdict(verdict="unverified", evidence=[], rationale="r", confidence="low")
        if name == "summary":
            return LLMSummary(sentences=[
                LLMStatement(text="The Port of Velmora handled 2.4 million TEU in FY2025.", citations=["S03:p1"]),
                LLMStatement(text="The port handled 3 million TEU.", citations=["S03:p1"]),  # must be dropped
            ])
        raise AssertionError(name)

    llm = FakeLLM(h)
    rep = await factcheck_article(art, retriever, llm)
    assert len(rep.claims) == 5
    assert "claims_dropped_invalid_quote:1" in rep.flags
    assert rep.counts["supported"] == 1
    assert [s.text for s in rep.summary] == ["The Port of Velmora handled 2.4 million TEU in FY2025."]
    assert rep.summary_dropped == 1
    assert "could not be verified" in rep.uncertainty_note
    assert "UNVERIFIED" in to_markdown(rep) and "<script" not in to_html(rep)


async def test_wrong_entity_supported_downgrades(retriever: Retriever) -> None:
    llm = _verdict("supported", "S05:p4", "The vessel will be assigned to the Eastern Fleet")
    r = await check_claim("c", "KNS Arvel will join the Western Fleet.", "q", retriever, llm)
    assert r.verdict == "unverified" and any(f.startswith("entity_not_in_evidence") for f in r.flags)


async def test_entity_found_elsewhere_in_cited_document(retriever: Retriever) -> None:
    llm = _verdict("supported", "S07:p5", "It is too early to quantify")
    r = await check_claim("c", "Governor Elling said it was too early to quantify the effect of the Orlan trade agreement.", "q", retriever, llm)
    assert r.verdict == "supported"
