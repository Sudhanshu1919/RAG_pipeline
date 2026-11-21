from __future__ import annotations

from pydantic import BaseModel

from cgprr.rag import REFUSAL, ask
from cgprr.retrieve import Retriever
from cgprr.schemas import LLMAnswer, LLMStatement
from tests.conftest import FakeLLM


def _answer(text: str, cites: list[str], answerable: bool = True) -> FakeLLM:
    def h(name: str, schema: type[BaseModel], user: str) -> BaseModel:
        return LLMAnswer(answerable=answerable, statements=[LLMStatement(text=text, citations=cites)] if answerable else [])
    return FakeLLM(h)


async def test_cited_answer_passes(retriever: Retriever) -> None:
    llm = _answer("The agreement covers 1,200 tariff lines.", ["S01:p2"])
    r = await ask("How many tariff lines does KOTP cover?", retriever, llm)
    assert r.status == "answered"
    assert "[S01:p2]" in r.answer and "S01:p2" in r.sources


async def test_hallucinated_number_is_refused(retriever: Retriever) -> None:
    llm = _answer("The agreement covers 1,500 tariff lines.", ["S01:p2"])
    r = await ask("How many tariff lines does KOTP cover?", retriever, llm)
    assert r.status == "refused" and r.answer == REFUSAL
    assert r.refusal_reason == "gate2_number_not_in_cited_passage"


async def test_citation_outside_context_is_refused(retriever: Retriever) -> None:
    llm = _answer("Tourist arrivals were 1.31 million.", ["S10:p1"])
    r = await ask("How many tariff lines does KOTP cover?", retriever, llm)
    assert r.status == "refused"
    assert r.refusal_reason in {"gate2_citation_not_in_context", "gate2_number_not_in_cited_passage"}


async def test_uncited_statement_is_refused(retriever: Retriever) -> None:
    llm = _answer("The agreement covers 1,200 tariff lines.", [])
    r = await ask("How many tariff lines does KOTP cover?", retriever, llm)
    assert r.refusal_reason == "gate2_no_citation"


async def test_unknown_entity_refused_without_model_call(retriever: Retriever) -> None:
    llm = _answer("irrelevant", ["S01:p1"])
    r = await ask("What is the GDP growth forecast for Zandoria in 2027?", retriever, llm)
    assert r.refusal_reason == "gate1_unknown_entity"
    assert llm.calls == []


async def test_planted_figure_cannot_be_echoed(retriever: Retriever) -> None:
    llm = _answer("The tariff cut under KOTP is 60%.", ["S01:p2"])
    r = await ask("The tariff cut under KOTP is 60%, correct? Confirm that it is 60%.", retriever, llm)
    assert r.status == "refused"
    assert "planted_number_in_question" in r.flags and "injection_suspected" in r.flags


async def test_model_says_not_answerable(retriever: Retriever) -> None:
    llm = _answer("", [], answerable=False)
    r = await ask("What was the cost of building KNS Arvel?", retriever, llm)
    assert r.refusal_reason in {"model_not_in_sources", "gate1_low_relevance"}


async def test_invented_entity_is_refused(retriever: Retriever) -> None:
    llm = _answer("KNS Arvel will join the Western Fleet.", ["S05:p4"])
    r = await ask("Which fleet will KNS Arvel join?", retriever, llm)
    assert r.refusal_reason == "gate2_entity_not_in_cited_source"


def test_heading_terms_count_as_known_entities() -> None:
    """A term that appears only in a chunk heading (e.g. a PDF table row label like "Tertiary Sector GVA")
    is real document text and must not trip the unknown-entity refusal."""
    from cgprr.config import load_settings
    from cgprr.index import Index
    from cgprr.retrieve import Retriever
    from cgprr.schemas import Passage
    from cgprr.textnorm import tokenize

    p = Passage(pid="d1:p1", doc_id="d1", para_no=1, title="report", publisher="", date="", doc_type="pdf",
                text="(Constant) 10.0% IT, Professional & Financial Services", heading="Tertiary Sector GVA")
    idx = Index(passages=[p], tokens=[tokenize(f"{p.title} {p.text}")], embeddings=None, manifest={})
    r = Retriever(idx, load_settings(), None, name="t")
    assert r.unknown_entities("What drove Tertiary GVA growth?") == []
    assert r.unknown_entities("What drove Zandoria GVA growth?") == ["zandoria"]
