"""Evaluation. Expected behaviour is fixed in eval files before any run; nothing here tunes tests
to pass. Items come from the collection's eval folder: hand-written files (always split "test",
status "reviewed") plus auto-generated drafts (evalgen.py; only their "test" split is scored —
the "calibration" split is reserved for threshold calibration). Scores are reported for all
test items and, separately, for reviewed items only.

Suites:
- retrieval: deterministic, no LLM. Gold evidence hit@k and gate-1 routing; with a read-pass
             index, also the same metrics WITHOUT the read-pass keys (ablation).
- qa:        answer / refuse / adversarial, plus answer latency p50/p95 per stage.
- claims:    verdict classifier on gold claims (isolates verification from extraction).
- e2e:       full pipeline per article: extraction coverage vs gold (one-to-one matching),
             verdict agreement on matched claims, summary sentences kept/dropped.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any

from cgprr.evalgen import load_items
from cgprr.factcheck.pipeline import factcheck_article
from cgprr.factcheck.verdict import check_claim
from cgprr.ingest import list_articles, load_article
from cgprr.llm import LLM
from cgprr.rag import ask
from cgprr.retrieve import Retriever
from cgprr.textnorm import content_terms, coverage, normalise, ungrounded_numbers

VERDICTS = ("supported", "unsupported", "unverified")


def _has(text: str, needle: str) -> bool:
    return normalise(needle) in normalise(text)


def _key_match(text: str, key: str) -> bool:
    """Lenient answer check, reported separately from the strict one: every number in the key appears
    in the answer (by value) and >= 60% of the key's content terms do. Auto-generated keys are verbatim
    spans ("warmup_steps = 4000") that a correct paraphrase rarely contains."""
    return _has(text, key) or (not ungrounded_numbers(key, [text]) and coverage(key, content_terms(text)) >= 0.6)


def _pct(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[min(len(s) - 1, round(q * (len(s) - 1)))]


# --------------------------------------------------------------------------- retrieval (no LLM)


async def _retrieval_pass(retriever: Retriever, qs: list[dict[str, Any]], gold: list[dict[str, Any]],
                          use_questions: bool) -> dict[str, Any]:
    qa_hit = qa_n = 0
    rows: list[str] = []
    for q in qs:
        hits, _ = await retriever.search(q["question"], use_questions=use_questions)
        ids = [h.passage.pid for h in hits]
        gate1_refuse = bool(retriever.unknown_entities(q["question"])) or not retriever.relevant(hits)
        hit = bool(set(ids) & set(q["gold_evidence"])) if q["gold_evidence"] else None
        if q["expected"] == "answer":
            qa_n += 1
            qa_hit += bool(hit)
        rows.append(f"| {q['qid']} | {q['category']} | {q['expected']} | {'—' if hit is None else ('yes' if hit else 'NO')} "
                    f"| {'refuse' if gate1_refuse else 'pass'} | {max((h.coverage for h in hits), default=0):.2f} |")
    cl_hit = cl_n = 0
    routing: Counter[tuple[str, str]] = Counter()
    for g in gold:
        hits, _ = await retriever.search(g["claim"], use_questions=use_questions)
        routed = "unverified(gate1)" if not retriever.relevant(hits) else "to_model"
        routing[(g["gold_verdict"], routed)] += 1
        if g["evidence"]:
            cl_n += 1
            cl_hit += bool({h.passage.pid for h in hits} & set(g["evidence"]))
    return {"qa_hit": qa_hit, "qa_n": qa_n, "claim_hit": cl_hit, "claim_n": cl_n, "rows": rows, "routing": routing}


async def eval_retrieval(retriever: Retriever, eval_dir: Path) -> tuple[str, dict[str, Any]]:
    qs = load_items(eval_dir, "qa")
    gold = load_items(eval_dir, "claims")
    main = await _retrieval_pass(retriever, qs, gold, True)
    k = retriever.s.top_k
    has_read = retriever.idx.aug_tokens is not None or retriever.idx.q_embeddings is not None
    md = [(f"### Retrieval (deterministic, no LLM) — top_k={k}, dense={'on' if retriever.idx.embeddings is not None else 'off'}, "
           f"read-pass keys={'on' if has_read else 'none'}, thresholds: coverage>={retriever.min_coverage:g}, dense>={retriever.min_dense:g}"), "",
          f"- QA answerable: gold passage in top-{k}: **{main['qa_hit']}/{main['qa_n']}**",
          f"- Gold claims with evidence: gold passage in top-{k}: **{main['claim_hit']}/{main['claim_n']}**"]
    res: dict[str, Any] = {"qa_hit": main["qa_hit"], "qa_n": main["qa_n"], "claim_hit": main["claim_hit"], "claim_n": main["claim_n"],
                           "gate1_routing": {f"{a}|{b}": n for (a, b), n in main["routing"].items()}}
    if has_read:
        abl = await _retrieval_pass(retriever, qs, gold, False)
        md.append(f"- Same, **without** read-pass keys (ablation): QA {abl['qa_hit']}/{abl['qa_n']}, "
                  f"claims {abl['claim_hit']}/{abl['claim_n']}")
        res["ablation_no_read_pass"] = {"qa_hit": abl["qa_hit"], "claim_hit": abl["claim_hit"],
                                        "gate1_routing": {f"{a}|{b}": n for (a, b), n in abl["routing"].items()}}
    md += ["", "| qid | category | expected | gold in top-k | gate 1 | best coverage |", "|---|---|---|---|---|---|", *main["rows"], "",
           "Gate-1 routing of gold claims:", "", "| gold verdict | routed | n |", "|---|---|---|"]
    md += [f"| {gv} | {rt} | {n} |" for (gv, rt), n in sorted(main["routing"].items())]
    return "\n".join(md), res


# --------------------------------------------------------------------------- core QA


async def eval_qa(retriever: Retriever, llm: LLM, eval_dir: Path) -> tuple[str, dict[str, Any]]:
    qs = load_items(eval_dir, "qa")
    rows: list[str] = []
    agg: Counter[str] = Counter()
    reviewed: Counter[str] = Counter()
    lat: dict[str, list[float]] = {"total": [], "retrieve": [], "generate": []}
    for q in qs:
        r = await ask(q["question"], retriever, llm)
        exp = q["expected"]
        cited = {c for s in r.statements for c in s.citations}
        cite_ok: bool | None = None
        if exp == "answer":
            agg["answerable"] += 1
            ok = r.status == "answered" and all(_has(r.answer, m) for m in q["must_contain"])
            cite_ok = r.status == "answered" and bool(cited & set(q["gold_evidence"]))
            agg["answer_correct"] += ok
            agg["answer_lenient"] += r.status == "answered" and all(_key_match(r.answer, m) for m in q["must_contain"])
            agg["citation_correct"] += bool(cite_ok)
            agg["false_refusal"] += r.status == "refused"
        elif exp == "refuse":
            agg["refusal_cases"] += 1
            ok = r.status == "refused"
            agg["refusal_correct"] += ok
        else:  # refuse_or_correct / refuse_or_limit
            agg["adversarial"] += 1
            ok = r.status == "refused" or not any(_has(r.answer, m) for m in q.get("must_not_contain", []))
            agg["adversarial_correct"] += ok
        agg["pass"] += ok
        if q.get("status") == "reviewed":
            reviewed["n"] += 1
            reviewed["pass"] += ok
        lat["total"].append(r.latency_ms)
        for stage in ("retrieve", "generate"):
            if stage in r.timings_ms:
                lat[stage].append(r.timings_ms[stage])
        rows.append(f"| {q['qid']} | {q['category']} | {exp} | {r.status} | {r.refusal_reason or ''} | "
                    f"{'PASS' if ok else 'FAIL'} | {'—' if cite_ok is None else ('yes' if cite_ok else 'no')} | {r.latency_ms:.0f} |")
    lat_stats = {k: {"p50": round(_pct(v, 0.5), 1), "p95": round(_pct(v, 0.95), 1), "n": len(v)} for k, v in lat.items()}
    slo = float(os.environ.get("CGPRR_SLO_ASK_P95_MS", "0") or 0)
    slo_ok = slo <= 0 or lat_stats["total"]["p95"] <= slo
    md = ["### Core RAG Q&A", "",
          "| metric | score |", "|---|---|",
          f"| answer correct (answerable) | {agg['answer_correct']}/{agg['answerable']} |",
          f"| answer correct, lenient key match (numbers + 60% of key terms; not in overall pass) | {agg['answer_lenient']}/{agg['answerable']} |",
          f"| citation correct (answerable) | {agg['citation_correct']}/{agg['answerable']} |",
          f"| false refusals (answerable) | {agg['false_refusal']}/{agg['answerable']} |",
          f"| refusal correct (not in sources) | {agg['refusal_correct']}/{agg['refusal_cases']} |",
          f"| adversarial handled (planted / speculation) | {agg['adversarial_correct']}/{agg['adversarial']} |",
          f"| **overall pass** | **{agg['pass']}/{len(qs)}** |",
          f"| overall pass, reviewed items only | {reviewed['pass']}/{reviewed['n']} |", "",
          "Latency (ms; refusals at gate 1 make no model call):", "",
          "| stage | p50 | p95 | n |", "|---|---|---|---|",
          *[f"| {k} | {v['p50']:.0f} | {v['p95']:.0f} | {v['n']} |" for k, v in lat_stats.items()],
          *( ["", f"**Latency target p95 <= {slo:.0f} ms: {'met' if slo_ok else 'MISSED'}**"] if slo > 0 else []), "",
          "| qid | category | expected | status | reason | result | cite ok | ms |", "|---|---|---|---|---|---|---|---|", *rows]
    return "\n".join(md), {**dict(agg), "reviewed_pass": reviewed["pass"], "reviewed_n": reviewed["n"],
                           "latency": lat_stats, "slo_p95_ms": slo or None, "slo_met": slo_ok}


# --------------------------------------------------------------------------- verdicts on gold claims


def _prf(conf: dict[tuple[str, str], int]) -> list[str]:
    out = []
    for v in VERDICTS:
        tp = conf.get((v, v), 0)
        fp = sum(conf.get((g, v), 0) for g in VERDICTS if g != v)
        fn = sum(conf.get((v, p), 0) for p in VERDICTS if p != v)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        out.append(f"| {v} | {p:.2f} | {r:.2f} | {f:.2f} | {tp + fn} |")
    return out


async def eval_claims(retriever: Retriever, llm: LLM, eval_dir: Path, *, consistency: bool = False) -> tuple[str, dict[str, Any]]:
    gold = load_items(eval_dir, "claims")
    conf: Counter[tuple[str, str]] = Counter()
    model_conf: Counter[tuple[str, str]] = Counter()
    rows: list[str] = []
    downgrades = 0
    for g in gold:
        r = await check_claim(g["claim_id"], g["claim"], g["article_quote"], retriever, llm, consistency=consistency)
        conf[(g["gold_verdict"], r.verdict)] += 1
        if r.model_verdict:
            model_conf[(g["gold_verdict"], r.model_verdict)] += 1
        downgrades += r.confidence == "low"
        ok = r.verdict == g["gold_verdict"]
        rows.append(f"| {g['claim_id']} | {g['tag']} | {g['gold_verdict']} | {r.verdict} | {r.model_verdict or 'gate1'} | "
                    f"{r.confidence} | {'ok' if ok else 'MISS'} | {', '.join(r.flags)} |")
    n = len(gold) or 1
    acc = sum(conf[(v, v)] for v in VERDICTS) / n
    m_n = sum(model_conf.values())
    m_acc = sum(model_conf[(v, v)] for v in VERDICTS) / m_n if m_n else 0.0
    # Safety metric: a claim that is NOT supported but is reported as supported is the costly error.
    false_support = sum(conf[(g, "supported")] for g in VERDICTS if g != "supported")
    md = [f"### Claim verdicts on {len(gold)} gold claims (consistency={'on' if consistency else 'off'})", "",
          f"- Final accuracy (after code gates): **{acc:.2f}**",
          f"- Raw model accuracy on claims that reached the model: {m_acc:.2f} ({m_n} claims)",
          f"- **False 'supported' (unsupported/unverified reported as supported): {false_support}**",
          f"- Code downgrades (low confidence): {downgrades}", "",
          "| class | precision | recall | F1 | gold n |", "|---|---|---|---|---|", *_prf(dict(conf)), "",
          "Confusion (rows = gold, cols = predicted):", "",
          "| gold \\ pred | " + " | ".join(VERDICTS) + " |", "|---|---|---|---|"]
    md += [f"| {g} | " + " | ".join(str(conf[(g, p)]) for p in VERDICTS) + " |" for g in VERDICTS]
    md += ["", "| claim | tag | gold | final | model | conf | result | flags |", "|---|---|---|---|---|---|---|---|", *rows]
    return "\n".join(md), {"accuracy": acc, "model_accuracy": m_acc, "false_supported": false_support,
                           "downgrades": downgrades, "n": len(gold), "confusion": {f"{a}|{b}": c for (a, b), c in conf.items()}}


# --------------------------------------------------------------------------- end-to-end


def _similarity(quote: str, gold_quote: str) -> float:
    a, b = normalise(quote), normalise(gold_quote)
    if a in b or b in a:
        return 1.0
    ta, tb = content_terms(quote), content_terms(gold_quote)
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


def match_claims(extracted: list[str], gold: list[str], threshold: float = 0.5) -> dict[int, int]:
    """One-to-one greedy matching (best similarity first): gold index -> extracted index."""
    pairs = sorted(((_similarity(e, g), gi, ei) for gi, g in enumerate(gold) for ei, e in enumerate(extracted)), reverse=True)
    used_g: set[int] = set()
    used_e: set[int] = set()
    out: dict[int, int] = {}
    for sim, gi, ei in pairs:
        if sim < threshold:
            break
        if gi in used_g or ei in used_e:
            continue
        out[gi] = ei
        used_g.add(gi)
        used_e.add(ei)
    return out


async def eval_e2e(retriever: Retriever, llm: LLM, root: Path, eval_dir: Path, *, consistency: bool = False) -> tuple[str, dict[str, Any]]:
    gold = [g for g in load_items(eval_dir, "claims") if g.get("source") == "hand"]
    article_dir = retriever.s.article_dir
    strict = retriever.s.coll().require_synthetic
    rows: list[str] = []
    tot_gold = tot_found = agree = extracted = summ_kept = summ_drop = 0
    for aid in list_articles(article_dir):
        rep = await factcheck_article(load_article(article_dir, aid, require_synthetic=strict), retriever, llm, consistency=consistency)
        g_a = [g for g in gold if g["article_id"] == aid]
        m = match_claims([c.article_quote for c in rep.claims], [g["article_quote"] for g in g_a])
        found = len(m)
        agree += sum(rep.claims[ei].verdict == g_a[gi]["gold_verdict"] for gi, ei in m.items())
        tot_gold += len(g_a)
        tot_found += found
        extracted += len(rep.claims)
        summ_kept += len(rep.summary)
        summ_drop += rep.summary_dropped
        rows.append(f"| {aid} | {len(rep.claims)} | {found}/{len(g_a)} | {rep.counts['supported']}/{rep.counts['unsupported']}/{rep.counts['unverified']} "
                    f"| {len(rep.summary)} | {rep.summary_dropped} | {', '.join(rep.flags)} |")
    md = ["### End-to-end fact-check", "",
          f"- Gold claims recovered by extraction (one-to-one): **{tot_found}/{tot_gold}** (extracted total: {extracted})",
          f"- Verdict agreement on recovered claims: **{agree}/{tot_found}**",
          f"- Summary sentences kept / dropped by gate 2: {summ_kept} / {summ_drop}", "",
          "| article | claims | gold recovered | S/U/V | summary kept | summary dropped | flags |", "|---|---|---|---|---|---|---|", *rows]
    return "\n".join(md), {"gold_recovered": tot_found, "gold_total": tot_gold, "verdict_agree": agree,
                           "summary_kept": summ_kept, "summary_dropped": summ_drop}


async def run_and_write(s: Any, llm: LLM, suite: str = "all", *, consistency: bool = False) -> tuple[str, dict[str, Any]]:
    """Run one suite (or all) on the selected collection and write out/<name>/eval_<suite>.{md,json}.
    Shared by `cgprr eval` and the web UI. The caller owns (and closes) `llm`."""
    import json

    from cgprr.index import load

    eval_dir = s.eval_dir
    s.out_dir.mkdir(parents=True, exist_ok=True)
    ret = Retriever(load(s), s, llm, name=s.collection)
    sections: list[str] = []
    results: dict[str, Any] = {}
    if suite in ("retrieval", "all"):
        md, res = await eval_retrieval(ret, eval_dir)
        sections.append(md)
        results["retrieval"] = res
    if suite in ("qa", "all"):
        md, res = await eval_qa(ret, llm, eval_dir)
        sections.append(md)
        results["qa"] = res
    if suite in ("claims", "all"):
        md, res = await eval_claims(ret, llm, eval_dir, consistency=consistency)
        sections.append(md)
        results["claims"] = res
    if suite in ("e2e", "all"):
        md, res = await eval_e2e(ret, llm, s.root, eval_dir, consistency=consistency)
        sections.append(md)
        results["e2e"] = res
    rp = ret.idx.manifest.get("read_pass") or {}
    th = ret.idx.manifest.get("thresholds") or {}
    header = (f"# Evaluation — collection `{s.collection}`, model `{s.llm_model}`, "
              f"embed `{s.embed_model if ret.idx.embeddings is not None else 'off'}`, "
              f"read pass `{rp.get('status', 'off') if isinstance(rp, dict) else 'off'}`, "
              f"backend `{s.backend}`, top_k={s.top_k}, gate-1 thresholds coverage>={ret.min_coverage:g} / "
              f"dense>={ret.min_dense:g} ({th.get('source', 'default') if isinstance(th, dict) else 'default'})\n")
    report = header + "\n\n".join(sections) + "\n"
    (s.out_dir / f"eval_{suite}.md").write_text(report, encoding="utf-8")
    (s.out_dir / f"eval_{suite}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return report, results
