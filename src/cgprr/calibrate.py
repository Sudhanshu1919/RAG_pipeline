"""Gate-1 thresholds chosen from the collection's own data instead of hard-coded constants.

Signals per question (exactly what gate 1 uses at answer time): the best lexical coverage and the
best dense cosine among the retrieved hits. A question passes gate 1 if coverage >= c OR dense >= d.

Positives: calibration-split questions that should reach the model — answerable QA items and
supported/unsupported claims (their gold passage exists).
Negatives: calibration-split refusal questions, the shipped off-topic list, and (when other
collections exist) a sample of their answerable questions. Negatives that the unknown-entity check
already refuses are excluded: they never reach the thresholds, so they must not push them up.

Choice: maximise Youden's J = TPR - FPR subject to TPR >= 0.95 (let at least 95% of answerable
questions through; gate 2 and the model handle the rest), clamped to safe ranges. Among equally good
thresholds take the maximum-margin one (middle of the gap between positives and negatives), and with
fewer than 20 positives never go stricter than the defaults. With fewer than 5 positives or 3 negatives the defaults are kept and the manifest says so.
Test-split items are never used here.

Edge cases NOT handled: per-document thresholds; drift after a collection grows (re-run build).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from cgprr import obs
from cgprr.evalgen import OFF_TOPIC, load_items
from cgprr.retrieve import Retriever

MIN_TPR = 0.95
SMALL_SAMPLE = 20  # below this many positives, calibration may loosen the defaults but not tighten them
_COV_RANGE = (0.2, 0.9)
_DENSE_RANGE = (0.3, 0.95)


async def _signals(retriever: Retriever, question: str) -> tuple[float, float | None] | None:
    if retriever.unknown_entities(question):
        return None  # refused before thresholds apply
    hits, _ = await retriever.search(question)
    if not hits:
        return 0.0, None
    cov = max(h.coverage for h in hits)
    dens = [h.dense for h in hits if h.dense is not None]
    return cov, (max(dens) if dens else None)


def _rate(sig: Sequence[tuple[float, float | None]], c: float, d: float | None) -> float:
    if not sig:
        return 0.0
    return sum(1 for cov, dv in sig if cov >= c or (d is not None and dv is not None and dv >= d)) / len(sig)


def _below(values: list[float], x: float) -> float | None:
    lower = [v for v in values if v < x]
    return max(lower) if lower else None


def choose(pos: Sequence[tuple[float, float | None]], neg: Sequence[tuple[float, float | None]], defaults: tuple[float, float]
           ) -> dict[str, object]:
    covs = sorted({round(c, 4) for c, _ in [*pos, *neg]} | {defaults[0]})
    dens = sorted({round(d, 4) for _, d in [*pos, *neg] if d is not None} | {defaults[1]})
    use_dense = any(d is not None for _, d in pos)
    grid = [(c, d) for c in covs if _COV_RANGE[0] <= c <= _COV_RANGE[1]
            for d in (dens if use_dense else [None]) if d is None or _DENSE_RANGE[0] <= d <= _DENSE_RANGE[1]]
    scored = [(_rate(pos, c, d) - _rate(neg, c, d), c, d) for c, d in grid if _rate(pos, c, d) >= MIN_TPR]
    if not scored:
        return {"min_coverage": defaults[0], "min_dense": defaults[1], "source": "default:no_threshold_meets_tpr"}
    best_j = max(j for j, _, _ in scored)
    optimal = [(c, d) for j, c, d in scored if j == best_j]
    # Every threshold in the J-optimal set separates the sample equally well. Picking its strictest
    # corner puts the threshold on the lowest positive seen, which overfits a small sample (measured:
    # 7 calibration positives gave coverage 0.875 and refused half the test questions). Take the
    # middle of the gap instead: halfway between the strictest optimal value and the highest sample
    # point below the most lenient optimal value (maximum margin), if that keeps J.
    def margin(values: list[float], strict: float, lenient: float) -> float:
        lo = _below(values, lenient)
        return lenient if lo is None else round((lo + strict) / 2, 4)
    c_s, c_l = max(c for c, _ in optimal), min(c for c, _ in optimal)
    c = margin(covs, c_s, c_l)
    d: float | None = None
    if use_dense:
        ds = [x for _, x in optimal if x is not None]
        d = margin(dens, max(ds), min(ds))
    if _rate(pos, c, d) < MIN_TPR or _rate(pos, c, d) - _rate(neg, c, d) < best_j:
        c, d = c_l, min((x for cc, x in optimal if cc == c_l and x is not None), default=None)
    source = "calibrated"
    if len(pos) < SMALL_SAMPLE:
        # Too few positives to justify a stricter gate than the defaults: only loosening is allowed.
        c, d = min(c, defaults[0]), (min(d, defaults[1]) if d is not None else None)
        source = f"calibrated:capped_at_defaults(n_pos<{SMALL_SAMPLE})"
    dd = d if use_dense else None
    return {"min_coverage": c, "min_dense": d if d is not None else defaults[1],
            "tpr": round(_rate(pos, c, dd), 3), "fpr": round(_rate(neg, c, dd), 3), "source": source}


async def calibrate(retriever: Retriever, eval_dir: Path, *, others: list[str] | None = None) -> dict[str, object]:
    defaults = (retriever.s.min_coverage, retriever.s.min_dense)
    qa = load_items(eval_dir, "qa", split="calibration")
    claims = load_items(eval_dir, "claims", split="calibration")
    pos_q = [str(q["question"]) for q in qa if q["expected"] == "answer"]
    pos_q += [str(c["claim"]) for c in claims if c["gold_verdict"] in ("supported", "unsupported")]
    neg_q = [str(q["question"]) for q in qa if q["expected"] == "refuse"] + OFF_TOPIC + list(others or [])

    pos = [s for s in [await _signals(retriever, q) for q in pos_q] if s is not None]
    neg = [s for s in [await _signals(retriever, q) for q in neg_q] if s is not None]
    if len(pos) < 5 or len(neg) < 3:
        out: dict[str, object] = {"min_coverage": defaults[0], "min_dense": defaults[1],
                                  "source": f"default:too_few_items(pos={len(pos)},neg={len(neg)})"}
    else:
        out = choose(pos, neg, defaults)
    out.update({"n_pos": len(pos), "n_neg": len(neg),
                "pos_coverage": sorted(round(c, 3) for c, _ in pos), "neg_coverage": sorted(round(c, 3) for c, _ in neg)})
    obs.log("calibrate.done", **{k: v for k, v in out.items() if not isinstance(v, list)})
    return out
