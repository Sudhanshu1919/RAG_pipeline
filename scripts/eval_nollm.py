"""Score `cgprr ask --no-llm` (zero model calls) on a collection's gold Q&A set, with the same pass rule
as `cgprr eval qa`: answerable -> every must_contain string present; refuse -> refused;
refuse_or_* -> refused, or no must_not_contain string present.
Usage:  python scripts/eval_nollm.py [collection]   (default cgprr-news; build the collection first)
"""

from __future__ import annotations

import asyncio
import json
import sys

from cgprr.config import load_settings
from cgprr.evaluate import _has
from cgprr.index import load
from cgprr.rag import ask
from cgprr.retrieve import Retriever


async def main(name: str) -> int:
    s = load_settings().for_collection(name)
    ret = Retriever(load(s), s, None)
    counts = {"answer": [0, 0], "refuse": [0, 0], "adversarial": [0, 0]}
    for line in (s.eval_dir / "qa_questions.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        q = json.loads(line)
        r = await ask(q["question"], ret, None, use_llm=False)
        exp = q["expected"]
        if exp == "answer":
            key, ok = "answer", r.status == "extracted" and all(_has(r.answer, m) for m in q["must_contain"])
        elif exp == "refuse":
            key, ok = "refuse", r.status == "refused"
        else:
            key = "adversarial"
            ok = r.status == "refused" or not any(_has(r.answer, m) for m in q.get("must_not_contain", []))
        counts[key][0] += ok
        counts[key][1] += 1
        print(f"{q['qid']:6s} {exp:18s} {r.status:9s} {r.refusal_reason or '':24s} {'pass' if ok else 'FAIL'}")
    total = sum(c[0] for c in counts.values()), sum(c[1] for c in counts.values())
    print(f"\npass {total[0]}/{total[1]}  (answerable {counts['answer'][0]}/{counts['answer'][1]}, "
          f"refusal {counts['refuse'][0]}/{counts['refuse'][1]}, adversarial {counts['adversarial'][0]}/{counts['adversarial'][1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "cgprr-news")))
