"""Render a FactCheckReport as Markdown and as a self-contained HTML page (all text escaped)."""

from __future__ import annotations

import html

from cgprr.schemas import FactCheckReport

_BADGE = {"supported": "SUPPORTED", "unsupported": "UNSUPPORTED", "unverified": "UNVERIFIED"}
_COLOR = {"supported": "#1b7f3b", "unsupported": "#b42318", "unverified": "#8a6d00"}


def to_markdown(r: FactCheckReport) -> str:
    lines = [f"# Fact-check: {r.article_title} ({r.article_id})", "",
             f"Model: `{r.model}` · trace `{r.trace_id}` · {r.latency_ms:.0f} ms", "",
             "## Summary (verified facts only)", ""]
    if r.summary:
        lines += [f"- {s.text} [{', '.join(s.citations)}]" for s in r.summary]
    else:
        lines.append("_No summary sentence passed verification._")
    lines += ["", "## Claims", "",
              f"Supported {r.counts['supported']} · Unsupported {r.counts['unsupported']} · Unverified {r.counts['unverified']}", ""]
    for c in r.claims:
        lines.append(f"### {_BADGE[c.verdict]} — {c.text}")
        lines.append(f"- Article: “{c.article_quote}”")
        lines.append(f"- Confidence: **{c.confidence}** (model said: {c.model_verdict or '—'}/{c.model_confidence or '—'})")
        for e in c.evidence:
            mark = "" if e.valid else " ⚠ not verbatim — ignored"
            lines.append(f"- Evidence [{e.passage_id}]: “{e.quote}”{mark}")
        lines.append(f"- Rationale: {c.rationale}")
        if c.flags:
            lines.append(f"- Flags: {', '.join(c.flags)}")
        if c.labels:
            lines.append(f"- Content warnings: {', '.join(lb.split(':', 1)[-1] for lb in c.labels)}")
        lines.append("")
    lines += ["## Uncertainty", "", r.uncertainty_note, "", "## Sources", ""]
    lines += [f"- **{pid}** {txt}" for pid, txt in r.sources.items()]
    return "\n".join(lines) + "\n"


def to_html(r: FactCheckReport) -> str:
    e = html.escape
    rows = []
    for c in r.claims:
        ev = "".join(
            f"<li><code>{e(x.passage_id)}</code> “{e(x.quote)}”{'' if x.valid else ' <b>⚠ not verbatim — ignored</b>'}</li>"
            for x in c.evidence
        ) or "<li><i>none</i></li>"
        flags = f"<div class='flags'>flags: {e(', '.join(c.flags))}</div>" if c.flags else ""
        if c.labels:
            flags += f"<div class='flags'>content warning: {e(', '.join(lb.split(':', 1)[-1] for lb in c.labels))}</div>"
        rows.append(
            f"<tr><td><span class='b' style='background:{_COLOR[c.verdict]}'>{_BADGE[c.verdict]}</span>"
            f"<div class='conf'>confidence: {e(c.confidence)}</div></td>"
            f"<td><b>{e(c.text)}</b><div class='q'>article: “{e(c.article_quote)}”</div>{flags}</td>"
            f"<td><ul>{ev}</ul><div class='r'>{e(c.rationale)}</div></td></tr>"
        )
    summ = "".join(f"<li>{e(s.text)} <code>[{e(', '.join(s.citations))}]</code></li>" for s in r.summary) \
        or "<li><i>No summary sentence passed verification.</i></li>"
    srcs = "".join(f"<li><code>{e(k)}</code> {e(v)}</li>" for k, v in r.sources.items())
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Fact-check {e(r.article_id)}</title>
<style>body{{font:15px/1.5 system-ui,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem;color:#111}}
table{{border-collapse:collapse;width:100%}}td{{border-top:1px solid #ddd;padding:.6rem;vertical-align:top}}
.b{{color:#fff;font-weight:700;font-size:12px;padding:2px 6px;border-radius:4px}}
.q,.r,.conf,.flags{{color:#555;font-size:13px}}.flags{{color:#8a6d00}}code{{font-size:12px}}
.note{{background:#fff8e1;border-left:4px solid #8a6d00;padding:.6rem 1rem}}</style></head><body>
<h1>Fact-check: {e(r.article_title)} <small>({e(r.article_id)})</small></h1>
<p>Model <code>{e(r.model)}</code> · trace <code>{e(r.trace_id)}</code> · {r.latency_ms:.0f} ms ·
Supported {r.counts['supported']} · Unsupported {r.counts['unsupported']} · Unverified {r.counts['unverified']}</p>
<h2>Summary (verified facts only)</h2><ul>{summ}</ul>
<h2>Claims</h2><table>{''.join(rows)}</table>
<h2>Uncertainty</h2><p class="note">{e(r.uncertainty_note)}</p>
<h2>Sources</h2><ul>{srcs}</ul></body></html>"""
