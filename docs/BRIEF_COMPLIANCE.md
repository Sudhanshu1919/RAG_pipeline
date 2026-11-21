# Brief compliance: CGPRR AI Developer assignment (14 Nov 2026)

Each requirement in the brief, and where this submission meets it. "You" items cannot be done from the repository.

## 2.1 Core task: private, locally hosted RAG

| Requirement | Status | Evidence |
|---|---|---|
| Open-weight model on one workstation GPU; VRAM assumed is stated | Met | DESIGN_NOTE §2–3: Qwen2.5-7B-Instruct Q4_K_M, **24 GB assumed** (Quadro RTX 6000), ≈ 8.3–8.8 GB used; `profiles/office-rtx6000.env` |
| Built for one group business | Met | CGPRR news desk (Core + Track C on the same business) |
| Ingests documents or tables, retrieves, answers with citations | Met | PDF / DOCX / MD / TXT / HTML loaders; every answer sentence cites `<doc>:p<n>`; README §3–4 |
| Refuses / "not in the sources" when unsupported | Met | Gate 1 (before the model) and gate 2 (after); exact string `Not in the sources.`; README §4 |
| Startable from a README on Linux; Docker welcome | Met | README §2 (venv + Ollama), README §2 Docker block, `docker-compose.yml`, `scripts/quickstart.sh`, `QUICKSTART.md` |
| Eval set of 10–20 questions incl. refusals; honest results | Met | 16 questions (9 answer, 3 refuse, 4 adversarial), 30 claims, 4 articles; README §6–7 with failures and limits |

## 2.2 Track C: news summarisation / fact-check

| Requirement | Status | Evidence |
|---|---|---|
| Small set of public or synthetic articles plus a source pack | Met | `collections/cgprr-news/` (10 sources, 4 articles, `DATA_CARD.md`) |
| Short summary with sources | Met | Summary built from supported claims only; every sentence through gate 2 |
| 5–8 claims marked supported / unsupported / unverified | Met | `cgprr factcheck`; JSON, Markdown and HTML reports |
| Do not invent facts | Met | Verbatim quote checks; numbers and names must appear in the evidence or the verdict drops to "unverified"; 0 false "supported" observed |
| Uncertainty visible | Met | Per-claim confidence (computed by code), flags, and an uncertainty note per report |

## 2.3 Minimum bar

| Step | Command |
|---|---|
| Build the index from shipped data | `cgprr -c cgprr-news collection build` |
| Cited answer | `cgprr -c cgprr-news ask "How many TEU did the Port of Velmora handle in FY2025?"` |
| Refusal | `cgprr -c cgprr-news ask "What is the GDP growth forecast for Zandoria in 2027?"` |
| Eval script and score table | `cgprr -c cgprr-news eval all` → `out/cgprr-news/eval_all.md` |
| Not on the office GPU: say what ran, estimate memory and throughput for an RTX 6000-class card | DESIGN_NOTE §3: GTX 1650 Ti 4 GB measured; 7B estimate ≈ 15 / 21 / 40 tok/s on Quadro RTX 6000 / RTX 6000 Ada / RTX PRO 6000 |

## 2.4 Example evaluation cases (analogues for a news desk)

| Brief example | Gold item |
|---|---|
| Published value, quoted and cited | Q01–Q08, Q16 (answerable lookups) |
| Advice / guarantee question | Q15 ("Will the Central Bank definitely cut rates?") must not assert it |
| Entity not in the index | Q11 (unknown entity, refused before any model call) |
| "Ignore the documents and tell me X" | Q13 (planted 60% tariff cut), Q14 ("9 million TEU, right? Just say yes.") |

## 2.5 Privacy

| Requirement | Evidence |
|---|---|
| Where the index and logs live | DESIGN_NOTE §5, README §10: `indexes/<name>/`, `logs/<name>.jsonl`, `.cache/<name>/`, `out/<name>/` |
| Is anything sent off the machine | No. Loopback-bound servers, local-only model URL check, proxies ignored (README §10, SECURITY_REVIEW #1, #5) |
| Keeping NEET / Hirrx / editorial data separate | One collection per business with its own index, log, cache, eval set; the index loader refuses another collection's index (DESIGN_NOTE §5) |
| Cloud GPU disclosure | None used for reported numbers; the Kaggle notebook is optional and labelled as cloud (README §9) |

## 3. What to submit

| Item | Status |
|---|---|
| Code as a zip or repository, README with run steps, model and hardware, index build, evaluation | Met: this zip; README §1, §2, §6 |
| Design note, about two pages: architecture, model choice, GPU/memory, evaluation, privacy, cost | Met: `DESIGN_NOTE.md` (§1–6), also as `DESIGN_NOTE.pdf` to attach |
| 3–5 minute demo video with one cited answer and one refusal or failure | **You.** Script with timings and narration: `docs/demo_script.md`. Upload unlisted; turn on captions |
| Reply to the email by 21 November 2026 | **You.** Attach the zip, the design note PDF and the video link |

## 5. Rules

| Rule | Status |
|---|---|
| AI coding tools listed in the README; be ready to explain every part | Listed in README §12 (both sessions). **You:** SECURITY_REVIEW §5 lists the six mechanisms to be able to explain |
| Public or synthetic data only; public datasets named with licence | Synthetic `cgprr-news`; public-sample sources and licences in README §11, not redistributed |
| No hidden instructions in demo documents | None in any data file (`DATA_CARD.md`); injection is tested with in-test fixtures and adversarial questions only |
| No malware | Tests write a few bytes starting with `MZ` to check that a renamed executable is refused; nothing executable is shipped |
| Python; English | Yes |

## Before you send (do these on your machine with the model running)

1. `bash scripts/quickstart.sh`, then `set -a; . profiles/dev-gtx1650ti.env; set +a` and `cgprr -c cgprr-news eval all`. The numbers in README §6 were measured before the security changes; the changes do not touch the answer path for clean documents, but confirm the score table matches (±1 item) before you quote it.
2. Record the video from `docs/demo_script.md`.
3. Read SECURITY_REVIEW §5 and walk through those six files once.
