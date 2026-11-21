# CGPRR local RAG + fact-check over document collections (Core task + Track C)

A private, locally hosted assistant for CGPRR's editorial desk.

- **Collections:** point it at any set of documents (PDF, Word .docx, Markdown/TXT, HTML), as one or several named collections, and choose per question which collections to search.
- **Read once at ingest:** every file is parsed, screened, split along its headings, and read once by the local model, which writes a context line and likely questions per chunk. Retrieval uses them; answers never do.
- **Core Q&A:** cited answers, or exactly "Not in the sources." when the selected collections do not support an answer.
- **Track C:** takes an article (any supported file), extracts 5–8 claims, marks each **supported / unsupported / unverified** with verbatim evidence, and writes a short summary that cites its sources. Uncertainty is shown per claim and per report.
- **Main path:** open-weight models on a local GPU, served by Ollama. No cloud LLM API is used anywhere.

The rule throughout: **the model proposes; code decides.** Every refusal, downgrade, quote check and security decision happens in code that the model's output cannot override.

**New here?** Read [QUICKSTART.md](QUICKSTART.md) first (setup, adding your own documents, troubleshooting), or run `bash scripts/quickstart.sh`. The security review and fixes are in [SECURITY_REVIEW.md](SECURITY_REVIEW.md). For the fewest model calls, load `profiles/lite.env` (build: 0 calls; `ask --no-llm`: 0 calls).

**Minimum bar in five commands** (after the setup in §2; on the laptop, use `profiles/dev-gtx1650ti.env`):
```bash
cgprr -c cgprr-news collection build                                                   # 1. build the index from the shipped data
cgprr -c cgprr-news ask "How many TEU did the Port of Velmora handle in FY2025?"        # 2. cited answer
cgprr -c cgprr-news ask "What is the GDP growth forecast for Zandoria in 2027?"         # 3. refusal: "Not in the sources."
cgprr -c cgprr-news eval all                                                           # 4. score table -> out/cgprr-news/eval_all.md
cgprr -c cgprr-news factcheck --article A01                                            # Track C report -> out/cgprr-news/
```
The two-page design note is [DESIGN_NOTE.md](DESIGN_NOTE.md), with the long version in [docs/DESIGN_DETAILS.md](docs/DESIGN_DETAILS.md). Measured results are in §6.

## 1. Hardware and models

**Development hardware (measured 2026-11-16 / 2026-11-19):**

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce GTX 1650 Ti (laptop), 4,096 MiB VRAM, 50 W cap, Turing sm_75 |
| VRAM in use at idle | ~1,050–1,100 MiB: the GPU also drives the Windows display (WDDM), so ~3.0 GiB is usable |
| CPU / RAM | Intel Core i5-10300H; 5.7 GiB RAM visible to WSL2 |
| OS | Windows 11 + WSL2 Ubuntu 26.04.1 (Python 3.14); also run natively on Windows (Python 3.13) |
| Ollama | 0.40.16 |
| Cloud GPU | none used for the numbers in this README (a Kaggle T4 notebook is provided, §9) |

| Profile (`profiles/*.env`) | Generator | Embeddings / guard | Used for |
|---|---|---|---|
| `dev-gtx1650ti` | Qwen3-4B-Instruct-2507 Q4_K_M (~2.5 GB, Apache-2.0); ~86% of layers on the GPU | bge-m3 (MIT) for queries **on the CPU**; llama-guard3:1b at build time | all measured runs |
| `target-7b-offload` | Qwen2.5-7B-Instruct Q4_K_M, GPU+CPU split | off | office model, accuracy only (slow) |
| `kaggle-t4-7b` | Qwen2.5-7B-Instruct Q4_K_M, fully on a 16 GB T4 | bge-m3 + llama-guard3:1b on the GPU | Kaggle notebook |
| `office-rtx6000` | Qwen2.5-7B-Instruct, 8,192 ctx, 4 parallel | bge-m3 on the GPU | office deployment (not run here) |

**Known Ollama issue on this card (Windows build):** the default CUDA 13 runner crashes loading Qwen3 (`CUDA error: shared object initialization failed`, in `ggml_cuda_kernel_can_use_pdl`). Start the server with `OLLAMA_LLM_LIBRARY=cuda_v12`. The Linux build (WSL) does not crash. The Kaggle notebook detects the crash and falls back automatically.

## 2. Setup (WSL2 Ubuntu or any Linux; Windows also works)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"                     # includes the PDF / Word / HTML loaders (pure Python)
pytest -q                                   # 125 tests, no GPU or model server needed

curl -fsSL https://ollama.com/install.sh | sh
# Settings for a 4 GB card (where `ollama serve` runs):
#   OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=2
ollama pull qwen3:4b-instruct-2507-q4_K_M
ollama pull bge-m3              # dense retrieval
ollama pull llama-guard3:1b     # optional build-time content classifier

set -a; . profiles/dev-gtx1650ti.env; set +a
cgprr health
```
`OLLAMA_MAX_LOADED_MODELS=2` keeps the generator on the GPU and the query embedder on the CPU. With 1, every query would evict the generator.

**Docker (office GPU host; needs the NVIDIA Container Toolkit).** Two containers: Ollama with the GPU, and the app. Both are reachable only from this machine. This path targets the office card; the measured runs in §6 did not use Docker.
```bash
mkdir -p indexes logs out .cache data                 # host folders the app writes to
docker compose up -d ollama
docker compose exec ollama ollama pull qwen2.5:7b-instruct-q4_K_M
docker compose exec ollama ollama pull bge-m3
docker compose run --rm app collection build          # = cgprr collection build (default collection cgprr-news)
docker compose run --rm app ask "How many tariff lines does the Kestria–Orlan Trade Partnership Agreement cover?"
docker compose run --rm app ask "What is the GDP growth forecast for Zandoria in 2027?"
docker compose run --rm app eval all                  # score table -> out/cgprr-news/eval_all.md
```

## 3. Collections

A collection is a folder with a `collection.toml`. It is the unit of data separation: its own index, log, cache, eval set and output folder. Nothing is shared between collections unless a question explicitly selects several.

Collections are found in two places:
- `collections/`: shipped samples (`cgprr-news`). Versioned and read-only.
- `data/collections/`: your own documents. This folder is git-ignored and never versioned; set `CGPRR_DATA_DIR` to put it elsewhere.

**Your own documents, end to end:**
```bash
cgprr collection create court-reports ~/Downloads/reports/ --description "Court reports 2025"   # copies files into data/collections/court-reports/docs/
cgprr -c court-reports collection add ~/Downloads/more.pdf    # add files later
cgprr -c court-reports collection files                       # each file: included?, new / changed / indexed / excluded / rejected
cgprr -c court-reports collection exclude docs/draft.pdf      # keep a file on disk but never parse or index it
cgprr -c court-reports collection build      # screen -> chunk -> index (searchable) -> read pass -> calibrate
cgprr -c court-reports collection status     # read-pass progress
cgprr -c court-reports collection review     # rejected files, ingest decisions, quarantined chunks
cgprr -c court-reports ask "..."             # cited answer or "Not in the sources."
cgprr -c court-reports factcheck --file article.pdf   # report in out/court-reports/
cgprr collection list
```

**What to check before uploading:**
- **Accepted:** PDF with a text layer (scanned images are not OCR'd), DOCX, MD/TXT, HTML.
- **Rejected:** legacy `.doc`, macro-enabled files, encrypted PDFs, executables, and files over 50 MB or 500 pages.
- **Refused at once:** every upload is type-checked from its first bytes, so a renamed executable is refused before it is stored.
- **One collection per unit or topic,** so confidential sets never mix.
- **Personal data:** emails, phone numbers, card and Aadhaar numbers are redacted unless `redact` in `collection.toml` says otherwise. Secrets are always redacted.

`collection.toml` options: `sources` (globs), `exclude` (files never parsed), `chunking` (`"structure"`, the default, or `"paragraph"`), `chunk_tokens` (350), `require_synthetic` (refuse unlabelled files; on for `cgprr-news`), `redact` (which identifier kinds to redact; secrets are always redacted), `mask_profanity`.

**Build steps.** A collection is **searchable after step A** (parse → screen → chunk → BM25 → embed; seconds to minutes). **Step B**, the read pass, then runs one short model call per chunk, and the improved index is swapped in atomically. The read pass is cached by content, so a rebuild only re-reads changed chunks. `--no-read` skips it.

**What the read pass may and may not do.** Its context lines, questions and keywords become extra BM25 tokens and extra vectors (a chunk's dense score is the best of its own vector and its question vectors). It never touches:
- the vocabulary behind the unknown-entity check;
- gate 2;
- the verbatim-quote checks.

So an entity or figure the reader invents can never ground an answer. A generated question that carries a figure absent from its chunk is dropped.

## 4. Ask, fact-check, serve

```bash
cgprr -c cgprr-news ask "How many tariff lines does the Kestria–Orlan Trade Partnership Agreement cover?"
cgprr -c cgprr-news,public-sample ask "..."        # several collections; ids become <collection>/<doc>:p<n>
cgprr -c cgprr-news ask --no-llm "..."             # zero model calls: gate 1, then the best source sentences verbatim
cgprr -c cgprr-news factcheck --article A01        # or --all, or --file any.pdf|.docx|.html|.md
cgprr -c cgprr-news serve --port 8765              # local server: indexes and models stay warm
curl -s localhost:8765/ask -H 'Content-Type: application/json' -d '{"question": "...", "collections": ["cgprr-news"]}'
```
The server binds to 127.0.0.1 only (there is no authentication) and refuses another host unless `CGPRR_ALLOW_REMOTE=1`. It also refuses requests whose `Host` is not loopback (DNS rebinding), whose `Origin` is another site, and POSTs that are not `Content-Type: application/json` (cross-site form posts). `{"llm": false}` in an /ask body is the no-model mode below. With `"stream": true` it sends progress events (`retrieved`, `generating`, `final`). Raw model tokens are deliberately not streamed: an answer is shown only after gate 2 has checked it.

**Gates (code, not model):**
- **Gate 1, before any model call:** a proper noun absent from every selected collection is refused. So is a question whose best hit is below the collection's coverage/cosine thresholds.
- **Gate 2, after generation:** every statement must cite a passage that was actually shown. Every number must be in a cited passage, and every proper noun in a cited document. The statement must contain no secret or personal identifier that its sources do not contain, and no URL that is not in its sources.

**Fact-check verdicts** (this system's definitions):
- **supported:** a passage states the same fact, and every number in the claim appears in the cited evidence.
- **unsupported:** a passage covers the same point and contradicts the claim.
- **unverified:** no passage covers it, or the evidence failed a code check.

## 5. Content security

**Layer 1: files.**
- The file type is detected from the bytes, not the extension.
- **Rejected:** executables, extension/content mismatches, encrypted PDFs, macro Office files (`.docm`, or a `vbaProject.bin` inside a `.docx`), zip bombs, legacy binary Office files, and files over the size/page limits.
- Hidden text in DOCX (hidden, white or <2 pt runs) and HTML (`display:none`, `hidden`, `aria-hidden`) is removed and reported.
- HTML scripts, navigation and remote resources are dropped; nothing is fetched.
- Every file is parsed in a separate worker process with a timeout (plus a memory cap on Linux), including an article passed to `factcheck --file`.
- Control characters (ANSI terminal escapes, NUL, BEL) and invisible characters (zero-width space, bidi overrides, soft hyphen) are stripped from all extracted text, so a document cannot rewrite your terminal when its text is printed, and `ig\u200bnore instructions` still matches the injection rules.
- Folder uploads skip hidden files/folders (`.git`, `.DS_Store`) and symlinks; files in hidden folders inside a collection are never indexed.

**Layers 2–4: chunks.** Every block of text is screened *before* chunking, so a quarantined block never takes benign neighbouring text with it.

| Finding | Action |
|---|---|
| Prompt injection; credential or account-verification requests aimed at the reader | **Quarantine** (not indexed; a person can release it) |
| API keys, private keys, tokens, `password: …` values | **Redact** (`[REDACTED:secret]`), always |
| Card numbers (Luhn-valid), Aadhaar (Verhoeff-valid), emails, phone numbers | **Redact** (configurable per collection) |
| Step-by-step instructions for weapons / explosives / malware / drug synthesis | **Quarantine** |
| Sexual content + minors (rule heuristic) | **Quarantine for review**; classifier S4 → the whole file is dropped, nothing stored |
| Reporting on violence, crime, drugs, extremism, hate, self-harm; quoted slurs; profanity | **Keep and label.** The fact stays answerable, and its `content_warning:<category>` label is shown with every answer or fact-check row that cites it. |

A news desk has to check stories about crime, scams and abuse, so **topic alone never removes a fact**. A scam warning ("police urged residents never to share your OTP") is kept. A message that *asks* for the OTP is quarantined.

- **Llama Guard 3 (1B, local),** when `CGPRR_GUARD_MODEL` is set, runs once per chunk at build time, adding nothing at answer time. It can only add labels or make an outcome stricter.
- **Audit trail:** every decision goes to `indexes/<name>/ingest_report.jsonl` (ids and rules only, never text). Quarantined text stays local in `quarantine.jsonl` for review, and `cgprr collection review --release HASH` records a human decision that survives rebuilds.

## 6. Evaluation

```bash
cgprr -c NAME eval generate          # draft eval set from the collection's own documents
cgprr -c NAME eval review            # list drafts; --accept G001 G002 | --accept all | --reject G005
cgprr -c NAME collection build       # calibrates gate-1 thresholds on the calibration split
cgprr -c NAME eval all               # retrieval + Q&A + claims + end-to-end (if articles exist)
scripts/measure.sh all               # same, plus VRAM, power, energy, tokens/s
```
**Auto-generated eval** (`evalgen.py`). Code builds whatever it can:
- **Answerable:** the model writes a question, but its answer key must be copied verbatim from the chunk.
- **Refusal cases:** entity-swap, planted-figure and off-topic questions are built by code.
- **Contradicted claims:** code changes a real figure.

Every item is a draft until reviewed. Items are split deterministically: **30% calibration** (used only to set thresholds) and **70% test** (used only for scores). Questions too similar to an indexed read-pass question are dropped, to avoid leakage.

**Calibrated thresholds** (`calibrate.py`). The thresholds maximise Youden's J (TPR − FPR) while letting through ≥95% of answerable calibration questions. Negatives that the unknown-entity check already refuses are excluded. With too few items, the defaults (coverage 0.5, cosine 0.60) are kept and the manifest says so.

The latency target is `CGPRR_SLO_ASK_P95_MS`. `cgprr eval qa` reports p50/p95 per stage and exits with code 4 when the target is missed.

### Results

All runs: laptop GTX 1650 Ti, WSL2 Ubuntu, Qwen3-4B-Instruct-2507 Q4_K_M (14%/86% CPU/GPU), `profiles/dev-gtx1650ti.env`, one request at a time. Raw outputs: `out/baselines/linux-original-4b/` (v0.1 code) and `out/runs-linux-v4/` (current code; `out/runs-linux-v2/` holds the run that exposed the bugs fixed below, plus the server-latency run).

**`cgprr-news`** (10 synthetic sources, 39 chunks, hand-written gold eval: 16 questions, 30 claims, 4 articles)

| | v0.1 baseline | A: v0.2, BM25 only | B: v0.2 full (guard + read pass + dense) |
|---|---|---|---|
| Retrieval: gold in top-5 (QA / claims) | 9/9, 25/25 | 9/9, 25/25 | 9/9, 25/25 (also without read-pass keys) |
| Gold claims wrongly stopped at gate 1 | 1 (A02-C2) | 1 (A02-C2) | **0**: the read pass fixes A02-C2 |
| Q&A pass | 14/16 | **15/16** | 14/16 |
| Q&A answer / citation correct | 7/9, 7/9 | 8/9, 8/9 | 7/9, 8/9 |
| Refusals and adversarial (7 items) | 7/7 | 7/7 | 7/7 |
| Claim accuracy after gates | 0.97 | 0.97 | 0.93 |
| **False "supported"** | **0** | **0** | **0** |
| End-to-end: gold recovered (one-to-one) / verdict agreement | 28/30 (many-to-one, overcounted) / 24/28 | 24/30 / 18/24 | 24/30 / 18/24 |
| Q&A latency p50 / p95 | 5.0 s / 7.8 s | 4.2 s / 8.4 s | 4.4 s / 6.1 s |
| All LLM calls: p50 / p95, output tok/s | 12.5 s / 67.8 s, 7.84 | 11.9 s / 61.9 s, 8.06 | 11.8 s / 51.4 s, 9.04 |
| `eval all`: wall time, avg GPU power, energy | 782 s, 26.6 W, 0.0058 kWh | 720 s, 28.6 W, 0.0057 kWh | 818 s, 26.7 W, 0.0061 kWh |
| Peak VRAM (idle ≈ 1,080 MiB) | 3,482 MiB | 3,443 MiB | 3,440 MiB |

Reading it:
- On this tiny, clean corpus BM25 already finds every gold passage, so dense retrieval and the read pass have little to add. Their one measurable gain is A02-C2, a paraphrased claim that BM25 under-scores and that now reaches the model.
- B's two extra Q&A failures (Q07 refused, Q08 incomplete) and two claim misses (A01-C8, A04-C4) come from the different passage set that dense retrieval shows the model. With the same index, a rerun gives identical results (temperature 0).
- **A second, independent build varies.** The fresh-clone test regenerated the read pass from scratch. On the same eval it scored Q&A 14/16 (Q06 and Q08 fail), claim accuracy **1.00** and verdict agreement 20/24, still with 0 false "supported". Read single-run differences of one or two items between configurations as within run-to-run variation, not as a ranking.
- On this corpus BM25 alone is enough, and A is the simpler configuration. B pays off on paraphrase-heavy real documents, below.
- The one-to-one end-to-end count replaced v0.1's matching, which let one extracted claim match several gold claims. The 28/30 baseline figure overstates recall.

B's build step, run once: Llama Guard over 39 chunks, the read pass (39 chunks + 10 documents, 113 generated questions) and embeddings took **381 s**.

**`public-sample`** (3 real public documents: the arXiv "Attention Is All You Need" PDF, the Wikipedia Chandrayaan-3 HTML page, and a DOCX built from the Wikipedia RBI summary; 188 chunks; **auto-generated, unreviewed** eval)

| | v2 (first run) | v4 (after fixes) |
|---|---|---|
| Ingest | 8 emails redacted, 1 hidden element removed, 0 quarantined (after narrowing `role_hijack`), 63 Llama Guard labels | same |
| Read pass (188 chunks, 709 questions) | 243 s (resumed from cache after an interrupted run) | cached |
| Eval set | 36 answerable questions, 10 refusal, 6 adversarial, 21 claims; 4 dropped as near-duplicates of read-pass questions; 30% held out for calibration | same items |
| Gate-1 thresholds | coverage 0.875 / cosine 0.703, **overfitted** to 7 positives | 0.5 / 0.6 (`capped_at_defaults`, 8 positives < 20) |
| Retrieval: gold in top-5 (QA / claims) | 24/28, 13/13 (25/28 without read-pass keys) | same |
| Q&A pass (test split) | 17/38 | **25/38** |
| False refusals (answerable) | 15/28 (6 "unknown entity" and 6 below threshold at gate 1; 3 by the model) | 7/28 (all by the model, none at gate 1) |
| Answer correct: strict / lenient key match | 7/28 | 15/28 / 19/28 |
| Refusals and adversarial | 10/10 | 10/10 |
| Claim accuracy / false "supported" | 0.71 / **0** | 0.82 / **0** |
| Q&A latency p95 (target 10 s) | 14.8 s, **missed** (exit 4) | 8.8 s, met |
| `eval all`: wall time, energy | 522 s, 0.0040 kWh | 609 s, 0.0047 kWh |

The first real-document run found three defects, each fixed and covered by a test:
1. **Entity check and tokenizer disagreed.** `Chandrayaan-3` was checked as `chandrayaan-` while the vocabulary held `chandrayaan`, so a name present in the documents was refused, and gate 2 downgraded two supported claims.
2. **Calibration picked the strictest of equally good thresholds.** That sat exactly on the lowest calibration positive. It now takes the maximum-margin threshold, and with fewer than 20 positives it may only loosen the defaults.
3. **Prompt size on long chunks.** About 1,400 prompt tokens per question at ~90 tok/s prefill (layers partly on the CPU) missed the latency target. `CGPRR_QA_PASSAGE_TOKENS=900` keeps the top-ranked passages that fit.

The remaining Q&A failures are mostly eval-set defects, not answer defects:
- Verbatim answer keys ("warmup_steps = 4000", "Starlink G6-31 (23 satellites)") that a correct paraphrase does not contain. The lenient column checks numbers plus 60% of key terms and is reported separately, never folded into the pass count.
- Two trivia questions generated from a figure caption.

Review (`cgprr eval review`) exists for exactly this. The scores above are on **draft** items.

**Warm server** (`cgprr serve`, `cgprr-news`):
- The first request took 46 s, including the cold model load.
- Later answers took 4.4–4.7 s; retrieval was under 2 ms and gate-1 refusals 2 ms.
- A repeated question returned in 3 ms from the cache.
- A one-shot `cgprr ask` costs about 0.4 s more for process start and index load.

**No-model mode** (`cgprr ask --no-llm`, `cgprr-news` gold set, BM25 index, `python scripts/eval_nollm.py`; no GPU, ~2 ms per question):

| | `ask --no-llm` (0 model calls) | `ask` with the model (run A above) |
|---|---|---|
| Answerable: key fact in the returned text | 8/9 | 8/9 |
| Refusals (`refuse`) | **1/3** | 3/3 |
| Adversarial (`refuse_or_*`) | 4/4 | 4/4 |
| Pass | 13/16 | 15/16 |

It finds the right sentences, but it cannot tell "the entity is in the sources but this fact is not" (Q09 cost of KNS Arvel, Q12 Orlan's vote): gate 1 passes, and it returns the nearest sentences. Use it as a cited lookup, never as a verified answer.

## 7. Known limits

- **Small gold set.** 16 questions and 30 claims: one item moves a score by 3–6 points. "0 false supported" means none observed in ~51 claims (30 gold + 21 auto); with zero errors in 51 trials the 95% upper bound on the true rate is still about 6%.
- **Gate-1 thresholds on `cgprr-news` are the defaults (coverage 0.5, cosine 0.60).** The hand-written gold items are test-only, so build-time calibration has no positives and keeps the defaults; those defaults were chosen while developing against the same gold set, so the `cgprr-news` refusal scores are optimistic. `public-sample` uses a held-out calibration split.
- **Same process wrote the data and the system.** The synthetic corpus, the gold answers and the code came from the same AI-assisted process (§12), so the system may fit its own test data. The public-sample results are the out-of-sample check.

- **Correct numbers in a false relation pass the number gate.** A04-C3 ("rose to 5.1% from 5.4%" when inflation fell) can only be caught by the model's judgement.
- **unsupported vs unverified** is the hardest boundary for a small model. When unsure, the prompt and the code both push towards "unverified".
- **Entity grounding is a capitalisation heuristic, not NER.** Sentence-initial or lower-case inventions are not caught.
- **Screening is English-only and pattern-based.** Obfuscated secrets (base64, split lines) and text hidden in PDFs are not detected. Content labels are coarse keyword signals.
- **Documents:** scanned PDFs (no OCR), multi-column PDF layouts, DOCX text boxes/footnotes, and JavaScript-rendered HTML are not handled.
- **Auto-generated eval items are drafts** written partly by the same model family that is evaluated; scores on unreviewed items are indicative only.
- **Laptop numbers come from a 4B model with ~14% of its layers on the CPU.** Do not present them as the office system's (7B, full GPU).
- **The passage budget trades recall for latency.** With 350-token chunks, `CGPRR_QA_PASSAGE_TOKENS=900` shows the model only the top 2–3 of 5 retrieved passages. Most remaining false refusals on `public-sample` are the model answering "not in sources", and some may have lost the gold passage to the budget. Set the budget to 0 on a full-GPU machine.
- **Calibration needs ≥20 reviewed answerable questions** before it may tighten the gate-1 thresholds. Until then it can only loosen them.
- **Llama Guard 3 1B over-labels real text** (63 of 188 public-sample chunks, none of them meaningful). Its labels are hints, not findings.

## 8. Layout

```
src/cgprr/         config, collections, schemas (Pydantic), obs (JSON logs), textnorm, guard (injection),
                   security (redaction, labels, quarantine), classify (Llama Guard), loaders/ (md, txt, pdf, docx,
                   html, sandbox), chunking, ingest, reader (read pass), index, retrieve (BM25+dense, multi-collection),
                   llm (retry/breaker/cache), verify (gate 2), prompts, rag, calibrate, evalgen, evaluate, serve,
                   ops (create/add/include/exclude/build/review, shared by the CLI), cli
src/cgprr/factcheck/   claims, verdict, summarise, pipeline, render
collections/cgprr-news/   synthetic source pack, articles, hand-written gold eval, DATA_CARD.md
data/collections/  your own collections (git-ignored; CGPRR_DATA_DIR)
kaggle/            7B notebook for a Kaggle T4 + its generator
tests/             125 pytest tests with a fake LLM (code paths and gates; not model quality)
scripts/           quickstart.sh (setup + smoke test), measure.sh (VRAM, power, throughput), eval_nollm.py
```

## 9. Configuration

All settings come from `CGPRR_*` environment variables (`src/cgprr/config.py`; ready-made sets in `profiles/`):

| Variable | Purpose |
|---|---|
| `CGPRR_COLLECTION` | Default collection (`cgprr-news`) |
| `CGPRR_LLM_URL`, `CGPRR_LLM_MODEL`, `CGPRR_BACKEND` | Model server |
| `CGPRR_USE_DENSE`, `CGPRR_EMBED_MODEL`, `CGPRR_EMBED_ON_CPU` | Dense retrieval |
| `CGPRR_READ_PASS` | Read pass (default 1) |
| `CGPRR_GUARD_MODEL` | Build-time classifier (empty = off) |
| `CGPRR_NUM_CTX`, `CGPRR_NUM_PREDICT`, `CGPRR_QA_MAX_TOKENS` | Context and output limits |
| `CGPRR_QA_PASSAGE_TOKENS` | Prompt budget for Q&A passages (0 = all top-k). 900 on the laptop, where prefill on partly CPU-resident layers dominates latency |
| `CGPRR_TOP_K` (5), `CGPRR_MAX_PER_DOC` (3) | Retrieval breadth and per-document cap |
| `CGPRR_MIN_COVERAGE`, `CGPRR_MIN_DENSE` | Override the calibrated thresholds |
| `CGPRR_KEEP_ALIVE` (30m) | How long models stay loaded |
| `CGPRR_SLO_ASK_P95_MS` | Latency target for the Q&A eval |
| `CGPRR_ISOLATE_PARSING` (1) | Parse files in worker processes |
| `CGPRR_LOG_QUERY_TEXT` (0) | Log query text instead of a hash |
| `CGPRR_ALLOW_REMOTE_LLM` (0) | Allow a `CGPRR_LLM_URL` that is not loopback / private / a single-label host (refused by default) |
| `CGPRR_ALLOW_REMOTE` (0) | Let `cgprr serve` bind beyond loopback; also disables its Host/Origin checks |

**Kaggle (7B):**
1. Upload `cgprr-factcheck-kaggle.zip` as a dataset and import `kaggle/cgprr_kaggle_7b.ipynb`.
2. Set Accelerator → GPU T4 x2 and Internet → On, then Run All.

Kaggle is a cloud GPU. If you report those numbers, say so; the statements above about no cloud GPU refer to the laptop runs.

## 10. Privacy

Indexes, logs, caches, quarantine files and reports stay on local disk; after the one-time model pulls nothing leaves the machine.
- The model URL must be local (loopback, private network, or a Docker service name) unless `CGPRR_ALLOW_REMOTE_LLM=1`, and the client ignores `HTTP(S)_PROXY`, so prompts (which carry document text) cannot be routed off the machine by an environment setting.
- Logs never contain prompts, passages or (by default) query text.
- The ingest report never contains document text.
- Secrets and personal identifiers are redacted before indexing, so they never reach a prompt or an answer.

## 11. Data

- **`collections/cgprr-news/`:** synthetic and fictional, labelled `synthetic: true` (see its `DATA_CARD.md`).
- **`public-sample`**, built at evaluation time from three public documents:
  - the arXiv paper 1706.03762, under the licence on its arXiv page;
  - the Wikipedia "Chandrayaan-3" page, CC BY-SA 4.0;
  - the Wikipedia "Reserve Bank of India" summary, CC BY-SA 4.0.

  It is used for local evaluation only. It is git-ignored and not redistributed in the zip.

No data file contains planted instructions. Security handling is tested with fixtures built inside the tests.

## 12. AI tools used

- **Claude Code** (Anthropic; model Claude Opus 5.5), an AI coding assistant in the terminal and VS Code.
  - **Used for:** planning the design, drafting code, synthetic data, tests and documentation, and running the builds and evaluations on the laptop.
- **How the output was checked:**
  - 125 automated tests;
  - real-model runs on the laptop, with the raw outputs kept in `out/`;
  - reading the evaluation failures case by case.

  Three defects found this way were fixed and given regression tests: hyphenated names refused as unknown, overfitted calibration thresholds, and the latency miss on long chunks.
- **Claude (Anthropic; Claude Opus 5.5), in a separate review session on 20 Nov 2026:** security review and hardening (local-server DNS-rebinding/cross-site checks, control and zero-width character stripping, sandboxed `--file` parsing, local-only model URL, safer folder uploads), the no-model `ask --no-llm` mode, `profiles/lite.env`, `scripts/quickstart.sh`, `QUICKSTART.md` and `SECURITY_REVIEW.md`. Each fix has a regression test; the review lists what was verified and what remains open.
- The author reviewed every part and can explain it.
