# Design details: CGPRR local RAG and fact-check

The long version behind the two-page [DESIGN_NOTE.md](../DESIGN_NOTE.md): the full architecture, memory arithmetic, evaluation results and content policy.

## 1. Architecture

```
collections/<name>/ (PDF, DOCX, MD/TXT, HTML) + collection.toml
  build step A (searchable):
    loaders in isolated workers ─ layer 1: type from bytes, encrypted/macro/zip-bomb/executable → rejected; hidden text removed
    screen each block           ─ layers 2–4: injection/credential requests/operational harm → quarantine (kept apart);
                                  secrets & identifiers → redacted; sensitive subject matter → labelled, kept
    [optional Llama Guard 3 1B] ─ stricter only: labels, S4 → drop file
    structure-aware chunking    ─ never crosses a heading; ~350-token chunks; quarantined block = its own chunk
    index                       ─ BM25 tokens + bge-m3 vectors, ingest_report.jsonl, quarantine.jsonl
  build step B (enrich, background-able):
    read pass (1 call/chunk, cached) ─ context line + 2–4 questions + keywords → extra BM25 tokens and question vectors
                                       (never used for vocabulary, gate 2 or quotes)
    calibrate                   ─ gate-1 thresholds from the collection's own calibration questions (Youden J, TPR ≥ 0.95)

ask(question, collections a,b…) ─ guard scan ─ per-collection hybrid retrieve (BM25 + max(cos chunk, cos questions), RRF,
                                    ≤3 hits per document) ─ cross-collection RRF, ids namespaced "<coll>/<doc>:p<n>"
              ─ GATE 1 (code): unknown entity | below the collection's thresholds → "Not in the sources." (no model call)
              ─ local LLM, JSON-schema-constrained output {answerable, statements[{text, citations P1..Pk}]}
              ─ GATE 2 (code): ≥1 citation, citations ∈ shown passages, every number ∈ cited passage, every proper noun ∈ cited
                               document, no secret/identifier absent from the sources, no URL absent from the sources
              ─ answer with [S03:p1] citations + content labels | refusal + reason → JSONL log

factcheck(article) ─ extract 5–8 claims {text, article_quote}; drop any quote not verbatim in the article
                   ─ per claim (concurrent): retrieve → GATE 1 → verdict {verdict, evidence[{passage_id, quote}]}
                     → code: drop non-verbatim quotes; supported/unsupported without valid evidence → unverified;
                       supported with a number or proper noun absent from the evidence → unverified; optional 2nd pass, disagreement → unverified
                   ─ summary from SUPPORTED claims only; each sentence passes GATE 2 or is dropped
                   ─ report JSON/MD/HTML: verdict, code-computed confidence, flags, uncertainty note
```
Rule: **the model proposes; code decides.** Refusal and downgrade happen in code that the model's output cannot override. All model outputs are Pydantic models, constrained at decode time (Ollama `format` / vLLM guided JSON) and validated again on receipt. Nothing parses free text. Untrusted text (questions, articles, passages) is delimited and declared to be data. Source paragraphs that match injection patterns are quarantined from the index.

Resilience: async httpx client; exponential backoff with jitter on transport errors, 5xx and 429; a circuit breaker; a concurrency semaphore sized to the GPU; an exact-match disk cache (deliberately not semantic: near-duplicate fact-check prompts with different figures must not share answers). Critical-path failures (the model being unreachable during Q&A or claim extraction) fail loudly (exit 3). Non-critical failures degrade: dense retrieval falls back to BM25, a failed claim check becomes "unverified, check_error", and a failed summary is reported as unavailable.

## 2. Model choice

- **Generator:** Qwen2.5-7B-Instruct, Q4_K_M GGUF (Apache-2.0). It follows JSON schemas reliably at 7B and its licence allows on-premise use. At 4-bit it fits the smallest RTX 6000 generation with room to spare. Any Ollama/vLLM model can be swapped in with `CGPRR_LLM_MODEL`; on a 48/96 GB card, a 14B–32B model is the first upgrade to test against the same eval.
- **Embeddings:** bge-m3 (MIT, ~568M parameters), served by the same Ollama instance, so there is no second runtime. On the 4 GB laptop it embeds queries on the CPU (`CGPRR_EMBED_ON_CPU=1`), so it never competes with the generator for VRAM. BM25 alone already reaches 9/9 and 25/25 gold-passage recall on `cgprr-news`; dense retrieval matters for paraphrase-heavy collections.
- **Content classifier:** Llama Guard 3 1B (Llama 3.2 Community Licence), build time only. On news text it over-flags reporting (the `cgprr-news` run labels election and defamation topics), which is why its verdicts map to labels, never to deletions, except S4.
- **Lexical scoring:** BM25 with the Lucene idf `log(1 + (N − n + 0.5)/(n + 0.5))`. rank_bm25's Okapi idf is exactly 0 for a term in half of a tiny collection, which silently turned ranking into file order on a 2-chunk test collection.
- **Vector search:** an exact cosine matrix product in NumPy. For 39 passages (and up to ~50k) exact search is faster and simpler than an ANN index. FAISS HNSW is the drop-in replacement above that size.

## 3. GPU and memory

**VRAM assumed for the office: 24 GB.** Three cards carry the "RTX 6000" name: the Quadro RTX 6000 (Turing, 24 GB), the RTX 6000 Ada (48 GB) and the RTX PRO 6000 Blackwell (96 GB). Designing for 24 GB means the system runs on whichever one the office has.

**What was actually run.** An NVIDIA GTX 1650 Ti laptop GPU: 4,096 MiB, 50 W cap, ~192 GB/s memory bandwidth, Turing. It sits on Windows 11 + WSL2, and the Windows display holds 738 MiB (WDDM), which leaves ~3.3 GiB. No cloud GPU was used. The office model (7B, ~4.7 GB of weights) does not fit, so:
- **Dev model, fully on the GPU:** Qwen3-4B-Instruct-2507 Q4_K_M. Context 3,072, q8_0 KV cache, flash attention, one request at a time, dense retrieval off. Used for the demo and for speed/VRAM measurement.
- **Office model, split across GPU and CPU:** Qwen2.5-7B-Instruct Q4_K_M with part of the layers in CPU RAM. Used only to measure accuracy with the office model; its speed is not representative.

| Memory (arithmetic, before measuring) | Dev laptop, 4B | Office, 7B |
|---|---|---|
| Weights (Q4_K_M) | ≈ 2.5 GB | ≈ 4.7 GB |
| KV per token (layers × KV heads × 128 × K,V) | 36×8: 144 KiB fp16 → 72 KiB q8_0 | 28×4: 56 KiB fp16 |
| KV at the context used | 3,072 ctx q8_0 ≈ 0.22 GB | 8,192 ctx × 4 parallel ≈ 1.9 GB |
| Embeddings | off | bge-m3 ≈ 1.2 GB |
| Compute buffers / runtime | ≈ 0.3 GB | ≈ 0.5–1 GB |
| **Total** | **≈ 3.0 GB of 3.3 usable** | **≈ 8.3–8.8 GB of 24 GB** |

**Measured** (`scripts/measure.sh`, dev profile, WSL2, `eval all` on `cgprr-news`, current code):
- **VRAM:** idle 1,078 MiB (the Windows display), peak 3,443 MiB, so +2,365 MiB for the model.
- **Placement:** Ollama reports `14%/86% CPU/GPU`. The arithmetic above (≈3.0 GB) did not quite fit beside the display, so about 5 of 36 layers run on the CPU.
- **Speed:** 8.1 output tok/s on average over 68 calls. Prompt processing is ~90 tok/s.
- **Latency per call:** p50 11.9 s, p95 61.9 s. The tail is long verdict and summary calls. Q&A alone is p50 4.2 s, p95 8.4 s.
- **bge-m3** runs at 100% on the CPU (1.2 GB of system RAM), so it never takes VRAM from the generator.
- **The 7B office model was not run on the laptop** (≈4.7 GB of weights cannot fit beside the display). The Kaggle notebook measures it on a T4.

Prefill is the latency lever on this card. With 350-token chunks, five passages make a ~1,400-token prompt, and prefill alone took 15 s at p95, missing the 10 s target. A 900-token passage budget brought that to 8.8 s.

**Estimate for the office card.** Single-stream decoding is limited by memory bandwidth: every token reads all the weights once. Expected office tok/s ≈ measured 4B tok/s × (office bandwidth ÷ 192 GB/s) × (2.5 GB ÷ 4.7 GB) for the 7B model. The bandwidth ratios are about ×3.5 (Quadro RTX 6000, 672 GB/s), ×5.0 (RTX 6000 Ada, 960 GB/s) and ×9.3 (RTX PRO 6000 Blackwell, ~1.8 TB/s). That puts 7B decoding there at ≈ 8.1 × 1.9 / 2.7 / 4.9 ≈ **15 / 21 / 40 tok/s**. The estimate is conservative, because the laptop figure already includes a CPU-offload penalty. Prompt processing (compute-bound) improves more than this, so the estimate is conservative. The 24 GB headroom allows 4–6 parallel requests or a 14B model. 48/96 GB allows a 32B model, or vLLM with continuous batching for several editors at once.

Token budget, from the actual prompts (characters ÷ 3.2, a conservative ratio): the largest shipped prompts are 574 tokens (extraction), 986 (verdict), 869 (Q&A) and 1,284 (summary). Output is capped at 768 tokens per call. One article with 8 claims ≈ 7–9k input and ≈ 1.5k output tokens. The client refuses any prompt that would exceed the context (`ContextOverflowError`) instead of letting the server silently cut off the system prompt.

## 4. Evaluation method

For `cgprr-news`, expected behaviour was written into `collections/cgprr-news/eval/*.jsonl` before any model run. Any other collection gets a **generated** eval set (`cgprr eval generate`):
- **Answerable:** the model writes the question, but code checks that the answer key is copied verbatim from the chunk.
- **Refusal cases:** entity-swap, planted-figure and off-topic items are built by code.
- **Contradicted claims:** code changes a real figure.
- **Leakage guard:** an item is dropped when it is too close to an indexed read-pass question (Jaccard ≥ 0.8).
- **Split and review:** items are split deterministically, 30% calibration and 70% test. All start as drafts until a person reviews them.

**Gate-1 calibration:** maximise Youden's J subject to TPR ≥ 0.95 on the calibration split. Among equally good thresholds take the maximum-margin one. With fewer than 20 positives, only loosening the defaults is allowed. The first rule as written, "strictest on ties", overfitted 7 positives into coverage 0.875 and refused half the test questions. That failure is why the margin rule and the cap exist.

Four suites:
1. **Retrieval** (deterministic): is the gold passage in the top 5, and is gate-1 routing correct. Measured: 9/9 and 25/25, with or without read-pass keys. A02-C2, a paraphrased gold-unsupported claim, is wrongly routed to unverified by BM25 alone. With the read pass and dense retrieval it reaches the model and is judged correctly.
2. **Core Q&A** (16 questions): 9 lookups, 3 not-in-sources, 1 unknown entity, 2 planted figures, 1 speculation. Columns: answer correct, citation correct, false refusals, refusal correct, adversarial handled.
3. **Claim verdicts on 30 gold claims** (16 supported / 9 unsupported / 5 unverified). Each gold claim is given to the verifier directly, so verifier quality is measured apart from extraction. Outputs: per-class precision/recall/F1, confusion matrix, raw model accuracy vs. accuracy after the gates, and **false "supported"**, the safety metric (a wrong "supported" is the costly error for a newsroom).
4. **End-to-end:** gold claims recovered by extraction, verdict agreement on recovered claims, and summary sentences kept/dropped by gate 2.

Hard cases are planted on purpose: right numbers in the wrong direction (A04-C3), "no date published" vs. a quoted date (A02-C6), a wrong ordinal (A03-C2) and a wrong vote (A04-C2). Thresholds on `cgprr-news` (coverage 0.5, cosine 0.60) were set before the runs and are disclosed. Failures are listed, not tuned away.

**Results** (laptop, 4B; full tables in README §6; raw files in `out/runs-linux-v4/`):

| | Q&A pass | claim accuracy | false "supported" | end-to-end recovered / agree | Q&A p95 |
|---|---|---|---|---|---|
| `cgprr-news`, v0.1 baseline | 14/16 | 0.97 | 0 | 28/30 (overcounted) / 24/28 | 7.8 s |
| `cgprr-news`, BM25 (A) | 15/16 | 0.97 | 0 | 24/30 / 18/24 | 8.4 s |
| `cgprr-news`, full v0.2 (B) | 14/16 | 0.93 | 0 | 24/30 / 18/24 | 6.1 s |
| `public-sample`, full v0.2, draft eval | 25/38 | 0.82 | 0 | — | 8.8 s |

Remaining failures:
- **`cgprr-news`:**
  - Q06 (A and the baseline): gate 2 refuses with `entity_not_in_cited_source`. This fails closed, as designed.
  - Q07 (B): the model answers "not in sources".
  - Q08: under B the answer lacks the expected key; the baseline refused it.
  - A01-C8 (scope): judged unverified instead of unsupported.
  - A04-C4: a supported number judged unsupported under B.
- **`public-sample`:**
  - 7 refusals by the model on answerable questions. The passage budget may hide the gold passage.
  - 9 answers that miss a verbatim key, of which 4 pass the lenient key check.
  - 3 claims where unsupported and unverified are confused.

On every run and every collection, nothing that the gold set marks unsupported or unverified was reported as supported.

## 5. Data privacy and content security

Everything runs on one machine: the index, logs, cache and reports sit under the repository on local disk. The model server binds to `127.0.0.1`, and so does `cgprr serve`, which refuses any other address unless `CGPRR_ALLOW_REMOTE=1`. After `ollama pull`, no network access is needed. Logs hold trace ids, latencies, token counts, passage ids and gate outcomes. They never hold prompts or passages, and query text is stored only as a hash unless `CGPRR_LOG_QUERY_TEXT=1`.

**Collections are the unit of separation.** Each `collections/<name>/` has its own documents, index, log file, LLM cache and eval set. The index loader refuses an index whose manifest names a different collection. A question sees only the collections selected for it (`-c a,b`). An office install gives each unit (`cgprr`, `neet`, `hirrx`, …) its own collections and service account. The shared GPU serves stateless model calls only, and nothing persists in the model server between requests. `cgprr-news` keeps `require_synthetic = true`, so the loader rejects any file there not labelled `synthetic: true`. Other collections accept real documents.

**Content policy** (applied at build time, before anything is indexed). Every decision is written to `ingest_report.jsonl` as ids and rule names, never as text.
- **Files:** the type comes from the bytes. Executables, type mismatches, encrypted PDFs, macro Office files, zip bombs, legacy binary Office files and files over the size/page limits are rejected. Hidden DOCX/HTML text is removed. Parsing runs in a worker process with a timeout and a memory cap.
- **Quarantine (kept apart, never indexed; a person can release it):** prompt injection, credential or account-verification requests aimed at the reader, and step-by-step operational harm. On real text a pattern can misfire: "act as a thermal blanket" on the Wikipedia Chandrayaan-3 page matched the role-hijack rule. The rule was narrowed to AI-role objects, and the case is now a regression test.
- **Redact:** secrets and keys always; Luhn-valid cards, Verhoeff-valid Aadhaar numbers, emails and phone numbers by default (per collection). The public-sample build redacted 8 emails (author addresses in the arXiv PDF).
- **Keep and label:** reporting on violence, crime, drugs, hate, elections and defamation. The fact stays answerable, and the label travels with every citation. Topic alone never deletes a fact: a newsroom must be able to check a story about a stabbing.
- **Llama Guard 3 1B** (optional, build time only) can only add labels or make an outcome stricter; S4 (child sexual exploitation) drops the whole file. On `cgprr-news` it labelled 7 of 39 chunks (elections ×3, violence ×3, defamation ×1) and removed nothing. On `public-sample` it labelled 63 of 188 chunks, and on inspection none was meaningful: "intellectual property" ×37 on reference lists and figures, "indiscriminate weapons" ×2 and "violence" ×6 on a lunar mission and a machine-learning paper. A 1B classifier at this false-positive rate can be trusted to label, never to delete. That is the policy, and the labels should be read as "look here", not "this is harmful".
- **Output side:** gate 2 refuses an answer containing a secret or identifier, or a URL, that is absent from its cited sources.

Testing used no cloud GPU: everything ran on the developer laptop (Windows 11 + WSL2). The Kaggle notebook is a separate, optional run on public or synthetic data. For an on-site install, the only changes are the profile (`profiles/office-rtx6000.env`) and a native Linux host. The code, data layout and logs stay the same.

## 6. Cost

Local only; no API fees, no cloud GPU. Energy = average board power × time × tariff. The laptop GPU is capped at 50 W, which gives an upper bound before measuring: 50 W × 1 h = 0.05 kWh. Measured during `eval all` on `cgprr-news` (dev profile, GPU board power from `nvidia-smi`): 28.6 W × 720 s = **0.0057 kWh**, which at an assumed ₹8/kWh is **≈ ₹0.05 per full evaluation** (4 articles + 46 eval items). The full pipeline (B) used 0.0061 kWh, and `public-sample` 0.0047 kWh. The one-time build with the read pass took 381 s for 39 chunks and ~40 min for 188 chunks on the laptop: well under ₹0.1 of GPU energy. These figures count the GPU board only. Whole-laptop draw (CPU, display) is perhaps 2–3× higher and still in paise. Office: the RTX 6000-class cards are rated at 300 W (Ada, Blackwell Max-Q) up to 600 W (Blackwell Workstation Edition) at full load. They finish each article several times faster, so energy per article stays in the same order of magnitude. The workstation is already owned, so the marginal cost is electricity. The one-time model downloads are ~2.5 GB (4B) and ~4.7 GB (7B).
