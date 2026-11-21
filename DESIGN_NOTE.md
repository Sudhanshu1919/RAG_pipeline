# Design note: CGPRR local RAG and fact-check (Core + Track C)

Business: CGPRR news desk. Core: cited Q&A over a source pack, refusing with "Not in the sources." when the sources do not support an answer. Track C: fact-check an article (5–8 claims, each supported / unsupported / unverified, with verbatim evidence) and write a summary from supported claims only. Everything runs on one machine with open-weight models. Full detail: [docs/DESIGN_DETAILS.md](docs/DESIGN_DETAILS.md).

## 1. Architecture

```
documents (PDF, DOCX, MD/TXT, HTML) → parse in isolated workers → screen each block → chunk by heading
  → index: BM25 + bge-m3 vectors  → (optional) read pass: context line + likely questions per chunk, search keys only
question → retrieve (BM25 + dense, RRF fusion) → GATE 1 → local LLM, JSON-schema output → GATE 2 → cited answer | refusal
article  → extract claims (quote must be verbatim) → per claim: retrieve → GATE 1 → verdict + evidence quotes → code checks
         → summary from supported claims, every sentence through GATE 2 → report (JSON / Markdown / HTML)
```

**The rule: the model proposes; code decides.**

- **Gate 1 (before any model call):** refuse if the question names an entity found in no document, or if no passage clears the relevance threshold.
- **Gate 2 (after generation):** every statement must cite a passage that was shown; every number must appear in a cited passage; every proper noun must appear in the cited document; no secret, identifier or URL may be absent from the sources. One failure refuses the whole answer.
- **Fact-check:** an evidence quote that is not verbatim makes the evidence invalid. A "supported" verdict with no valid evidence, or with a number or name missing from it, is downgraded to "unverified". Confidence is computed by code.

The read pass helps retrieval only. The vocabulary, gate 2 and the quote checks use raw document text, so anything the model invents at ingest can never support an answer.

## 2. Model choice

- **Generator: Qwen2.5-7B-Instruct, Q4_K_M** (Apache-2.0), served by Ollama. It follows JSON schemas reliably at 7B and fits the card with room to spare. The laptop used for development ran **Qwen3-4B-Instruct-2507 Q4_K_M**.
- **Embeddings: bge-m3** (MIT), from the same Ollama server.
- **Optional build-time classifier: Llama Guard 3 1B.** It over-labels real text (63 of 188 chunks on public documents), so it may only add labels, never delete.
- **Keyword retrieval: BM25 with the Lucene idf.** The plain Okapi idf scores 0 on tiny collections.
- **Swappable:** any Ollama or OpenAI-compatible model works through `CGPRR_LLM_MODEL`.

## 3. GPU and memory

**Assumed VRAM: 24 GB.** That is the smallest RTX 6000-class card (Quadro RTX 6000); the Ada (48 GB) and Blackwell (96 GB) cards only add headroom.

| Office, 7B | Estimate |
|---|---|
| Weights, Q4_K_M | ≈ 4.7 GB |
| KV cache: 28 layers × 4 KV heads × 128 × 2 (K, V) × 2 B = 56 KiB/token; 8k context × 4 parallel | ≈ 1.9 GB |
| bge-m3 + runtime buffers | ≈ 1.7–2.2 GB |
| **Total** | **≈ 8.3–8.8 GB of 24 GB** |

**What actually ran:** a GTX 1650 Ti laptop GPU (4 GB, Turing) under WSL2. No cloud GPU was used. Measured with `scripts/measure.sh` on the 4B model, which ran 14% on the CPU and 86% on the GPU:

- **Peak VRAM:** 3,443 MiB, against 1,078 MiB at idle (the display).
- **Speed:** 8.1 output tokens per second.
- **Q&A latency:** 4.2 s median, 8.4 s p95.

Decoding speed is set by memory bandwidth, so scaling by bandwidth and weight size gives an estimate for 7B on the office card: **≈ 15 / 21 / 40 tok/s** on the Quadro RTX 6000 / RTX 6000 Ada / RTX PRO 6000. A Kaggle T4 notebook is included to measure 7B directly.

## 4. Evaluation method

- **Gold set** for `cgprr-news`, written before any model run:
  - 16 questions: 9 answerable, 3 not in the sources, 1 unknown entity, 2 planted figures, 1 speculation;
  - 30 gold claims;
  - 4 articles with planted errors.
- **Real documents:** an arXiv PDF, a Wikipedia HTML page and a DOCX, with an **auto-generated** eval set. These items are drafts until reviewed; 30% is held out for calibrating thresholds.
- **Headline metric: false "supported"**, meaning an unsupported or unverified claim reported as supported. That is the costly newsroom error.

| Run (laptop, 4B) | Q&A pass | Claim accuracy | False supported | Q&A p95 |
|---|---|---|---|---|
| cgprr-news, original code | 14/16 | 0.97 | 0 | 7.8 s |
| cgprr-news, BM25 only | 15/16 | 0.97 | 0 | 8.4 s |
| cgprr-news, full pipeline (guard + read pass + dense) | 14/16 | 0.93 | 0 | 6.1 s |
| public documents, auto eval (drafts) | 25/38 | 0.82 | 0 | 8.8 s |

A second, independent build of the full pipeline (fresh clone, read pass regenerated) scored 14/16 and claim accuracy 1.00. Differences of one or two items between rows are within run-to-run variation. "0 false supported" is 0 in ~51 claims: the 95% upper bound on the true rate is still ~6%. The `cgprr-news` gate-1 thresholds are defaults chosen while developing on the same gold set, so its refusal scores are optimistic; `public-sample` uses a held-out calibration split. With no model at all (`ask --no-llm`: retrieval, gate 1, verbatim sentences) the gold Q&A set scores 13/16: it finds 8/9 answers but refuses only 1 of 3 missing facts, so it is a lookup tool, not the answer path.

**What the real-document eval exposed, all fixed and covered by tests:**
1. Hyphenated names such as "Chandrayaan-3" were refused as unknown.
2. Thresholds calibrated on only 7 examples refused half the test questions.
3. Long chunks pushed latency past the 10 s target.

**Failures that remain:**

- Some correct answers miss a verbatim answer key.
- The model refuses 7 of 28 answerable real-document questions.
- Unsupported and unverified get confused on scope claims.
- Correct numbers in a false relation pass the number check.

## 5. Data privacy

- **Nothing leaves the machine.** The index, logs, cache and reports stay on local disk. The model server and `cgprr serve` bind to 127.0.0.1; the server also rejects non-local `Host`/`Origin` headers (DNS rebinding, cross-site requests). The model URL must be local unless explicitly overridden, and proxy variables are ignored, so a setting cannot route prompts off the machine.
- **Logs** hold ids, latencies and gate outcomes. They never hold prompts or passages, and questions only as a hash.
- **Office install:** one **collection** per business (NEET, Hirrx, CGPRR), each with its own index, log, cache, eval set and service account. A question sees only the collections it selects, and the index loader refuses an index built for another collection. The GPU serves stateless model calls only.
- **Content screening at build time:**
  - Injection and credential requests are quarantined for human review.
  - Secrets, card and Aadhaar numbers, emails and phone numbers are redacted.
  - Reporting on crime or violence is kept and labelled, so the facts stay checkable.
- **Cloud:** none was used for the reported numbers. An on-site install only changes the profile (`profiles/office-rtx6000.env`) and uses a native Linux host.
- **Data:** all of it is synthetic or public.

## 6. Cost

No API fees, and no cloud GPU. The marginal cost is electricity.

- **Measured:** one full evaluation (4 articles, 46 eval items) drew 28.6 W of GPU power for 720 s, which is **0.0057 kWh ≈ ₹0.05** at an assumed ₹8/kWh. That is GPU board only; the whole laptop draws perhaps 2–3× more.
- **Build:** the one-time read pass took 381 s for 39 chunks.
- **Office:** an RTX 6000-class card draws 300–600 W but finishes several times faster, so the energy per article stays in paise. The one-time model downloads are about 4.7 GB (7B) plus 1.2 GB (bge-m3).
