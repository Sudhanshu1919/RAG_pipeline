# Security and evaluation review (2026-11-20)

Scope: all of `src/`, the CLI, the local server, Docker files, profiles, and the shipped outputs.
Method: full test suite, `ruff`, `mypy --strict`, `bandit`, `pip-audit`, a line-by-line read of the file loaders, sandbox, ingest, screening, server, LLM client and upload code, and live probes (a crafted HTTP request against `cgprr serve`, files with terminal escapes and zero-width characters, renamed executables, a hidden `.git` folder, a symlink in an upload folder).

Tool results before the fixes: 108 tests pass; ruff clean; `pip-audit`: no known vulnerabilities in the dependencies; `bandit`: only low-severity notes (`random` used for retry jitter and eval sampling, not for security; one `assert`). `mypy --strict`: 15 errors, all typing noise around untyped libraries except one confusing variable reuse (fixed). After the fixes: **125 tests pass**, ruff clean.

## 1. Fixed

| # | Severity | Problem (verified) | Fix | Test |
|---|---|---|---|---|
| 1 | **High** | `cgprr serve` answered any request that reached the port. A web page you visit could use DNS rebinding (its domain resolving to 127.0.0.1) to read answers and source passages from your collections, or POST `text/plain` forms cross-site to run model calls. Probe: `Host: evil.example` + `Origin: http://evil.example` + `Content-Type: text/plain` returned 200 with passage ids. | The server now refuses a non-loopback `Host`, a non-loopback `Origin`, and a POST that is not `application/json` (403). `CGPRR_ALLOW_REMOTE=1` turns the checks off together with the bind restriction. | `test_local_server_health_and_ask` |
| 2 | Medium | Terminal injection: control characters in a document (ANSI/OSC escapes, BEL) were kept in the index and printed raw by `ask` and `collection review`, so a document could retitle or clear the terminal and fake output. | `textnorm.clean_text` strips C0/C1 control characters and invisible format characters from every extracted block, heading and metadata field, from strict-collection Markdown, and from server `/factcheck` text. | `test_clean_text_*`, `test_loaded_blocks_have_no_control_characters` |
| 3 | Medium | Zero-width characters split words, so `Ig\u200bnore all previous instructions` passed the injection screen. | Removed before scanning (and from stored text). ZWJ/ZWNJ are kept in stored text for Indic scripts. | `test_zero_width_characters_do_not_hide_an_injection` |
| 4 | Medium | `factcheck --file` parsed the article in the CLI process: no timeout and no memory cap, unlike every collection file. A malformed PDF could hang or exhaust the process. | Parsed in the same isolated worker (timeout, 2 GB address-space cap on Linux). | `test_factcheck_file_is_parsed_in_the_sandbox` |
| 5 | Medium | `CGPRR_LLM_URL` was not checked, and the HTTP client honoured `HTTP(S)_PROXY`. One typo or an inherited proxy variable would send every prompt, with document text, off the machine, contradicting "nothing leaves the machine". | Non-local URLs are refused unless `CGPRR_ALLOW_REMOTE_LLM=1` (local = loopback, private IP, `.local/.lan/.internal`, or a single-label Docker host such as `ollama`). The client sets `trust_env=False`. | `test_is_local_url`, `test_remote_model_url_is_refused_unless_allowed` |
| 6 | Low | Folder upload (`collection create/add DIR`) copied hidden files (`.git/*`, `.env.md`) and followed symlinks to files outside the folder; it read every file into memory before the size check; folders over 50 files had the rest silently refused. | Hidden files/folders and symlinks are skipped, size is checked before reading, large folders go in batches of 50, missing paths are reported. `create` exits 1 when nothing was added. | `test_folder_upload_skips_hidden_and_symlinks_and_batches` |
| 7 | Low | Files inside a hidden folder of a collection (e.g. `docs/.git/…`) were indexed; only hidden *file names* were skipped. | Any hidden path component is skipped. | `test_files_in_hidden_folders_are_not_indexed` |
| 8 | Low | `PermissionError` and other `OSError`s escaped as Python tracebacks. | Reported as `ERROR: …`, exit 1. | — |
| 9 | Privacy | The shipped `out/` logs held two local usernames in absolute paths. | Replaced with `<user>` in the packaged copy. | — |

## 2. Lightweight / minimal-AI changes

- `profiles/lite.env`: one 2.5 GB model, BM25 only, no read pass, no Llama Guard, no embeddings. Build = 0 model calls. This is the configuration that scored best on the shipped gold set (README §6, run A: Q&A 15/16, 0 false "supported").
- `cgprr ask --no-llm` (server: `{"llm": false}`): retrieval + gate 1, then the best-matching source sentences verbatim with ids. 0 model calls, ~2 ms. Measured on the gold set with `scripts/eval_nollm.py`: 13/16 (answerable 8/9, refusal 1/3, adversarial 4/4). It cannot refuse "entity present, fact absent" questions, so it is a cited lookup, not an answer.
- `scripts/quickstart.sh`: setup, tests, sample build and a no-model question in one command.
- Package slimmed from 213 MB to under 2 MB compressed: the zip carried a Windows `.venv` (201 MB, unusable on any other machine), stale model-output caches from an earlier collection name, logs, a built index, tool caches, `egg-info`, a duplicate output folder (`out/cgprr` = `out/baselines/windows-original-4b`), and an empty `data/collections/my-docs` that appeared in `collection list` and failed to build.

## 3. Open risks (not fixed; state them if asked)

1. **No authentication on `cgprr serve`.** Loopback plus the new Host/Origin checks stop browser attacks, but any local user or process on the same machine can query every collection. Fine for a single-user laptop; an office deployment needs a token and TLS in front.
2. **Prompt-injection screening is English regex.** Paraphrased or non-English instructions pass. The real defence is gate 2 (citations, numbers, entities, verbatim quotes checked in code), which limits what an injection can make the model *say*; it does not stop a wrong relation between correct numbers (README §7, A04-C3).
3. **Zip-bomb check trusts sizes declared in the zip headers.** A forged header passes; the worker's 2 GB memory cap is the backstop on Linux/macOS. **Windows has no memory cap.**
4. **Hidden-text removal is partial.** DOCX: run-level hidden/white/<2 pt only (style-level hidden, near-white colours, text boxes, comments are not handled). HTML: inline styles only (CSS classes not evaluated). PDF: none (white or off-page text is indexed).
5. **Redaction is pattern-based.** Obfuscated secrets (base64, split lines) are missed. False positives exist: `The pin is 560001` (a postal PIN code) is redacted as a secret (verified).
6. **Document-derived text on disk is not encrypted.** `.cache/<name>/` holds model outputs that quote documents and is not purged when a document is removed; `indexes/<name>/quarantine.jsonl` holds quarantined (redacted) text. Delete both folders when a collection must be forgotten.
7. **Supply chain.** Python dependencies have lower bounds only (no lock file) and `docker-compose.yml` uses `ollama/ollama:latest`. Pin both for a reproducible submission (`pip freeze > requirements.lock`, an explicit image tag).
8. **Markdown reports are not escaped.** Raw HTML in an article claim would render in some Markdown viewers. The HTML report escapes everything.
9. **LLM cache key ignores `num_ctx`/`num_predict`.** Changing them does not invalidate cached outputs; clear `.cache/` after changing model settings.
10. **Ollama itself** is outside this code: if `OLLAMA_HOST=0.0.0.0` is set, the model server is exposed to the network.

## 4. Evaluation: what the numbers do and do not show

- **The gold set is small.** 16 questions, 30 claims, 4 articles: one item moves Q&A by 6 points. The README correctly treats 1–2 item differences as run-to-run noise. Do not rank configurations on it.
- **"0 false supported" is the headline safety result**, over 30 gold + 21 auto claims. With 0 errors in ~51 trials the 95% upper bound on the true rate is still ~6% (rule of three). Say "none observed", not "never".
- **Same author for data and system.** The synthetic corpus, the gold answers and the code were produced in the same AI-assisted process, so the system may be tuned to its own test data. The public-sample eval is auto-generated and unreviewed.
- **Gate-1 thresholds on `cgprr-news` are the defaults (0.5 / 0.60)**, not calibrated values: the hand-written gold items are test-only, so calibration finds 0 positives (`default:too_few_items(pos=0,neg=9)`). `config.py` says the defaults were "calibrated on the shipped eval set", i.e. chosen on the test set. Disclose this.
- **Unit tests use a fake LLM.** They prove the gates and code paths, not answer quality. Only `cgprr eval` with a real model measures quality.
- **Reproducibility.** Temperature 0 with a fixed seed gives identical results on the same index; a rebuilt read pass changes results (README §6). Report numbers with the profile, model tag and Ollama version.

## 5. Before submitting

The README (§12) states that the author reviewed every part and can explain it. Make that true for these six mechanisms, because they carry every claim the project makes:
1. Gate 1 (`rag.py`, `retrieve.py`): unknown-entity check on raw vocabulary, coverage/cosine thresholds; no model call on refusal.
2. Gate 2 (`verify.py`): every statement cites a shown passage; numbers, proper nouns, URLs and identifiers must appear in the cited sources.
3. Verbatim quote checks (`textnorm.is_substring_quote`, `claims.py`, `verdict.py`) and the downgrade rules to "unverified".
4. Why the read pass cannot ground an answer (it feeds retrieval tokens only, never the vocabulary or gate 2).
5. File security layer (`loaders/__init__.py`, `sandbox.py`): type from bytes, worker process, timeout, memory cap.
6. Calibration (`calibrate.py`): Youden's J with a 95% answerable floor, and why it may only loosen thresholds with fewer than 20 positives.
