# Quick start

This tool answers questions from **your own documents** and fact-checks articles against them. It runs on your machine: a local model (served by Ollama) writes the answers, and code checks every answer against the documents before you see it. If the documents do not support an answer, it says exactly `Not in the sources.`

Run every command from the project folder (the one with `pyproject.toml`). Commands that start with `cgprr` need the virtual environment active (step 2).

---

## 1. What you need

| | Minimum | Notes |
|---|---|---|
| Python | 3.11 or newer | `python3 --version` |
| Disk | ~4 GB | ~2.5 GB for the model |
| GPU | optional | NVIDIA with 4 GB VRAM works; CPU-only works but answers are slow |
| OS | Linux, WSL2 on Windows, macOS, or Windows | Linux/WSL is the tested path |

## 2. Install (once)

**Linux / WSL2 / macOS**
```bash
bash scripts/quickstart.sh        # creates .venv, installs, runs the tests, builds the sample, asks one question
```
Or by hand:
```bash
python3 -m venv .venv
. .venv/bin/activate              # run this again in every new terminal
pip install -e ".[dev]"
pytest -q                         # all tests should pass; no GPU or model needed
```

**Windows (PowerShell)**
```powershell
py -3 -m venv .venv               # Python 3.11+
.venv\Scripts\Activate.ps1        # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -e ".[dev]"
pytest -q
```

## 3. Start the local model (once per boot)

Install Ollama from https://ollama.com, then:
```bash
ollama serve &                                   # leave it running (on Windows the Ollama app does this)
ollama pull qwen3:4b-instruct-2507-q4_K_M        # ~2.5 GB, one time
```

Load a settings profile in each new terminal. **`lite` makes the fewest model calls** and is the recommended default:
```bash
set -a; . profiles/lite.env; set +a              # Linux / WSL / macOS
```
```powershell
# Windows PowerShell
Get-Content profiles\lite.env | Where-Object { $_ -match '^[A-Z_]+=.+' } | ForEach-Object { $k, $v = $_ -split '=', 2; Set-Item "env:$k" $v }
```
Check: `cgprr health` must print `"reachable": true`.

Other profiles: `dev-gtx1650ti.env` (the measured laptop setup), `office-rtx6000.env` and `kaggle-t4-7b.env` (7B model, dense retrieval, more model calls).

## 4. Try it on the sample data

```bash
cgprr -c cgprr-news collection build                                              # index the 10 sample documents (seconds)
cgprr -c cgprr-news ask "How many TEU did the Port of Velmora handle in FY2025?"   # cited answer
cgprr -c cgprr-news ask "What is the GDP growth forecast for Zandoria in 2027?"    # -> Not in the sources.
cgprr -c cgprr-news ask --no-llm "How many TEU did the Port of Velmora handle?"    # no model at all: source sentences
cgprr -c cgprr-news factcheck --article A01                                       # report in out/cgprr-news/
```
Open `out/cgprr-news/factcheck_A01.html` in a browser to read the fact-check report.

## 5. Add your own documents

**Step 1: check your files.**

| Accepted | Refused |
|---|---|
| PDF with selectable text | scanned PDFs (no OCR: they give "no_text"), encrypted PDFs |
| Word `.docx` | `.doc`, `.docm` and any file with macros |
| `.md`, `.txt`, `.html` | executables, zip files, anything renamed to look like a document |
| up to 50 MB and 500 pages per file | larger files |

Keep unrelated or confidential sets in **separate collections**; a question only searches the collections you name.

**Step 2: create a collection from a folder or files.** The name must be lowercase letters, digits, `-` or `_`.
```bash
cgprr collection create court-reports ~/Documents/reports/ --description "Court reports 2025"
```
Files are **copied** into `data/collections/court-reports/docs/` (your originals are untouched). Each file is listed as `added` or `refused` with the reason. Hidden files, hidden folders and symlinks inside the folder are skipped.

On Windows/WSL, a Windows folder is at `/mnt/c/Users/<you>/Documents/...`.

**Step 3: build the index.**
```bash
cgprr -c court-reports collection build
cgprr -c court-reports collection review     # what was rejected, redacted or quarantined, and why
```
Emails, phone numbers, card and Aadhaar numbers are replaced by `[REDACTED:...]`; passwords and API keys always are. Text that tries to instruct the AI ("ignore previous instructions") is quarantined: kept out of the index until a person releases it with `collection review --release HASH`.

**Step 4: use it.**
```bash
cgprr -c court-reports ask "What sentence did the court hand down in the Rao case?"
cgprr -c court-reports factcheck --file ~/Downloads/article.docx      # any accepted file type
cgprr -c court-reports,cgprr-news ask "..."                           # search two collections at once
```

**Later changes.**
```bash
cgprr -c court-reports collection add ~/Downloads/new-report.pdf     # add more files
cgprr -c court-reports collection files                              # every file: new / changed / indexed / excluded / rejected
cgprr -c court-reports collection exclude docs/draft.pdf             # keep the file, stop indexing it
cgprr -c court-reports collection include docs/draft.pdf             # index it again
cgprr -c court-reports collection build                              # always rebuild after a change
```
To remove a file, delete it from `data/collections/<name>/docs/` and rebuild. To remove a whole collection, delete `data/collections/<name>/` and `indexes/<name>/`.

## 6. How much AI each command uses

With `profiles/lite.env`:

| Command | Model calls |
|---|---|
| `collection build` | 0 |
| `ask --no-llm` | 0 (returns source sentences verbatim; cannot refuse a fact that is missing but whose names are present) |
| `ask` | 1, or 0 when the code refuses first; a repeated question is answered from the cache |
| `factcheck` | 1–2 to extract 5–8 claims, 1 per claim, 1 for the summary |
| `eval all` | one per test item; cached on re-runs |

Turning on `CGPRR_READ_PASS=1` adds one call per chunk at build time; `CGPRR_USE_DENSE=1` adds an embedding model; `CGPRR_GUARD_MODEL=llama-guard3:1b` adds one classifier call per chunk; `factcheck --consistency` doubles the claim checks.

## 7. Evaluation

```bash
cgprr -c cgprr-news eval all              # retrieval + Q&A + claims + end-to-end -> out/cgprr-news/eval_all.md
python scripts/eval_nollm.py cgprr-news   # the same Q&A set with zero model calls
```
For your own collection there is no gold set yet:
```bash
cgprr -c court-reports eval generate      # draft questions and claims from your documents
cgprr -c court-reports eval review        # read them; then --accept G001 G002 ... or --reject G005
cgprr -c court-reports collection build   # recalibrates the refusal thresholds (needs >= 20 accepted questions)
cgprr -c court-reports eval all
```
Scores on unreviewed drafts are indicative only: the drafts were written partly by the model being tested.

## 8. Local web server (optional)

```bash
cgprr -c cgprr-news serve --port 8765
curl -s localhost:8765/ask -H 'Content-Type: application/json' -d '{"question": "How many TEU did the port handle?"}'
```
It listens on this machine only and has no login. Do not expose it to a network.

## 9. Troubleshooting

| Message | Fix |
|---|---|
| `cgprr: command not found` | Activate the environment: `. .venv/bin/activate` (Windows: `.venv\Scripts\Activate.ps1`) |
| `collection 'x' not found` | Run from the project folder, check `cgprr collection list`, or set `CGPRR_ROOT` to the project folder |
| `"reachable": false` / exit code 2 or 3 | Start Ollama (`ollama serve`), pull the model, check `CGPRR_LLM_URL` |
| `model ... not found` | `ollama pull <the CGPRR_LLM_MODEL in your profile>` |
| `CUDA error: shared object initialization failed` (Windows) | Restart Ollama with `OLLAMA_LLM_LIBRARY=cuda_v12` |
| `refused ...: type_mismatch` / `executable` | The file is not what its extension says. Re-save it as PDF or DOCX |
| `no_text` | Scanned PDF. Run OCR first (e.g. `ocrmypdf in.pdf out.pdf`) and add the output |
| `is a shipped sample and read-only` | Create your own collection; `cgprr-news` cannot be changed |
| `CGPRR_LLM_URL ... is not a local address` | Point it at your local Ollama; `CGPRR_ALLOW_REMOTE_LLM=1` only for a server you own |
| Answers are slow | Use `profiles/lite.env`; on a 4 GB GPU keep `CGPRR_QA_PASSAGE_TOKENS=900` |
| Exit code 4 from `eval qa` | Answers met the quality checks but missed the latency target `CGPRR_SLO_ASK_P95_MS` |

## 10. Where things are stored

| Folder | Contents | Safe to delete? |
|---|---|---|
| `data/collections/<name>/` | your copied documents and `collection.toml` | only if you want the collection gone |
| `indexes/<name>/` | the search index, ingest report, quarantine | yes, rebuild recreates it |
| `.cache/<name>/` | cached model outputs | yes (answers are recomputed) |
| `logs/` | JSON logs (no document text, queries hashed) | yes |
| `out/<name>/` | fact-check reports and eval results | yes |
