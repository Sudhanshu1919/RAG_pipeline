#!/usr/bin/env bash
# One-shot setup and smoke test for a new machine (Linux / WSL2 / macOS). Safe to re-run.
# Usage:  bash scripts/quickstart.sh
# Steps: Python check -> .venv -> install -> tests -> lite profile -> build sample index (no model needed)
#        -> one no-LLM question -> if Ollama is running: one model question.
set -euo pipefail
cd "$(dirname "$0")/.."

py=$(command -v python3 || command -v python || true)
[ -n "$py" ] || { echo "Python 3.11+ is required: install it first." >&2; exit 1; }
"$py" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' || { echo "Python 3.11+ is required (found $("$py" -V))." >&2; exit 1; }

[ -d .venv ] || "$py" -m venv .venv
# shellcheck disable=SC1091
. .venv/bin/activate
python -m pip install -q --upgrade pip
python -m pip install -q -e ".[dev]"

echo "== tests (no GPU or model server needed)"
python -m pytest -q

set -a
# shellcheck disable=SC1091
. profiles/lite.env
set +a

echo "== build the sample collection (BM25 only: zero model calls)"
cgprr -c cgprr-news collection build --no-read --no-calibrate > /dev/null
cgprr collection list

echo "== ask without a model (verbatim source sentences)"
cgprr -c cgprr-news ask --no-llm "How many TEU did the Port of Velmora handle in FY2025?"

echo "== ask with the local model"
if cgprr health > /dev/null 2>&1; then
  cgprr -c cgprr-news ask "How many TEU did the Port of Velmora handle in FY2025?"
else
  echo "Ollama is not reachable at ${CGPRR_LLM_URL:-http://127.0.0.1:11434}. Start it and pull the model:"
  echo "  ollama serve &   then   ollama pull ${CGPRR_LLM_MODEL}"
fi
echo
echo "Done. Next: see QUICKSTART.md (add your own documents: cgprr collection create NAME FOLDER)."
