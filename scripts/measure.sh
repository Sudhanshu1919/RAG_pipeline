#!/usr/bin/env bash
# Measure peak VRAM and power while running the full evaluation, then print token throughput.
# Usage: scripts/measure.sh [eval-suite]   (default: all)
# Requires nvidia-smi. Writes out/<collection>/measure_gpu.csv. Run with no other GPU jobs active.
# Edge cases NOT handled: multi-GPU hosts (GPU 0 only), MIG partitions, power.draw = [N/A]
# (some laptop/WSL2 drivers): energy is then reported as 0 and must be estimated from the 50 W cap.
set -euo pipefail
SUITE="${1:-all}"
OUT="out/${CGPRR_COLLECTION:-cgprr-news}"
mkdir -p "$OUT" logs
CSV="$OUT/measure_gpu.csv"

command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found: GPU figures cannot be measured on this host" >&2; exit 2; }

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader -i 0 | tee "$OUT/gpu_name.txt"
# Baseline with the model unloaded: on WDDM laptops the display already holds VRAM.
if command -v ollama >/dev/null; then ollama stop "${CGPRR_LLM_MODEL:-qwen2.5:7b-instruct-q4_K_M}" 2>/dev/null || true; ollama stop "${CGPRR_EMBED_MODEL:-bge-m3}" 2>/dev/null || true; sleep 2; fi
BASE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | head -1)
echo "baseline_vram_mib=$BASE"
nvidia-smi --query-gpu=timestamp,memory.used,power.draw,utilization.gpu --format=csv,noheader,nounits -i 0 -lms 250 > "$CSV" &
SMI=$!
trap 'kill $SMI 2>/dev/null || true' EXIT

START=$(date +%s)
: > "logs/${CGPRR_COLLECTION:-cgprr-news}.jsonl"   # fresh log so stats cover this run only
EVAL_RC=0
CGPRR_CACHE=0 cgprr eval "$SUITE" > "$OUT/measure_eval_stdout.md" || EVAL_RC=$?   # 4 = latency target missed
END=$(date +%s)
kill $SMI 2>/dev/null || true

echo "--- GPU/CPU placement reported by Ollama (100% GPU = no CPU offload) ---"
command -v ollama >/dev/null && ollama ps || true

python3 - "$CSV" "$START" "$END" "$BASE" <<'PY'
import csv, sys
rows = [r for r in csv.reader(open(sys.argv[1])) if len(r) >= 4]
base = float(sys.argv[4])
mem = [float(r[1]) for r in rows]; pwr = [float(r[2]) for r in rows if r[2].strip() not in ("[N/A]", "")]
wall = int(sys.argv[3]) - int(sys.argv[2])
avg_w = sum(pwr) / len(pwr) if pwr else 0.0
print(f"peak_vram_mib={max(mem) if mem else 0:.0f}")
print(f"peak_vram_above_baseline_mib={(max(mem) - base) if mem else 0:.0f}")
print(f"avg_power_w={avg_w:.1f}")
print(f"wall_seconds={wall}")
print(f"energy_kwh={avg_w * wall / 3.6e6:.5f}")
PY
cgprr stats
echo "eval_exit_code=$EVAL_RC"
exit "$EVAL_RC"
