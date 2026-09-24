#!/bin/bash
# run_gpqa_llama_test.sh — 一键全流程编排：GPQA-Diamond 评测 Meta-Llama-3.1-8B-Instruct FP16 + RedHat W8A8
# Usage: bash run_gpqa_llama_test.sh [--smoke N] [--skip-download] [--skip-fp16] [--skip-w8a8]
set -euo pipefail

BASEDIR="/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test"
MODEL_FP16_DIR="$BASEDIR/models/Meta-Llama-3.1-8B-Instruct"
MODEL_W8A8_DIR="$BASEDIR/models/Meta-Llama-3.1-8B-Instruct-W8A8-RedHat"
FP16_PORT=8811
W8A8_PORT=8812
SMOKE=0
SKIP_DOWNLOAD=false
SKIP_FP16=false
SKIP_W8A8=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --smoke) SMOKE="$2"; shift 2 ;;
        --skip-download) SKIP_DOWNLOAD=true; shift ;;
        --skip-fp16) SKIP_FP16=true; shift ;;
        --skip-w8a8) SKIP_W8A8=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done
cd "$BASEDIR"
echo "============================================"
echo " GPQA-Diamond Meta-Llama-3.1-8B-Instruct"
echo "============================================"

# ── Step 1: Download models ──
if [ "$SKIP_DOWNLOAD" != "true" ]; then
    echo ""
    echo "[Step 1] Downloading models..."

    if [ ! -d "$MODEL_FP16_DIR" ]; then
        echo "  Downloading Meta-Llama-3.1-8B-Instruct (FP16)..."
        python3 -c "
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download('meta-llama/Meta-Llama-3.1-8B-Instruct', local_dir='$MODEL_FP16_DIR')
" 2>&1
    else
        echo "  FP16 model already exists at $MODEL_FP16_DIR"
        ls "$MODEL_FP16_DIR" 2>/dev/null | head -5
    fi

    if [ ! -d "$MODEL_W8A8_DIR" ]; then
        echo "  Downloading RedHat/Meta-Llama-3.1-8B-Instruct-quantized.w8a8 (W8A8)..."
        python3 -c "
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download('RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8', local_dir='$MODEL_W8A8_DIR')
" 2>&1
    else
        echo "  W8A8 model already exists at $MODEL_W8A8_DIR"
        ls "$MODEL_W8A8_DIR" 2>/dev/null | head -5
    fi
    echo "  Download complete."
fi

# ── Step 2: Check model files ──
echo ""; echo "[Step 2] Checking model files..."
for d in "$MODEL_FP16_DIR" "$MODEL_W8A8_DIR"; do
    if [ -d "$d" ]; then
        echo "  ✓ $d"
        ls "$d" 2>/dev/null | head -5
    else
        echo "  ✗ $d MISSING"
    fi
done

# ── Step 3: Check quantization config ──
echo ""; echo "[Step 3] Checking W8A8 quantization config..."
if [ -d "$MODEL_W8A8_DIR" ]; then
    echo "  JSON files in W8A8 model dir:"
    find "$MODEL_W8A8_DIR" -maxdepth 1 -name "*.json" 2>/dev/null | head -10
    echo "  Config files (*.yml, *.yaml):"
    find "$MODEL_W8A8_DIR" -maxdepth 1 \( -name "*.yml" -o -name "*.yaml" \) 2>/dev/null | head -10
fi

# ── Step 4: FP16 eval ──
if [ "$SKIP_FP16" != "true" ]; then
    echo ""
    echo "[Step 4] Starting FP16 vLLM server..."

    # Kill any lingering vLLM processes
    lsof -ti:$FP16_PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
    pkill -9 -f 'start_vllm_llama_fp16' 2>/dev/null || true
    pkill -9 -f 'VLLM::EngineCor' 2>/dev/null || true
    sleep 2

    tmux new-session -d -s vllm_fp16_llama "cd $BASEDIR && python3 start_vllm_llama_fp16.py 2>&1 | tee vllm_fp16_llama.log"
    echo "  FP16 server starting (tmux: vllm_fp16_llama)..."

    echo "  Waiting for FP16 server..."
    for i in $(seq 1 120); do
        if curl -s http://127.0.0.1:$FP16_PORT/v1/models >/dev/null 2>&1; then
            echo "  FP16 server ready after ${i}s"
            break
        fi
        if [ $i -eq 120 ]; then
            echo "  ✗ FP16 server failed to start within 120s"
            tmux capture-pane -t vllm_fp16_llama -p -S -50
            exit 1
        fi
        sleep 2
    done

    echo ""
    echo "  Running FP16 evaluation..."
    python3 eval_gpqa_llama.py --mode fp16 --smoke $SMOKE 2>&1 | tee eval_fp16_llama.log
    echo "  FP16 eval complete (log: eval_fp16_llama.log)"

    # Stop FP16 server to free NPU memory
    tmux kill-session -t vllm_fp16_llama 2>/dev/null || true
    sleep 3
    echo "  FP16 server stopped (NPU memory freed for W8A8)."
fi

# ── Step 5: W8A8 eval ──
if [ "$SKIP_W8A8" != "true" ]; then
    echo ""
    echo "[Step 5] Starting Red Hat W8A8 vLLM server..."

    lsof -ti:$W8A8_PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
    pkill -9 -f 'start_vllm_llama_w8a8' 2>/dev/null || true
    pkill -9 -f 'VLLM::EngineCor' 2>/dev/null || true
    sleep 2

    tmux new-session -d -s vllm_w8a8_llama "cd $BASEDIR && python3 start_vllm_llama_w8a8_redhat.py 2>&1 | tee vllm_w8a8_llama.log"
    echo "  W8A8 server starting (tmux: vllm_w8a8_llama)..."

    echo "  Waiting for W8A8 server..."
    for i in $(seq 1 120); do
        if curl -s http://127.0.0.1:$W8A8_PORT/v1/models >/dev/null 2>&1; then
            echo "  W8A8 server ready after ${i}s"
            break
        fi
        if [ $i -eq 120 ]; then
            echo "  ✗ W8A8 server failed to start within 120s"
            tmux capture-pane -t vllm_w8a8_llama -p -S -50
            exit 1
        fi
        sleep 2
    done

    echo ""
    echo "  Running W8A8 evaluation..."
    python3 eval_gpqa_llama.py --mode w8a8 --smoke $SMOKE 2>&1 | tee eval_w8a8_llama.log
    echo "  W8A8 eval complete (log: eval_w8a8_llama.log)"

    # Stop W8A8 server
    tmux kill-session -t vllm_w8a8_llama 2>/dev/null || true
    sleep 2
fi

# ── Step 6: Summary ──
echo ""
echo "============================================"
echo " All done!"
echo "============================================"
echo ""
echo "Results:"
for logf in eval_fp16_llama.log eval_w8a8_llama.log; do
    if [ -f "$logf" ]; then
        echo "  --- $logf ---"
        grep -E "(Accuracy|Results|latency|TTFT|TPOT|Throughput)" "$logf" 2>/dev/null || echo "  (metrics not found)"
    fi
done
echo ""