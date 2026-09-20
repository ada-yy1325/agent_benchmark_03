#!/bin/bash
# run_gpqa_test.sh — 一键全流程编排：GPQA-Diamond 评测 Qwen3-8B-Instruct FP16 + W8A8
# Usage: bash run_gpqa_test.sh [--smoke N] [--skip-download] [--skip-fp16] [--skip-w8a8]
set -euo pipefail

BASEDIR="/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test"
MODEL_FP16_DIR="$BASEDIR/models/Qwen/Qwen3-8B"
MODEL_W8A8_DIR="$BASEDIR/models/ZKMatrix/Qwen3-8B-w8a8-full"
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
echo " GPQA-Diamond Qwen3-8B-Instruct Eval Suite"
echo "============================================"

if [ "$SKIP_DOWNLOAD" != "true" ]; then
    echo ""
    echo "[Step 1] Downloading models..."
    if ! python3 -c "import modelscope" 2>/dev/null; then
        echo "  Installing modelscope..."
        pip install modelscope -q
    fi
    if [ ! -d "$MODEL_FP16_DIR" ]; then
        echo "  Downloading Qwen3-8B (FP16)..."
        python3 -c "
from modelscope.hub.snapshot_download import snapshot_download
md = snapshot_download('Qwen/Qwen3-8B', cache_dir='$BASEDIR/models')
import os
if not os.path.exists('$MODEL_FP16_DIR'):
    os.makedirs('$BASEDIR/models', exist_ok=True)
    os.symlink(md, '$MODEL_FP16_DIR')
" 2>&1 || {
            echo "  Fallback: downloading with local_dir..."
            python3 -m modelscope.hub.snapshot_download Qwen/Qwen3-8B --local_dir "$MODEL_FP16_DIR" 2>&1 || true
        }
    else
        echo "  FP16 model already exists at $MODEL_FP16_DIR"
    fi
    if [ ! -d "$MODEL_W8A8_DIR" ]; then
        echo "  Downloading Qwen3-8B-W8A8..."
        python3 -m modelscope.hub.snapshot_download ZKMatrix/Qwen3-8B-w8a8-full --local_dir "$MODEL_W8A8_DIR" 2>&1 || true
    else
        echo "  W8A8 model already exists at $MODEL_W8A8_DIR"
    fi
    echo "  Download complete."
fi

# ── Step 2: Check model files ──
echo ""; echo "[Step 2] Checking model files..."
for d in "$MODEL_FP16_DIR" "$MODEL_W8A8_DIR"; do
    if [ -d "$d" ]; then
        echo "  ✓ $d"
        ls "$d" 2>/dev/null | head -3
    else
        echo "  ✗ $d MISSING"
    fi
done

# ── Step 3: Check W8A8 quant config ──
echo ""; echo "[Step 3] Checking W8A8 quantization config..."
if [ -d "$MODEL_W8A8_DIR" ]; then
    echo "  JSON files in W8A8 model dir:"
    find "$MODEL_W8A8_DIR" -maxdepth 1 -name "*.json" 2>/dev/null | head -10
fi

# ── Step 4: FP16 eval ──
if [ "$SKIP_FP16" != "true" ]; then
    echo ""
    echo "[Step 4] Starting FP16 vLLM server..."
    
    # Kill any lingering vLLM processes: API server on port + engine-core orphans
    # (setproctitle 'VLLM::EngineCor') that hold NPU memory but aren't on the port.
    lsof -ti:$FP16_PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
    pkill -9 -f 'start_vllm_8b' 2>/dev/null || true
    pkill -9 -f 'VLLM::EngineCor' 2>/dev/null || true
    sleep 2

    tmux new-session -d -s vllm_fp16_gpqa "cd $BASEDIR && python3 start_vllm_8b_fp16.py 2>&1 | tee vllm_fp16_gpqa.log"
    echo "  FP16 server starting (tmux: vllm_fp16_gpqa)..."
    
    echo "  Waiting for FP16 server..."
    for i in $(seq 1 120); do
        if curl -s http://127.0.0.1:$FP16_PORT/v1/models >/dev/null 2>&1; then
            echo "  FP16 server ready after ${i}s"
            break
        fi
        if [ $i -eq 120 ]; then
            echo "  ✗ FP16 server failed to start within 120s"
            tmux capture-pane -t vllm_fp16_gpqa -p -S -50
            exit 1
        fi
        sleep 2
    done
    
    echo ""
    echo "  Running FP16 evaluation..."
    python3 eval_gpqa_qwen3_8b.py --mode fp16 --smoke $SMOKE 2>&1 | tee eval_fp16_gpqa.log
    echo "  FP16 eval complete (log: eval_fp16_gpqa.log)"

    # Stop FP16 server to free NPU memory before W8A8 (both reserve ~0.9*HBM,
    # so running them concurrently would OOM the second one).
    tmux kill-session -t vllm_fp16_gpqa 2>/dev/null || true
    sleep 3
    echo "  FP16 server stopped (NPU memory freed for W8A8)."
fi

# ── Step 5: W8A8 eval ──
if [ "$SKIP_W8A8" != "true" ]; then
    echo ""
    echo "[Step 5] Starting W8A8 vLLM server..."
    
    lsof -ti:$W8A8_PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
    pkill -9 -f 'start_vllm_8b' 2>/dev/null || true
    pkill -9 -f 'VLLM::EngineCor' 2>/dev/null || true
    sleep 2

    tmux new-session -d -s vllm_w8a8_gpqa "cd $BASEDIR && python3 start_vllm_8b_w8a8.py 2>&1 | tee vllm_w8a8_gpqa.log"
    echo "  W8A8 server starting (tmux: vllm_w8a8_gpqa)..."
    
    echo "  Waiting for W8A8 server..."
    for i in $(seq 1 120); do
        if curl -s http://127.0.0.1:$W8A8_PORT/v1/models >/dev/null 2>&1; then
            echo "  W8A8 server ready after ${i}s"
            break
        fi
        if [ $i -eq 120 ]; then
            echo "  ✗ W8A8 server failed to start within 120s"
            tmux capture-pane -t vllm_w8a8_gpqa -p -S -50
            exit 1
        fi
        sleep 2
    done
    
    echo ""
    echo "  Running W8A8 evaluation..."
    python3 eval_gpqa_qwen3_8b.py --mode w8a8 --smoke $SMOKE 2>&1 | tee eval_w8a8_gpqa.log
    echo "  W8A8 eval complete (log: eval_w8a8_gpqa.log)"
fi

# ── Step 6: Summary ──
echo ""
echo "============================================"
echo " All done!"
echo "============================================"
echo ""
echo "Results:"
for logf in eval_fp16_gpqa.log eval_w8a8_gpqa.log; do
    if [ -f "$logf" ]; then
        echo "  --- $logf ---"
        grep -E "(Accuracy|Results|latency|TTFT|TPOT|Throughput)" "$logf" 2>/dev/null || echo "  (metrics not found)"
    fi
done
echo ""
echo "Servers running in tmux sessions:"
echo "  tmux attach -t vllm_fp16_gpqa   (FP16: $FP16_PORT)"
echo "  tmux attach -t vllm_w8a8_gpqa   (W8A8: $W8A8_PORT)"
echo "  To stop: tmux kill-session -t vllm_fp16_gpqa ; tmux kill-session -t vllm_w8a8_gpqa"
fi