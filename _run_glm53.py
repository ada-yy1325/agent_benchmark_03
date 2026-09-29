#!/usr/bin/env python3
"""
Start vLLM server for GLM-5.3-Flash W8A8 and test inference.
Run AFTER quantization succeeds.
"""
import os, sys, json, time, subprocess, signal, requests

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "models", "GLM-5.3-Flash-W8A8")
LOG_FILE = os.path.join(BASE, "vllm_glm.log")

def log(msg):
    ts = time.strftime('%H:%M:%S')
    print(f"[{ts}] {msg}")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{ts}] {msg}\n")

def check_model():
    """Check if quantized model exists."""
    if not os.path.exists(MODEL_DIR):
        log(f"ERROR: Quantized model not found at {MODEL_DIR}")
        log(f"Run _quant_glm53.py first!")
        return False
    config = os.path.join(MODEL_DIR, "config.json")
    if not os.path.exists(config):
        log(f"ERROR: config.json not found in {MODEL_DIR}")
        return False
    log(f"Model found at {MODEL_DIR}")
    with open(config) as f:
        cfg = json.load(f)
    log(f"Architecture: {cfg.get('architectures', ['?'])}")
    return True

def start_server():
    """Start vLLM server in background tmux session."""
    log("Starting vLLM server for GLM-5.3-Flash W8A8...")
    
    cmd = (
        f"cd {BASE} && "
        f"HCCL_OVER_NIC=1 HCCL_INTRA_ROCE_ENABLE=1 "
        f"vllm serve {MODEL_DIR} "
        f"--quantization ascend "
        f"--tensor-parallel-size 4 "
        f"--enable-expert-parallel "
        f"--served-model-name GLM-5.3-Flash-W8A8 "
        f"--port 8801 "
        f"--trust-remote-code "
        f"--enforce-eager "
        f"--gpu-memory-utilization 0.9 "
        f"--max-model-len 512 "
        f"> {os.path.join(BASE, 'vllm_serve.log')} 2>&1"
    )
    
    # Kill old session if exists
    subprocess.run(["tmux", "kill-session", "-t", "vllm_glm"], 
                   capture_output=True)
    time.sleep(1)
    
    result = subprocess.run(
        ["tmux", "new-session", "-d", "-s", "vllm_glm", cmd],
        capture_output=True, text=True
    )
    
    if result.returncode == 0:
        log("vLLM server starting in tmux session 'vllm_glm'")
        return True
    else:
        log(f"Failed to start tmux: {result.stderr}")
        return False

def wait_for_server(timeout=600):
    """Wait for server to become ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get("http://localhost:8801/v1/models", timeout=5)
            if r.status_code == 200:
                elapsed = time.time() - start
                log(f"Server ready in {elapsed:.0f}s")
                return True
        except:
            pass
        time.sleep(10)
    log(f"Server not ready after {timeout}s timeout")
    return False

def test_inference():
    """Send a test prompt."""
    prompt = "Hello! Please introduce yourself briefly."
    log(f"Testing inference with prompt: '{prompt}'")
    
    try:
        r = requests.post(
            "http://localhost:8801/v1/chat/completions",
            json={
                "model": "GLM-5.3-Flash-W8A8",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 50,
                "temperature": 0.7,
            },
            timeout=120
        )
        if r.status_code == 200:
            result = r.json()
            reply = result["choices"][0]["message"]["content"]
            log(f"Inference SUCCESS!")
            log(f"Response: {reply}")
            return reply
        else:
            log(f"Inference FAILED: HTTP {r.status_code}")
            log(f"Response: {r.text[:500]}")
            return None
    except Exception as e:
        log(f"Inference error: {e}")
        return None

def check_npu_memory():
    """Check NPU memory usage."""
    result = subprocess.run(
        ["npu-smi", "info"], capture_output=True, text=True
    )
    for line in result.stdout.split("\n"):
        if "Memory-Usage" in line or "HBM-Usage" in line:
            log(line.strip())

if __name__ == "__main__":
    # Step 1: Check quantized model
    if not check_model():
        sys.exit(1)
    
    # Step 2: Start server
    log("=" * 50)
    log("STEP: Starting vLLM server")
    if not start_server():
        sys.exit(1)
    
    # Step 3: Wait for server
    log("=" * 50)
    log("STEP: Waiting for server (up to 10 min)")
    time.sleep(30)  # Give it a moment before first check
    if not wait_for_server(600):
        log("Server failed to start. Check vllm_serve.log")
        sys.exit(1)
    
    # Step 4: Check NPU memory
    check_npu_memory()
    
    # Step 5: Test inference
    log("=" * 50)
    log("STEP: Testing inference")
    reply = test_inference()
    
    if reply:
        log("=" * 50)
        log("ALL TESTS PASSED!")
    else:
        log("=" * 50)
        log("INFERENCE FAILED - see logs above")