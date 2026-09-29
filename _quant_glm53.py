#!/usr/bin/env python3
"""
Quantize GLM-5.3-Flash to W8A8 using msmodelslim.
Run after download completes.
"""
import os, sys, json, subprocess, time, glob

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "models", "GLM-5.3-Flash")
SAVE_DIR = os.path.join(BASE, "models", "GLM-5.3-Flash-W8A8")
YAML_PATH = "/opt/mamba/lib/python3.13/site-packages/msmodelslim/lab_practice/glm_5/glm_5_w8a8.yaml"
LOG_FILE = os.path.join(BASE, "quant_glm.log")

def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    print(msg)

def check_model():
    """Check if model is fully downloaded."""
    if not os.path.exists(MODEL_DIR):
        log(f"ERROR: Model dir {MODEL_DIR} not found!")
        return False
    config_file = os.path.join(MODEL_DIR, "config.json")
    if not os.path.exists(config_file):
        log(f"ERROR: config.json not found!")
        return False
    # Count complete safetensors files
    safetensors = glob.glob(os.path.join(MODEL_DIR, "*.safetensors"))
    incomplete = glob.glob(os.path.join(MODEL_DIR, "*.safetensors.incomplete"))
    total_size = sum(os.path.getsize(f) for f in safetensors) / 1024**3
    log(f"Complete shards: {len(safetensors)}, Incomplete: {len(incomplete)}")
    log(f"Downloaded: {total_size:.1f} GB")
    # Check if all 62 shards are done
    if len(incomplete) > 0:
        log("Still downloading. Waiting...")
        return False
    log("Model fully downloaded!")
    return True

def run_quant():
    """Run msmodelslim quantization."""
    log("Starting W8A8 quantization...")
    log(f"Model: {MODEL_DIR}")
    log(f"Config: {YAML_PATH}")
    log(f"Save to: {SAVE_DIR}")
    log(f"Device: npu")

    cmd = [
        sys.executable, "-m", "msmodelslim", "quant",
        "--model_path", MODEL_DIR,
        "--save_path", SAVE_DIR,
        "--config_path", YAML_PATH,
        "--device", "npu",
        "--trust_remote_code", "True",
    ]
    
    log(f"Command: {' '.join(cmd)}")
    log("Quantization started (this may take several hours)...")
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=86400)
    
    with open(LOG_FILE, "a") as f:
        f.write("\n=== STDOUT ===\n")
        f.write(result.stdout[-5000:])
        f.write("\n=== STDERR ===\n")
        f.write(result.stderr[-5000:])
    
    if result.returncode == 0:
        log("Quantization SUCCEEDED!")
        # Check output
        out_files = glob.glob(os.path.join(SAVE_DIR, "**"), recursive=True)
        out_size = sum(os.path.getsize(f) for f in out_files if os.path.isfile(f)) / 1024**3
        log(f"Output size: {out_size:.1f} GB")
        return True
    else:
        log(f"Quantization FAILED with code {result.returncode}")
        log(f"Last stderr: {result.stderr[-1000:]}")
        return False

if __name__ == "__main__":
    if not check_model():
        sys.exit(1)
    success = run_quant()
    sys.exit(0 if success else 1)