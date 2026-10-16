#!/usr/bin/env python3
"""Download GLM-5.3-Flash with max speed & resume support."""
import os, sys, time, glob, subprocess

BASE = os.path.dirname(os.path.abspath(__file__))
LOCAL_DIR = os.path.join(BASE, "models", "GLM-5.3-Flash")
LOG_FILE = os.path.join(BASE, "download_glm.log")

def log(msg):
    ts = time.strftime('%H:%M:%S')
    with open(LOG_FILE, "a") as f:
        f.write(f"[{ts}] {msg}\n")
    print(f"[{ts}] {msg}")

def check_existing():
    total, files, incomplete = 0, 0, 0
    if not os.path.exists(LOCAL_DIR):
        return 0, 0, 0
    for root, dirs, fnames in os.walk(LOCAL_DIR):
        for f in fnames:
            fp = os.path.join(root, f)
            if f.endswith((".incomplete", ".aria2")):
                incomplete += 1
            elif f.endswith((".safetensors", ".bin", ".pt")):
                total += os.path.getsize(fp)
                files += 1
    return total, files, incomplete

def try_hf_mirror():
    """Try downloading from HuggingFace mirror (fastest)."""
    # List of possible HF model names for GLM-5.3-Flash
    candidates = [
        "THUDM/GLM-5.3-Flash",
        "zai-org/GLM-5.3-Flash",
        "THUDM/glm-5.3-flash",
    ]
    
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    
    from huggingface_hub import HfApi, snapshot_download
    
    for model_id in candidates:
        try:
            log(f"Checking HF: {model_id}")
            info = HfApi().model_info(model_id, timeout=10)
            log(f"FOUND on HF: {model_id} ({len(info.siblings)} files)")
            log(f"Downloading from hf-mirror.com (Rust accelerator)...")
            result = snapshot_download(
                model_id,
                local_dir=LOCAL_DIR,
                local_dir_use_symlinks=False,
                resume_download=True,
                max_workers=8,
            )
            log(f"HF download complete: {result}")
            return True
        except Exception as e:
            log(f"  Not on HF: {model_id} ({type(e).__name__})")
    return False

def try_modelscope():
    """Download from ModelScope (reliable fallback)."""
    log("Downloading from ModelScope (8-thread parallel)...")
    from modelscope.hub.snapshot_download import snapshot_download
    result = snapshot_download(
        "zai-org/GLM-5.3-Flash",
        local_dir=LOCAL_DIR,
    )
    log(f"ModelScope download complete: {result}")
    return True

if __name__ == "__main__":
    log("=" * 50)
    # Check existing
    size, files, incomplete = check_existing()
    log(f"Existing: {files} files ({size/1024**3:.1f} GB), {incomplete} incomplete")
    if incomplete == 0 and files > 0:
        log("Model already complete, skipping download")
        sys.exit(0)
    
    os.makedirs(LOCAL_DIR, exist_ok=True)
    
    # Strategy 1: HF mirror (fastest)
    if try_hf_mirror():
        success = True
    else:
        # Strategy 2: ModelScope
        log("HF not available, using ModelScope...")
        try:
            try_modelscope()
            success = True
        except Exception as e:
            log(f"FAILED: {e}")
            success = False
    
    size, files, incomplete = check_existing()
    log(f"FINAL: {files} files, {size/1024**3:.1f} GB, {incomplete} pending")
    log("SUCCESS" if success and incomplete == 0 else "INCOMPLETE")