#!/usr/bin/env python3
"""
Fast download GLM-5.3-Flash using huggingface_hub with hf-mirror.com.
Resumable - safe to interrupt and resume.
"""
import os, sys, time, subprocess, shutil

BASE = os.path.dirname(os.path.abspath(__file__))
LOCAL_DIR = os.path.join(BASE, "models", "GLM-5.3-Flash")
LOG_FILE = os.path.join(BASE, "download_glm.log")

# Try Hugging Face orgs - model might be under THUDM or zai-org
HF_ORGS_TO_TRY = [
    "THUDM/GLM-5.3-Flash",
    "zai-org/GLM-5.3-Flash",
    "zai-org/GLM-5.3-Flash-Instruct",
]

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # Use Rust-based fast downloader

def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def find_model():
    """Try to find model on HF mirror."""
    from huggingface_hub import HfApi, HfFolder
    api = HfApi()
    for model_id in HF_ORGS_TO_TRY:
        try:
            info = api.model_info(model_id, timeout=15)
            size = sum(f.size for f in info.siblings) if info.siblings else 0
            log(f"Found: {model_id} ({size/1024**3:.1f} GB, {len(info.siblings)} files)")
            return model_id
        except Exception as e:
            log(f"Not found: {model_id} ({e})")
    return None

def download_fast(model_id):
    """Download using huggingface_hub with multi-threading."""
    log(f"Downloading {model_id} to {LOCAL_DIR}")
    os.makedirs(LOCAL_DIR, exist_ok=True)
    
    start = time.time()
    
    from huggingface_hub import snapshot_download
    
    result = snapshot_download(
        model_id,
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=8,  # 8 parallel threads
        ignore_patterns=["*.h5", "*.ot"],  # skip irrelevant files
    )
    
    elapsed = time.time() - start
    return result, elapsed

def download_fallback():
    """Fallback to modelscope if HF fails."""
    log("Falling back to ModelScope download...")
    from modelscope.hub.snapshot_download import snapshot_download
    start = time.time()
    result = snapshot_download(
        "zai-org/GLM-5.3-Flash",
        local_dir=LOCAL_DIR,
    )
    elapsed = time.time() - start
    return result, elapsed

def count_model_size():
    """Count downloaded model files."""
    total = 0
    files = 0
    incomplete = 0
    for root, dirs, fnames in os.walk(LOCAL_DIR):
        for f in fnames:
            fp = os.path.join(root, f)
            if f.endswith(".incomplete"):
                incomplete += 1
            elif f.endswith(".safetensors") or f.endswith(".bin") or f.endswith(".pt"):
                total += os.path.getsize(fp)
                files += 1
    return total, files, incomplete

if __name__ == "__main__":
    # Check if already downloaded
    if os.path.exists(LOCAL_DIR):
        size, files, incomplete = count_model_size()
        log(f"Existing: {size/1024**3:.1f} GB, {files} files, {incomplete} incomplete")
        if incomplete == 0 and files > 0:
            log("Model already complete!")
            sys.exit(0)
        elif size > 0:
            log(f"Resuming download ({size/1024**3:.1f} GB existing)...")
    
    # Step 1: Find model on HF mirror
    log("Searching for model on Hugging Face mirror (hf-mirror.com)...")
    model_id = find_model()
    
    if model_id:
        log(f"Using HF download: {model_id}")
        try:
            result, elapsed = download_fast(model_id)
            log(f"Download completed in {elapsed/60:.1f} min! Saved to {result}")
        except Exception as e:
            log(f"HF download failed: {e}")
            log("Trying ModelScope fallback...")
            result, elapsed = download_fallback()
            log(f"Download completed in {elapsed/60:.1f} min! Saved to {result}")
    else:
        log("Model not on HF mirror, using ModelScope...")
        result, elapsed = download_fallback()
        log(f"Download completed in {elapsed/60:.1f} min! Saved to {result}")
    
    # Final size check
    size, files, incomplete = count_model_size()
    log(f"Final: {files} files, {size/1024**3:.1f} GB")
    if incomplete > 0:
        log(f"WARNING: {incomplete} incomplete files remain!")