"""
Load GLM-5.3-Flash with Transformers + PyTorch NPU, with OOM protection.
Uses accelerate device_map="auto" + CPU offloading to handle 305 GB model.
"""
import os, sys, json, gc, time
import torch
import torch_npu
import psutil

# ─── Config ─────────────────────────────────────────────────────
MODEL_DIR = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/GLM-5.3-Flash"
OFFLOAD_DIR = "/tmp/offload_glm53"
MAX_MEMORY_PER_NPU = "58GiB"  # leave 6 GiB headroom per card
# ────────────────────────────────────────────────────────────────

os.makedirs(OFFLOAD_DIR, exist_ok=True)

def print_mem(msg="Memory"):
    """Print NPU and CPU memory usage."""
    npu_str = "N/A"
    try:
        import subprocess
        r = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=10)
        # Simple: count lines with MiB
        for line in r.stdout.split("\n"):
            if "Memory-Usage" in line:
                parts = line.split()
                npu_str = parts[-1] if parts else "N/A"
                break
    except Exception:
def try_load_model():
    """Try loading with OOM protection, return (model, tokenizer) or None."""
    print("=" * 60)
    print("GLM-5.3-Flash: Attempting to load with Transformers + NPU")
    print("=" * 60)
    print_mem("before load")

    # Build max_memory: 4 NPUs + CPU spill
    max_memory = {i: MAX_MEMORY_PER_NPU for i in range(4)}
    max_memory["cpu"] = "120GiB"

    start = time.time()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

        print(f"\n[1/4] Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
        print(f"  ✅ Vocab size: {tokenizer.vocab_size}")

        print(f"\n[2/4] Loading config...")
        config = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
        print(f"  ✅ Architecture: {config.architectures}")

        print(f"\n[3/4] Loading model (~305 GB, device_map='auto', per NPU={MAX_MEMORY_PER_NPU})...")
        print(f"  Offload: {OFFLOAD_DIR}")
        sys.stdout.flush()

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            trust_remote_code=True,
            device_map="auto",
            max_memory=max_memory,
            offload_folder=OFFLOAD_DIR,
            offload_state_dict=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

        elapsed = time.time() - start
        print(f"\n  ✅ Model loaded in {elapsed:.0f}s!")
        print_mem("after load")

        print(f"\n[4/4] Device map:")
        if hasattr(model, "hf_device_map"):
            for name, dev in model.hf_device_map.items():
                print(f"  {name}: {dev}")

        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n  Total params: {total_params/1e9:.2f}B")
        return model, tokenizer

    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "OOM" in str(e):
            print(f"\n  ❌ OOM: {e}")
        else:
            print(f"\n  ❌ RuntimeError: {e}")
    except Exception as e:
        print(f"\n  ❌ {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n  ⏱  Failed after {time.time()-start:.0f}s")
    return None, None


def run_inference(model, tokenizer):
    """Simple test inference."""
    print("\n" + "=" * 60)
    print("Testing inference...")
    print("=" * 60)

    prompt = "Hello! Please introduce yourself briefly."
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print(f"  Prompt: {prompt}")
    print_mem("before inference")

    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=50,
                do_sample=False, temperature=None, top_p=None,
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n  Response: {response}")
        print_mem("after inference")
        return True
    except Exception as e:
        print(f"\n  ❌ Inference: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import shutil
    if os.path.exists(OFFLOAD_DIR):
        shutil.rmtree(OFFLOAD_DIR)

    model, tokenizer = try_load_model()

    if model is not None:
        run_inference(model, tokenizer)
    else:
        print("\n❌ Model loading failed.")

    print_mem("final")
    print("\nDone.")
        pass
    cpu = psutil.virtual_memory()
    print(f"  [{msg}] NPU mem: {npu_str} | CPU: {cpu.used/1e9:.0f}G/{cpu.total/1e9:.0f}G ({cpu.percent}%)")