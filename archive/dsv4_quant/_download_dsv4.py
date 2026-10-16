import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=['download', 'check_config', 'start_server'], required=True)
    args = parser.parse_args()
    
    if args.action == 'download':
        from modelscope.hub.snapshot_download import snapshot_download
        import os
        MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash"
        LOCAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "DeepSeek-V4-Flash")
        os.makedirs(os.path.dirname(LOCAL_DIR), exist_ok=True)
        print(f"Downloading {MODEL_ID} -> {LOCAL_DIR}")
        snapshot_download(MODEL_ID, local_dir=LOCAL_DIR)
        print("Done!")
    
    elif args.action == 'check_config':
        import json, os
        BASE = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(BASE, "models", "DeepSeek-V4-Flash", "config.json")
        if os.path.exists(path):
            with open(path) as f:
                cfg = json.load(f)
            for k in ['model_type', 'quantization_config', 'expert_dtype', 'num_experts', 'num_activated_experts', 'hidden_size', 'intermediate_size', 'num_hidden_layers']:
                if k in cfg:
                    print(f"{k}: {cfg[k]}")
        else:
            print(f"config.json not found at {path}")
    
    elif args.action == 'start_server':
        import subprocess, sys
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "./models/DeepSeek-V4-Flash",
            "--port", "8801",
            "--trust-remote-code",
            "--enforce-eager",
            "--gpu-memory-utilization", "0.9",
            "--max-model-len", "4096",
            "--served-model-name", "DeepSeek-V4-Flash",
        ]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd)

if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""Download DeepSeek V4 Flash from ModelScope."""
import subprocess, sys, os

MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash"
LOCAL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models", "DeepSeek-V4-Flash"
)

print(f"=== Downloading {MODEL_ID} → {LOCAL_DIR} ===")
print(f"Disk: {os.popen('df -h /inspire/').read().strip()}")
sys.stdout.flush()

from modelscope.hub.snapshot_download import snapshot_download
snapshot_download(MODEL_ID, local_dir=LOCAL_DIR)

print("=== Download complete! ===")
print(os.popen(f'du -sh {LOCAL_DIR}').read().strip())