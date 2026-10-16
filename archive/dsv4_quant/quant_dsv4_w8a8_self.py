#!/usr/bin/env python3
"""
Self-quantize DeepSeek-V4-Flash (BF16) → W8A8 using msmodelslim.

Follows the official quantization method from the model's README:
  https://gitcode.com/Ascend/msmodelslim/tree/master/example/DeepSeek#deepseek-v4-flash含mtp层-w8a8-动态量化

Usage:
  python3 quant_dsv4_w8a8_self.py                    # Use default paths
  python3 quant_dsv4_w8a8_self.py --dry-run           # Print command only
  python3 quant_dsv4_w8a8_self.py --model-path ... --save-path ...
"""

import argparse
import subprocess
import sys
import os

# Default paths (change as needed)
DEFAULT_MODEL_PATH = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash"
DEFAULT_SAVE_PATH = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash-w8a8-self"


def build_command(model_path: str, save_path: str) -> list[str]:
    return [
        "msmodelslim", "quant",
        "--model_path", model_path,
        "--save_path", save_path,
        "--model_type", "DeepSeek-V4-Flash",
        "--quant_type", "w8a8",
        "--trust_remote_code", "True",
    ]


def main():
    parser = argparse.ArgumentParser(description="Quantize DeepSeek-V4-Flash BF16 → W8A8")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH,
                        help=f"Path to BF16 model (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--save-path", default=DEFAULT_SAVE_PATH,
                        help=f"Path to save quantized model (default: {DEFAULT_SAVE_PATH})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print command and exit without running")
    args = parser.parse_args()

    # Validate model path exists
    if not os.path.isdir(args.model_path):
        print(f"[ERROR] Model path does not exist: {args.model_path}")
        # List available models
        base_dir = os.path.dirname(args.model_path)
        if os.path.isdir(base_dir):
            print(f"  Available models in {base_dir}:")
            for d in sorted(os.listdir(base_dir)):
                if os.path.isdir(os.path.join(base_dir, d)):
                    print(f"    - {d}")
        sys.exit(1)

    # Build and print command
    cmd = build_command(args.model_path, args.save_path)
    print("=" * 70)
    print("  DeepSeek-V4-Flash W8A8 Self-Quantization")
    print("=" * 70)
    print(f"  Input:   {args.model_path}")
    print(f"  Output:  {args.save_path}")
    print(f"  Model:   DeepSeek-V4-Flash")
    print(f"  Quant:   w8a8 (dynamic per-channel)")
    print(f"  Tool:    msmodelslim")
    print()
    print(f"  Command:")
    print(f"    {' '.join(cmd)}")
    print("=" * 70)

    if args.dry_run:
        print("[DRY-RUN] Exiting without quantizing.")
        sys.exit(0)

    # Confirm
    print()
    print(f"[INFO] Starting quantization...")
    print(f"[INFO] This may take a while (model is ~300GB, needs CPU memory)")
    sys.stdout.flush()

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        print()
        print("=" * 70)
        print("  ✅ Quantization completed successfully!")
        print(f"  Output saved to: {args.save_path}")
        print("=" * 70)
    else:
        print()
        print(f"[ERROR] Quantization failed with return code {result.returncode}")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()