#!/usr/bin/env python3
"""
Phase 1.2: 4K 档矩阵压测 — 断点续跑脚本

遍历 concurrency ∈ [1, 2, 4, 8, 16, 32, 64]
每点独立 JSON，已完成自动跳过。

运行方式：
    cd /inspire/.../perf_opt
    python3 run_matrix_4k.py
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime

# ── 基础路径 ──────────────────────────────────────────
BASE_DIR = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/perf_opt"
RESULTS_DIR = os.path.join(BASE_DIR, "results", "matrix")
NPU_DIR = os.path.join(BASE_DIR, "npu_stats")
LOG_DIR = os.path.join(BASE_DIR, "server_logs")
BENCH_SCRIPT = os.path.join(BASE_DIR, "scripts", "bench_serving.py")

# ── 固定参数 ──────────────────────────────────────────
BASE_URL = "http://127.0.0.1:8000/v1"
SERVED_NAME = "dsv4"
MODEL_DIR = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash-w8a8-self/"
INPUT_LEN = 4096
OUTPUT_LEN = 1024
TEMP = 0.0
SEED = 42


def clamp_requests(c):
    return max(min(c * 4, 128), 4)


# ── 矩阵定义 ──────────────────────────────────────────
MATRIX = [
    # (concurrency, planned_requests, case_id)
    (1, clamp_requests(1), "4k_out1024_c1"),
    (2, clamp_requests(2), "4k_out1024_c2"),
    (4, clamp_requests(4), "4k_out1024_c4"),
    (8, clamp_requests(8), "4k_out1024_c8"),
    (16, clamp_requests(16), "4k_out1024_c16"),
    (32, clamp_requests(32), "4k_out1024_c32"),
    (64, clamp_requests(64), "4k_out1024_c64"),
]

LOG_PATH = os.path.join(LOG_DIR, f"matrix_4k_{datetime.now():%Y%m%d_%H%M%S}.log")


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def is_done(case_id, planned):
    """检查是否已成功完成（JSON 文件存在且 successful_requests == planned）"""
    json_path = os.path.join(RESULTS_DIR, f"{case_id}.json")
    if not os.path.exists(json_path):
        return False
    try:
        with open(json_path) as f:
            d = json.load(f)
        return d.get("successful_requests", 0) == planned
    except Exception:
        return False


def run_case(c, planned, case_id):
    json_path = os.path.join(RESULTS_DIR, f"{case_id}.json")
    npu_path = os.path.join(NPU_DIR, f"{case_id}.csv")

    cmd = [
        sys.executable, BENCH_SCRIPT,
        "--base-url", BASE_URL,
        "--served-name", SERVED_NAME,
        "--model-dir", MODEL_DIR,
        "--input-len", str(INPUT_LEN),
        "--output-len", str(OUTPUT_LEN),
        "--concurrency", str(c),
        "--num-requests", str(planned),
        "--case-id", case_id,
        "--out", json_path,
        "--npu-stats", npu_path,
        "--temp", str(TEMP),
        "--seed", str(SEED),
    ]

    log(f"▶ 开始: {case_id} (c={c}, planned={planned})")
    log(f"  命令: {' '.join(cmd[:8])} ...")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True, timeout=7200)
    elapsed = time.time() - t0

    if result.returncode == 0:
        success = "OK"
    else:
        success = "FAIL"

    log(f"  结果: {success} | 耗时: {elapsed:.0f}s | returncode={result.returncode}")


    if result.returncode != 0:
        log(f"  ⚠ {case_id} 失败，继续下一个")
        return False

    # 验证输出 JSON
    try:
        with open(json_path) as f:
            d = json.load(f)
        ok = d.get("successful_requests", 0)
        planned_req = d.get("planned_requests", 0)
        out_tok_s = d.get("output_tokens_per_s", 0)
        ttft_p50 = d.get("ttft_ms", {}).get("p50", 0)
        tpot_p50 = d.get("tpot_ms", {}).get("p50", 0)
        log(f"  数据: ok={ok}/{planned_req}, out_tok/s={out_tok_s}, TTFT_p50={ttft_p50}ms, TPOT_p50={tpot_p50}ms")
        return ok == planned
    except Exception as e:
        log(f"  ⚠  JSON 验证失败: {e}")
        return False


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(NPU_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    log("=" * 60)
    log("Phase 1.2: 4K 档矩阵压测 启动")
    log(f"输入={INPUT_LEN}, 输出={OUTPUT_LEN}, 并发={[c for c,_,_ in MATRIX]}")
    log(f"日志: {LOG_PATH}")
    log("=" * 60)

    completed = 0
    skipped = 0
    failed = 0
    stopped_early = False

    for c, planned, case_id in MATRIX:
        if is_done(case_id, planned):
            log(f"⏭ 跳过: {case_id} 已完成 (successful=={planned})")
            skipped += 1
            continue

        # 如果前一个点成功率 < 95%，停止后续
        if stopped_early:
            log(f"⏭ 跳过: {case_id} 已触发停止规则")
            break

        ok = run_case(c, planned, case_id)
        if ok:
            completed += 1

            # 检查是否需要停止
            json_path = os.path.join(RESULTS_DIR, f"{case_id}.json")
            try:
                with open(json_path) as f:
                    d = json.load(f)
                success_rate = d.get("successful_requests", 0) / max(1, d.get("planned_requests", 1))
                if success_rate < 0.95:
                    log(f"  🛑 停止规则触发: success_rate={success_rate:.2f} < 0.95, 停止更大并发")
                    stopped_early = True
            except:
                pass
        else:
            failed += 1
            log(f"  ⚠ 是否继续？出现失败，检查是否应该停止...")

    log("=" * 60)
    log(f"Phase 1.2 完成: 已跑={completed}, 跳过={skipped}, 失败={failed}, 提前停止={stopped_early}")
    log(f"产品目录: {RESULTS_DIR}/")
    log(f"日志: {LOG_PATH}")
    log("=" * 60)


if __name__ == "__main__":
    main()