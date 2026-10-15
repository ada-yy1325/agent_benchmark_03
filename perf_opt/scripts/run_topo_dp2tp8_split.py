#!/usr/bin/env python3
"""DP2xTP8 via two independent TP8 servers."""
import subprocess, time, json, os, sys
from pathlib import Path

TOPO = "dp2tp8"
TP = 8
BENCH = Path(__file__).resolve().parent / "bench_serving.py"
SERVED = "dsv4"
MODEL_DIR = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash-w8a8-self"
OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "topo" / TOPO
WORKDIR = str(Path(__file__).resolve().parent.parent.parent)  # project root
START_SCRIPT = str(Path(__file__).resolve().parent / "start_server_isolated.py")

import urllib.request

CONCURRENCIES = [1, 8, 32, 64]
INP = 4096
OUT = 1024
NREQ = 128
BATCHED = 8192
MAX_SEQS = 64

PORTS = [8000, 8001]
DEVICE_RANGES = ["0,1,2,3,4,5,6,7", "8,9,10,11,12,13,14,15"]


def kill_servers():
    print("  [KILL] Stopping old vLLM...", flush=True)
    os.system("pkill -9 -f VLLM 2>/dev/null")
    time.sleep(3)
    for port in PORTS:
        os.system(f"fuser -k {port}/tcp 2>/dev/null")
    time.sleep(2)


def start_server(rank, devices, port):
    name = f"topo_dp{rank}"
    os.system(f"tmux kill-session -t {name} 2>/dev/null")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = OUT_DIR / f"_server_dp{rank}.log"
    wrapper = OUT_DIR / f"_start_dp{rank}.sh"
    with open(wrapper, "w") as f:
        f.write(f"""#!/bin/bash
set -e
export PATH=/opt/mamba/bin:$PATH
export ASCEND_RT_VISIBLE_DEVICES={devices}
cd {WORKDIR}
exec python3 {START_SCRIPT} --port {port} --batched {BATCHED} --max-seqs {MAX_SEQS} 2>&1 | tee {log_file}
""")
    os.chmod(wrapper, 0o755)
    os.system(f"tmux new-session -d -s {name} bash {wrapper}")
    print(f"  [START] tmux:{name} devices={devices} port={port}", flush=True)
    time.sleep(3)


def wait_healthy(port, timeout=600):
    url = f"http://127.0.0.1:{port}/health"
    print(f"  [WAIT] :{port}/health (timeout={timeout}s)...", flush=True)
    for i in range(1, timeout + 1):
        try:
            urllib.request.urlopen(url, timeout=3)
            print(f"  [OK] port {port} healthy after {i}s", flush=True)
            return True
        except Exception:
            if i % 30 == 0:
                print(f"  [WAIT] port {port}: {i}s...", flush=True)
            time.sleep(1)
    print(f"  [FAIL] port {port} health timeout", flush=True)
    return False


def main():
    print(f"\n{'='*60}")
    print(f"  Phase 3 - DP2xTP8 (2 independent TP8 servers)")
    print(f"{'='*60}\n", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    kill_servers()

    t_start = time.time()
    for rank in range(2):
        print(f"\n  [LAUNCH] Server DP{rank}...", flush=True)
        start_server(rank, DEVICE_RANGES[rank], PORTS[rank])

    for rank in range(2):
        if not wait_healthy(PORTS[rank], 600):
            print(f"  [ABORT] Server DP{rank} failed", flush=True)
            return

    load_time = time.time() - t_start
    print(f"\n  [INFO] Both servers ready in {load_time:.0f}s", flush=True)

    all_results = []
    for idx, c in enumerate(CONCURRENCIES):
        print(f"\n  {'-'*50}")
        print(f"  >> c={c} ({idx+1}/{len(CONCURRENCIES)})", flush=True)

        # Launch benchmark procs for both ports
        procs = {}
        for port in PORTS:
            label = f"{TOPO}_port{port}_c{c}_{INP}k"
            cmd = [
                "/opt/mamba/bin/python3", str(BENCH),
                "--served-name", SERVED,
                "--model-dir", MODEL_DIR,
                "--base-url", f"http://127.0.0.1:{port}/v1",
                "--input-len", str(INP),
                "--output-len", str(OUT),
                "--concurrency", str(c),
                "--num-requests", str(NREQ // 2),
                "--out", str(OUT_DIR / f"{label}.json"),
                "--case-id", label,
                "--npu-stats", str(OUT_DIR / f"{label}_npu.csv"),
                "--temp", "0.0", "--seed", "1",
            ]
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True)
            procs[port] = (p, label)

        # Wait for both in parallel via threads
        import threading
        port_results = []
        lock = threading.Lock()

        def wait_one(port, p, label):
            try:
                stdout, stderr = p.communicate(timeout=7200)
                if p.returncode != 0:
                    err_file = OUT_DIR / f"{label}_err.txt"
                    err_file.write_text(stderr)
                    with lock:
                        port_results.append({"port": port, "success": False,
                                             "error": f"rc={p.returncode}", "stderr": stderr[:500]})
                else:
                    data = json.loads((OUT_DIR / f"{label}.json").read_text())
                    with lock:
                        port_results.append({
                            "port": port,
                            "ttft_p50": data["ttft_ms"]["p50"],
                            "ttft_p99": data["ttft_ms"]["p99"],
                            "tpot_p50": data["tpot_ms"]["p50"],
                            "tp": data["output_tokens_per_s"],
                            "ok": data["successful_requests"],
                            "total": data["planned_requests"],
                            "success": True,
                        })
            except subprocess.TimeoutExpired:
                p.kill()
                p.communicate()
                with lock:
                    port_results.append({"port": port, "success": False,
                                         "error": "timeout 7200s"})
            except Exception as e:
                with lock:
                    port_results.append({"port": port, "success": False,
                                         "error": str(e)})

        threads = []
        t0 = time.time()
        for port, (p, label) in procs.items():
            t = threading.Thread(target=wait_one, args=(port, p, label), daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        wall = time.time() - t0

        ok_results = [r for r in port_results if r.get("success")]
        if ok_results:
            total_tp = sum(r["tp"] for r in ok_results)
            avg_ttft_p50 = sum(r["ttft_p50"] for r in ok_results) / len(ok_results)
            avg_ttft_p99 = sum(r["ttft_p99"] for r in ok_results) / len(ok_results)
            avg_tpot = sum(r["tpot_p50"] for r in ok_results) / len(ok_results)
            total_ok = sum(r["ok"] for r in ok_results)
            total_req = sum(r["total"] for r in ok_results)
            r = {
                "concurrency": c,
                "ttft_p50": avg_ttft_p50,
                "ttft_p99": avg_ttft_p99,
                "tpot_p50": avg_tpot,
                "tp": total_tp,
                "ok": total_ok,
                "total": total_req,
                "wall_s": round(wall),
                "success": True,
                "detail": port_results,
            }
        else:
            r = {"concurrency": c, "success": False,
                 "error": "all ports failed", "wall_s": round(wall),
                 "detail": port_results}
        all_results.append(r)

        if r.get("success"):
            print(f"  [c{c}] TTFT p50={r['ttft_p50']:.0f}ms p99={r['ttft_p99']:.0f}ms | "
                  f"TP={r['tp']:.1f} tok/s | TPOT={r['tpot_p50']:.1f}ms | "
                  f"ok={r['ok']}/{r['total']} | wall={r['wall_s']}s", flush=True)
        else:
            print(f"  [c{c}] FAIL: {r.get('error','')}", flush=True)

    summary = {
        "topo": TOPO, "tp_per_server": TP, "num_servers": 2,
        "batched": BATCHED, "max_seqs": MAX_SEQS,
        "input_len": INP, "output_len": OUT,
        "load_time_s": int(load_time),
        "results": all_results,
    }
    with open(OUT_DIR / "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  [SUMMARY] -> {OUT_DIR / '_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
