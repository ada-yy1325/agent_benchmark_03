"""
Test whether 16 NPU devices can be opened simultaneously via multiprocessing.
This reproduces the vLLM worker scenario to check if the issue is hardware/driver-level.
"""
import torch
import multiprocessing as mp
import time
import os

def worker(device_id, ready_event, result_queue):
    """Worker process: try to set_device and report success/failure."""
    pid = os.getpid()
    try:
        torch.npu.set_device(device_id)
        ready_event.set()  # Signal that device is acquired
        # Hold the device for a bit to simulate vLLM worker lifetime
        time.sleep(2)
        result_queue.put((device_id, pid, "OK"))
    except Exception as e:
        result_queue.put((device_id, pid, f"FAIL: {str(e)[:100]}"))

def test_parallel_npu(n_devices=16):
    """Spawn n_devices workers in parallel, each trying to use one NPU."""
    print(f"Testing {n_devices} NPU devices in parallel...")
    
    ctx = mp.get_context("spawn")
    events = []
    result_queue = ctx.Queue()
    processes = []
    
    for i in range(n_devices):
        ev = ctx.Event()
        p = ctx.Process(target=worker, args=(i, ev, result_queue))
        p.start()
        events.append(ev)
        processes.append(p)
    
    # Wait up to 30s for all events to be set
    success = 0
    failed = 0
    for i, ev in enumerate(events):
        if ev.wait(timeout=30):
            success += 1
        else:
            failed += 1
            print(f"  NPU {i}: TIMEOUT (worker did not set device in 30s)")
    
    # Collect results
    results = {}
    for _ in range(n_devices):
        try:
            dev_id, pid, status = result_queue.get(timeout=5)
            results[dev_id] = (pid, status)
        except Exception:
            break
    
    print(f"\nResults: {success}/{n_devices} devices acquired")
    for dev_id in sorted(results.keys()):
        pid, status = results[dev_id]
        print(f"  NPU {dev_id} (PID {pid}): {status}")
    
    for p in processes:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()
    
    return success == n_devices

if __name__ == "__main__":
    test_parallel_npu(16)