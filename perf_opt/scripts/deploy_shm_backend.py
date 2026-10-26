#!/usr/bin/env python3
"""Deploy ShmBackend into the vllm-ascend installation.

This script:
1. Copies shm_backend.py into the ascend_store backend directory.
2. Registers the "shm" backend key in pool_worker.py's backend_map.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


def find_vllm_ascend_root() -> Path:
    """Locate the vllm_ascend package directory."""
    spec = importlib.util.find_spec("vllm_ascend")
    if spec is None or spec.origin is None:
        print("ERROR: vllm_ascend package not found!", file=sys.stderr)
        sys.exit(1)
    return Path(spec.origin).parent


def copy_backend(src: Path, backend_dir: Path) -> None:
    """Copy shm_backend.py to the backend directory."""
    dst = backend_dir / "shm_backend.py"
    import shutil
    shutil.copy2(str(src), str(dst))
    print(f"[OK] Copied {src.name} -> {dst}")


def register_in_backend_map(pool_worker_path: Path) -> None:
    """Add 'shm' entry to backend_map in pool_worker.py."""
    content = pool_worker_path.read_text()

    # Check if already registered
    if '"shm"' in content:
        print("[SKIP] 'shm' already in backend_map")
        return

    # Find the backend_map dictionary and add shm entry
    # Look for the last backend entry and add after it
    marker = '"yuanrong": {'
    if marker not in content:
        print("ERROR: Cannot find backend_map marker!", file=sys.stderr)
        sys.exit(1)

    shm_entry = '''
    "shm": {
        "name": "ShmBackend",
        "path": "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.shm_backend",
    },'''

    new_content = content.replace(
        marker + '\n        "name": "YuanrongBackend",\n'
        '        "path": "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.yuanrong_backend",\n'
        "    },",
        marker + '\n        "name": "YuanrongBackend",\n'
        '        "path": "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.yuanrong_backend",\n'
        "    },"
        + shm_entry,
    )

    if new_content == content:
        print("ERROR: Failed to modify backend_map!", file=sys.stderr)
        sys.exit(1)

    # Backup
    bak = pool_worker_path.with_suffix(".py.bak")
    import shutil
    shutil.copy2(str(pool_worker_path), str(bak))
    print(f"[OK] Backup saved to {bak}")

    pool_worker_path.write_text(new_content)
    print(f"[OK] Registered 'shm' in backend_map")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    src = script_dir / "shm_backend.py"
    if not src.exists():
        print(f"ERROR: {src} not found!", file=sys.stderr)
        sys.exit(1)

    vllm_ascend_root = find_vllm_ascend_root()
    backend_dir = (
        vllm_ascend_root
        / "distributed"
        / "kv_transfer"
        / "kv_pool"
        / "ascend_store"
        / "backend"
    )
    pool_worker_path = (
        vllm_ascend_root
        / "distributed"
        / "kv_transfer"
        / "kv_pool"
        / "ascend_store"
        / "pool_worker.py"
    )

    assert backend_dir.is_dir(), f"Backend dir not found: {backend_dir}"
    assert pool_worker_path.is_file(), f"pool_worker.py not found: {pool_worker_path}"

    copy_backend(src, backend_dir)
    register_in_backend_map(pool_worker_path)

    print("\n=== Deployment complete ===")
    print("You can now use backend='shm' with AscendStoreConnector.")
    print("Set environment VLLM_KV_SHM_DIR to change the shared directory.")
    print("Default: /dev/shm/vllm_kv_transfer")


if __name__ == "__main__":
    main()