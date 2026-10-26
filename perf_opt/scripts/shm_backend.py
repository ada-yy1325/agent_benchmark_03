"""Shared-memory backend for AscendStoreConnector.

Uses files in a shared directory (default /dev/shm/vllm_kv_transfer) for
KV cache exchange between prefill and decode instances on the same machine.

Data flow
---------
- **put**: NPU -> CPU (via torch.ops._C_ascend.swap_blocks_batch,
  aclrtMemcpyBatchAsync) -> shared file.
- **get**: shared file -> CPU -> NPU (same DMA op reversed).

Each block is atomically written (tmp + rename) so the consumer never sees
a partial file.  After the consumer successfully loads a file it deletes it
to reclaim space.
"""

from __future__ import annotations

import hashlib
import os
import struct
from typing import TYPE_CHECKING

import numpy as np
import torch
from vllm.logger import logger

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.backend import (
    Backend,
)

if TYPE_CHECKING:
    from vllm.config import ParallelConfig

# Keep these in sync with vllm_ascend.simple_kv_offload.npu_mem_ops
_DIRECTION_H2D = 0  # host -> device
_DIRECTION_D2H = 1  # device -> host

_HEADER_FMT = "=QQ"  # num_layers (uint64), total_bytes (uint64)
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)

_MAX_BATCH_COPY = 4096


def _safe_key(key: str) -> str:
    return hashlib.md5(key.encode()).hexdigest()


class ShmBackend(Backend):
    """ASCEND store backend backed by ordinary shared-directory files.

    Parameters
    ----------
    parallel_config:
        Used to discover the local rank for ``torch.npu.set_device``.
    """

    def __init__(self, parallel_config: ParallelConfig) -> None:
        self._shm_dir = os.environ.get(
            "VLLM_KV_SHM_DIR", "/dev/shm/vllm_kv_transfer"
        )
        os.makedirs(self._shm_dir, exist_ok=True)

        from vllm.distributed.parallel_state import get_world_group

        self._local_rank = get_world_group().local_rank
        self._device_set = False

        logger.info(
            "ShmBackend initialised  shm_dir=%s  local_rank=%d",
            self._shm_dir,
            self._local_rank,
        )

    # ------------------------------------------------------------------
    # Backend interface
    # ------------------------------------------------------------------

    def set_device(self) -> None:
        if self._device_set:
            return
        torch.npu.set_device(f"npu:{self._local_rank}")
        self._device_set = True

    def register_buffer(self, ptrs: list[int], lengths: list[int]) -> None:
        pass

    def exists(self, keys: list[str]) -> list[int]:
        """Return 1 for every key whose file exists, else 0."""
        result: list[int] = []
        for key in keys:
            fpath = os.path.join(self._shm_dir, _safe_key(key))
            result.append(1 if os.path.exists(fpath) else 0)
        return result

    def put(
        self,
        keys: list[str],
        addrs: list[list[int]],
        sizes: list[list[int]],
    ) -> None:
        """DMA NPU->CPU for all keys, then atomically write shared files.

        ``addrs[i]`` / ``sizes[i]`` are lists of NPU device pointers and
        byte-sizes for the sub-tensors that make up block *i*.
        """
        n_keys = len(keys)
        if n_keys == 0:
            return

        # ---- accumulate (src_ptr, dst_ptr, sz) for batched DMA ----------
        dma_src: list[int] = []
        dma_dst: list[int] = []
        dma_sz: list[int] = []
        # track per-key CPU buffers so they stay alive until the file is written
        key_bufs: list[tuple[str, torch.Tensor, list[int]]] = []

        for key, layer_addrs, layer_sizes in zip(keys, addrs, sizes):
            total = sum(layer_sizes)
            # pin_memory=True gives us a page-locked host buffer that the
            # NPU DMA engine can access directly.
            cpu_buf = torch.empty(total, dtype=torch.uint8, pin_memory=True)
            key_bufs.append((key, cpu_buf, layer_sizes))

            offset = 0
            for npu_ptr, sz in zip(layer_addrs, layer_sizes):
                if sz == 0:
                    continue
                dma_src.append(npu_ptr)
                dma_dst.append(cpu_buf.data_ptr() + offset)
                dma_sz.append(sz)
                offset += sz

        # ---- issue batched DMA in chunks (some drivers cap batch size) ----
        self._batch_copy(dma_src, dma_dst, dma_sz, _DIRECTION_D2H)

        # ---- write each key's buffer to disk atomically -------------------
        for key, cpu_buf, layer_sizes in key_bufs:
            self._write_file(key, layer_sizes, cpu_buf)
    def get(
        self,
        keys: list[str],
        addrs: list[list[int]],
        sizes: list[list[int]],
    ) -> None:
        """Read shared files, then DMA CPU->NPU."""
        n_keys = len(keys)
        if n_keys == 0:
            return

        dma_src: list[int] = []
        dma_dst: list[int] = []
        dma_sz: list[int] = []

        for key, layer_addrs, layer_sizes in zip(keys, addrs, sizes):
            fpath = os.path.join(self._shm_dir, _safe_key(key))
            if not os.path.exists(fpath):
                continue

            try:
                data, _file_sizes = self._read_file(fpath)
            except Exception:
                logger.exception("ShmBackend.get: failed to read %s", fpath)
                continue

            cpu_buf = torch.from_numpy(
                np.frombuffer(data, dtype=np.uint8)
            )

            offset = 0
            for npu_ptr, sz in zip(layer_addrs, layer_sizes):
                if sz == 0:
                    continue
                dma_src.append(cpu_buf.data_ptr() + offset)
                dma_dst.append(npu_ptr)
                dma_sz.append(sz)
                offset += sz

            # Delete the file to reclaim space (consumer has read it)
            try:
                os.remove(fpath)
            except OSError:
                pass

        if dma_src:
            self._batch_copy(dma_src, dma_dst, dma_sz, _DIRECTION_H2D)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _batch_copy(
        src: list[int],
        dst: list[int],
        sizes: list[int],
        direction: int,
    ) -> None:
        """Call ``swap_blocks_batch`` in chunks."""
        n = len(src)
        for start in range(0, n, _MAX_BATCH_COPY):
            end = min(start + _MAX_BATCH_COPY, n)
            chunk_src = torch.tensor(src[start:end], dtype=torch.int64)
            chunk_dst = torch.tensor(dst[start:end], dtype=torch.int64)
            chunk_sz = torch.tensor(sizes[start:end], dtype=torch.int64)
            torch.ops._C_ascend.swap_blocks_batch(
                chunk_src, chunk_dst, chunk_sz, direction
            )
        # Must synchronise so the host-side data is visible to subsequent
        # file writes / the device-side data is visible to kernels.
        torch.npu.synchronize()

    def _write_file(
        self, key: str, sizes: list[int], cpu_buf: torch.Tensor
    ) -> None:
        """Atomically write *cpu_buf* to a file named by *key*."""
        safe = _safe_key(key)
        fpath = os.path.join(self._shm_dir, safe)
        tmp = fpath + ".tmp"

        raw = cpu_buf.numpy().tobytes()
        total = len(raw)

        with open(tmp, "wb") as fh:
            fh.write(struct.pack(_HEADER_FMT, len(sizes), total))
            for sz in sizes:
                fh.write(struct.pack("=Q", sz))
            fh.write(raw)

        os.replace(tmp, fpath)

    @staticmethod
    def _read_file(path: str) -> tuple[bytes, list[int]]:
        """Return (data_bytes, sizes_list)."""
        with open(path, "rb") as fh:
            hdr = fh.read(_HEADER_SIZE)
            if len(hdr) < _HEADER_SIZE:
                raise RuntimeError(f"Truncated header in {path}")
            num_elem, total = struct.unpack(_HEADER_FMT, hdr)

            file_sizes: list[int] = []
            for _ in range(num_elem):
                sz_bytes = fh.read(8)
                if len(sz_bytes) < 8:
                    break
                file_sizes.append(struct.unpack("=Q", sz_bytes)[0])

            data = fh.read(total)
        return data, file_sizes