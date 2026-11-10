#!/usr/bin/env python3
"""Patch mooncake_connector.py: CPU staging for NPU memory in _send_blocks.

Problem: Mooncake engine's TCP transport (tcp_transport_session_impl.h) cannot
read NPU device memory — writeBody fails with "Bad address" (EFAULT).

Fix: Before calling engine.batch_transfer_sync_write(), copy NPU source data
to temporary CPU (pinned) buffers, then pass CPU pointers to the engine.
"""

import shutil

C = "/opt/mamba/lib/python3.13/site-packages/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py"

# Backup
shutil.copy2(C, C + ".bak_cpu_staging_v2")
src = open(C).read()

# ── Patch 1: register_kv_caches — add base_addr → NPU tensor mapping ──
old1 = """        self.kv_caches_base_addr = kv_data_ptrs
        self.seen_base_addresses = kv_data_ptrs

        ret_value = self.engine.batch_register_memory(kv_data_ptrs, kv_data_lens)"""

new1 = """        self.kv_caches_base_addr = kv_data_ptrs
        self.seen_base_addresses = kv_data_ptrs

        # PD fix: build NPU base_addr -> tensor reference for D2H copy
        self._npu_base_to_tensor = {}
        for base, cache in zip(kv_data_ptrs, self._registered_tensors):
            self._npu_base_to_tensor[base] = cache

        ret_value = self.engine.batch_register_memory(kv_data_ptrs, kv_data_lens)"""

# Also need to save registered tensors when processing cache_list
old1b = """            for cache in cache_list:
                self._log_debug_cache_registration(layer_name, cache)
                base_addr = cache.data_ptr()
                if base_addr in kv_data_ptrs:
                    continue"""

new1b = """            for cache in cache_list:
                self._log_debug_cache_registration(layer_name, cache)
                base_addr = cache.data_ptr()
                if base_addr in kv_data_ptrs:
                    continue
                self._registered_tensors.append(cache)"""

# Add _registered_tensors init
old1c = """        kv_data_ptrs = []
        kv_data_lens = []
        self.block_len_per_layer = []
        self.kv_block_data_len_per_layer = []"""

new1c = """        kv_data_ptrs = []
        kv_data_lens = []
        self.block_len_per_layer = []
        self.kv_block_data_len_per_layer = []
        self._registered_tensors = []"""

assert old1c in src, "Patch 1c MISSING"
src = src.replace(old1c, new1c, 1)

assert old1b in src, "Patch 1b MISSING"
src = src.replace(old1b, new1b, 1)

assert old1 in src, "Patch 1 MISSING"
src = src.replace(old1, new1, 1)

# ── Patch 2: _send_blocks — copy NPU→CPU before engine call ──
old2 = """    def _send_blocks(
        self,
        remote_session: str,
        src_ptrs: list[int],
        dst_ptrs: list[int],
        lengths: list[int],
    ) -> int:
        start_time = time.perf_counter()
        ret_value = self.engine.batch_transfer_sync_write(
            remote_session, src_ptrs, dst_ptrs, lengths
        )"""

new2 = """    def _send_blocks(
        self,
        remote_session: str,
        src_ptrs: list[int],
        dst_ptrs: list[int],
        lengths: list[int],
    ) -> int:
        # PD fix: copy NPU data to CPU staging before TCP send.
        # Mooncake engine TCP transport cannot read NPU memory directly.
        import torch
        self._staging_refs = []
        cpu_src_ptrs = []
        for npu_ptr, length in zip(src_ptrs, lengths):
            copied = False
            for base_addr, npu_tensor in getattr(self, '_npu_base_to_tensor', {}).items():
                total_bytes = npu_tensor.numel() * npu_tensor.element_size()
                if base_addr <= npu_ptr < base_addr + total_bytes:
                    offset = npu_ptr - base_addr
                    elem_size = npu_tensor.element_size()
                    start_elem = offset // elem_size
                    n_elem = (length + elem_size - 1) // elem_size
                    npu_slice = npu_tensor.contiguous().view(-1)[start_elem : start_elem + n_elem]
                    cpu_buf = torch.empty(length, dtype=torch.uint8, pin_memory=True)
                    if npu_slice.numel() > 0:
                        npu_bytes = npu_slice.contiguous().view(torch.uint8)[:length].cpu()
                        cpu_buf[:npu_bytes.numel()] = npu_bytes
                    self._staging_refs.append(cpu_buf)
                    cpu_src_ptrs.append(cpu_buf.data_ptr())
                    copied = True
                    break
            if not copied:
                logger.warning(\"PD fix: could not map NPU ptr 0x%x to tensor, sending directly\", npu_ptr)
                cpu_src_ptrs.append(npu_ptr)
        src_ptrs = cpu_src_ptrs
        start_time = time.perf_counter()
        ret_value = self.engine.batch_transfer_sync_write(
            remote_session, src_ptrs, dst_ptrs, lengths
        )"""

assert old2 in src, "Patch 2 MISSING"
src = src.replace(old2, new2, 1)

with open(C, "w") as f:
    f.write(src)

print("CPU STAGING V2 APPLIED")
print("Backup:", C + ".bak_cpu_staging_v2")