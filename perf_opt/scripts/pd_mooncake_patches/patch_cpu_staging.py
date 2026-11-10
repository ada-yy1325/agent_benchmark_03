#!/usr/bin/env python3
"""Patch mooncake_connector.py to use CPU staging buffers for NPU memory.

Problem: Mooncake engine's TCP transport cannot read NPU memory directly
(ClientSession::writeBody fails with "Bad address" EFAULT).

Fix: Allocate CPU pinned staging buffers during register_kv_caches(), register
them with the engine instead of NPU memory, and copy NPU→CPU before sending.

This is a workaround for the engine's lack of aclrtMemcpy in TCP transport path.
"""
import shutil, re

C = "/opt/mamba/lib/python3.13/site-packages/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py"

# Backup
shutil.copy2(C, C + ".bak_cpu_staging")
src = open(C).read()

# ── Patch 1: Replace batch_register_memory NPU pointers with CPU staging ──
# In register_kv_caches(), after computing kv_data_ptrs/kv_data_lens:
# - Keep NPU pointers for later D2H copy
# - Allocate CPU staging tensors
# - Register CPU staging pointers with engine
old_register = '''        self.kv_caches_base_addr = kv_data_ptrs
        self.seen_base_addresses = kv_data_ptrs

        ret_value = self.engine.batch_register_memory(kv_data_ptrs, kv_data_lens)'''
new_register = '''        self.kv_caches_base_addr = kv_data_ptrs
        self.seen_base_addresses = kv_data_ptrs

        # PD fix: allocate CPU staging buffers for NPU→CPU→TCP transfer
        # The Mooncake engine TCP transport cannot read NPU memory directly.
        import torch
        self._cpu_staging = []
        cpu_ptrs = []
        cpu_lens = []
        for npu_ptr, data_len in zip(kv_data_ptrs, kv_data_lens):
            # Allocate pinned CPU memory (page-locked, faster DMA)
            cpu_tensor = torch.empty(data_len, dtype=torch.uint8, device="cpu", pin_memory=True)
            self._cpu_staging.append(cpu_tensor)
            cpu_ptrs.append(cpu_tensor.data_ptr())
            cpu_lens.append(data_len)
        self._cpu_staging_ptrs = cpu_ptrs
        logger.info("PD fix: allocated %d CPU staging buffers (%d MB total)", 
                    len(cpu_ptrs), sum(cpu_lens) // (1024*1024))

        # Register CPU staging buffers with engine (NOT NPU memory)
        ret_value = self.engine.batch_register_memory(cpu_ptrs, cpu_lens)'''

assert old_register in src, "Patch 1 anchor MISSING"
src = src.replace(old_register, new_register, 1)

# ── Patch 2: In _send_blocks(), copy NPU→CPU before calling engine ──
old_send = '''    def _send_blocks(
        self,
        remote_session: str,
        src_ptrs: list[int],
        dst_ptrs: list[int],
        lengths: list[int],
    ) -> int:
        start_time = time.perf_counter()
        ret_value = self.engine.batch_transfer_sync_write(
            remote_session, src_ptrs, dst_ptrs, lengths
        )'''

new_send = '''    def _send_blocks(
        self,
        remote_session: str,
        src_ptrs: list[int],
        dst_ptrs: list[int],
        lengths: list[int],
    ) -> int:
        # PD fix: copy NPU data to CPU staging buffers before TCP send.
        # src_ptrs are NPU addresses; we allocated CPU staging in register_kv_caches.
        # Remap src_ptrs to CPU staging equivalents.
        if hasattr(self, "_cpu_staging_ptrs") and self._cpu_staging_ptrs:
            cpu_src_ptrs = []
            for npu_ptr in src_ptrs:
                # Find which CPU staging buffer corresponds to this NPU pointer
                # by matching within registered regions
                idx = next((i for i, base in enumerate(self.kv_caches_base_addr)
                           if base <= npu_ptr < base + self._cpu_staging[i].numel()), None)
                if idx is not None:
                    offset = npu_ptr - self.kv_caches_base_addr[idx]
                    cpu_src_ptrs.append(self._cpu_staging_ptrs[idx] + offset)
                else:
                    cpu_src_ptrs.append(npu_ptr)  # fallback
            # Copy NPU to CPU staging BEFORE engine transfer
            import torch
            for npu_ptr, cpu_ptr, length in zip(src_ptrs, cpu_src_ptrs, lengths):
                if npu_ptr != cpu_ptr:
                    npu_tensor = torch.frombuffer(
                        npu_ptr, dtype=torch.uint8, count=length,
                    )
                    cpu_tensor = torch.frombuffer(
                        cpu_ptr, dtype=torch.uint8, count=length,
                    )
                    cpu_tensor.copy_(npu_tensor)
            src_ptrs = cpu_src_ptrs
        start_time = time.perf_counter()
        ret_value = self.engine.batch_transfer_sync_write(
            remote_session, src_ptrs, dst_ptrs, lengths
        )'''

assert old_send in src, "Patch 2 anchor MISSING"
src = src.replace(old_send, new_send, 1)

# Write back
with open(C, "w") as f:
    f.write(src)

print("CPU STAGING PATCH APPLIED")
print("Backup saved to:", C + ".bak_cpu_staging")