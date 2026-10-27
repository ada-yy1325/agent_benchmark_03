#!/usr/bin/env python3
"""P2PShmConnector: 1P1D PD connector for vLLM-Ascend."""
import os, time
from typing import Any
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.forward_context import ForwardContext
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import AscendStoreConnector

PD_SYNC_DIR = "/dev/shm/vllm_pd_sync"
PD_SYNC_TIMEOUT = 120
class P2PShmConnector(AscendStoreConnector):
    """PD connector wrapping AscendStoreConnector with DONE-file sync."""

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole,
                 kv_cache_config: KVCacheConfig | None = None):
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        os.makedirs(PD_SYNC_DIR, exist_ok=True)
        dp_rank = getattr(vllm_config.parallel_config, "data_parallel_rank", 0)
        if self.kv_role == "kv_both":
            self.kv_role = "kv_producer" if dp_rank == 0 else "kv_consumer"
        self._pd_seen_reqs: set[str] = set()

    # ── Scheduler side ──
    def get_num_new_matched_tokens(self, request: Request, num_computed_tokens: int):
        if self.kv_role == "kv_consumer" and self.consumer_is_to_load:
            self._wait_for_producer(request.request_id)
        return super().get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        return super().update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(self, scheduler_output: SchedulerOutput):
        return super().build_connector_meta(scheduler_output)

    def request_finished(self, request, block_ids):
        result = super().request_finished(request, block_ids)
        self._cleanup_done(request.request_id)
        return result

    def request_finished_all_groups(self, request, block_ids):
        result = super().request_finished_all_groups(request, block_ids)
        self._cleanup_done(request.request_id)
        return result

    def update_connector_output(self, connector_output: KVConnectorOutput):
        return super().update_connector_output(connector_output)

    def bind_gpu_block_pool(self, gpu_block_pool: BlockPool) -> None:
        return super().bind_gpu_block_pool(gpu_block_pool)

    # ── Worker side ──
    def register_kv_caches(self, kv_caches: dict[str, Any]):
        super().register_kv_caches(kv_caches)

    def start_load_kv(self, forward_context: ForwardContext, **kwargs) -> None:
        super().start_load_kv(forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        super().wait_for_layer_load(layer_name)

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs) -> None:
        super().save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)

    def wait_for_save(self):
        """Producer: signal DONE after saving all layers."""
        super().wait_for_save()
        if self.kv_role != "kv_producer":
            return
        metadata = self._get_connector_metadata()
        if metadata is None:
            return
        for rm in metadata.requests:
            if rm.req_id not in self._pd_seen_reqs:
                done_file = os.path.join(PD_SYNC_DIR, f"{rm.req_id}.done")
                open(done_file, "w").close()
                self._pd_seen_reqs.add(rm.req_id)

    def get_finished(self, finished_req_ids):
        return super().get_finished(finished_req_ids)

    def get_block_ids_with_load_errors(self):
        return super().get_block_ids_with_load_errors()

    def build_connector_worker_meta(self):
        return super().build_connector_worker_meta()

    # ── Internal ──
    def _cleanup_done(self, req_id: str):
        done_file = os.path.join(PD_SYNC_DIR, f"{req_id}.done")
        if os.path.exists(done_file):
            os.remove(done_file)
        self._pd_seen_reqs.discard(req_id)

    def _wait_for_producer(self, req_id: str):
        done_file = os.path.join(PD_SYNC_DIR, f"{req_id}.done")
        if os.path.exists(done_file):
            return
        t0 = time.time()
        while not os.path.exists(done_file):
            time.sleep(0.002)
            if time.time() - t0 > PD_SYNC_TIMEOUT:
                raise TimeoutError(f"P2PShm: producer timeout for {req_id}")


def register():
    """Register P2PShmConnector with vLLM connector factory."""
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
    KVConnectorFactory.register_connector(
        "P2PShmConnector",
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.p2p_shm_connector",
        "P2PShmConnector",
    )