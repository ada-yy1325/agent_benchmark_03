#!/usr/bin/env python3
"""P2PShmConnector: PD connector for vLLM Ascend DP=2 mode.

Extends AscendStoreConnector to split kv_both into producer/consumer
based on data_parallel_rank.  The AscendStore pool (ShmBackend)
handles synchronization; we just need to set the correct role so
the consumer side skips save and queries the pool for prefill KV.

Key fix: monkey-patches get_zmq_rpc_path_lookup during init to use
dp_rank=0 for ALL LookupKeyClient instances, so both DP ranks
connect to the single LookupKeyServer on rank 0.
"""

from typing import Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import (
    AscendStoreConnector,
)


class P2PShmConnector(AscendStoreConnector):
    """Split kv_both → kv_producer (dp_rank 0) / kv_consumer (dp_rank 1)."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: Any = None,
    ) -> None:
        dp_rank = getattr(
            vllm_config.parallel_config, "data_parallel_rank", 0
        )

        # Monkey-patch ZMQ socket path so that LookupKeyClient on both
        # DP ranks connects to the single LookupKeyServer on rank 0.
        import vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler as ps  # noqa: E402

        _orig_get_path = ps.get_zmq_rpc_path_lookup

        def _unified_path(cfg: VllmConfig) -> str:
            saved = cfg.parallel_config.data_parallel_rank
            try:
                cfg.parallel_config.data_parallel_rank = 0
                return _orig_get_path(cfg)
            finally:
                cfg.parallel_config.data_parallel_rank = saved

        ps.get_zmq_rpc_path_lookup = _unified_path
        try:
            super().__init__(
                vllm_config=vllm_config, role=role,
                kv_cache_config=kv_cache_config,
            )
        finally:
            ps.get_zmq_rpc_path_lookup = _orig_get_path

        if self.kv_role == "kv_both":
            self.kv_role = "kv_producer" if dp_rank == 0 else "kv_consumer"

        # Propagate role to the internal scheduler so that
        # KVPoolScheduler.get_num_new_matched_tokens honours the split.
        if hasattr(self, "connector_scheduler"):
            self.connector_scheduler.kv_role = self.kv_role


def register() -> None:
    """Register the connector with vLLM's factory."""
    from vllm.distributed.kv_transfer.kv_connector.factory import (
        KVConnectorFactory,
    )

    KVConnectorFactory.register_connector(
        "P2PShmConnector",
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.p2p_shm_connector",
        "P2PShmConnector",
    )