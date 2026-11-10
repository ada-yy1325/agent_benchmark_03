#!/usr/bin/env python3
"""Add 3 debug traces to Mooncake connector for PD diagnosis."""
C = "/opt/mamba/lib/python3.13/site-packages/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py"
with open(C) as f:
    s = f.read()

# Trace 1: update_state_after_alloc entry — the key diagnostic
n1 = "        params = request.kv_transfer_params\n        logger.debug("
o1 = "        params = request.kv_transfer_params\n        logger.info(\"PDDBG update_state_after_alloc req=%s kv_transfer_params=%s\", request.request_id, params)\n        logger.debug("
assert n1 in s, "anchor n1 MISSING"
assert o1 not in s, "already patched"
s = s.replace(n1, o1, 1)
print("Trace 1: update_state_after_alloc OK")

# Trace 2: do_remote_decode branch
n2 = "elif params.get(\"do_remote_decode\"):\n            assert not self.is_kv_consumer\n            if not params.get(\"transfer_id\"):"
o2 = "elif params.get(\"do_remote_decode\"):\n            assert not self.is_kv_consumer\n            logger.info(\"PD-DEBUG do_remote_decode branch entered, transfer_id=%s\", params.get(\"transfer_id\"))\n            if not params.get(\"transfer_id\"):"
if o2 not in s:
    assert n2 in s, "anchor n2 MISSING"
    s = s.replace(n2, o2, 1)
    print("Trace 2: do_remote_decode OK")
else:
    print("Trace 2: already present")

# Trace 3: _reqs_need_send queued
n3 = "                self._reqs_need_send[request.request_id] = (request, [])"
o3 = "                logger.info(\"PD-DEBUG _reqs_need_send queued req_id=%s\", request.request_id)\n                self._reqs_need_send[request.request_id] = (request, [])"
if o3 not in s:
    assert n3 in s, "anchor n3 MISSING"
    s = s.replace(n3, o3, 1)
    print("Trace 3: _reqs_need_send OK")
else:
    print("Trace 3: already present")

with open(C, "w") as f:
    f.write(s)
print("ALL TRACES APPLIED")