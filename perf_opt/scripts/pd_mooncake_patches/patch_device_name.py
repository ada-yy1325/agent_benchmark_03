#!/usr/bin/env python3
"""Patch mooncake_connector.py: pass device_name="ascend" to engine.initialize().

The connector calls TransferEngine.initialize(hostname, md_server, protocol, "").
The 4th param (device_name) is empty, causing the engine to default to a
non-Ascend backend.  On Ascend NPU this means the AscendDirectTransport and
AscendSharedSegment are never activated, and the engine falls back to TCP
transport which cannot read NPU memory (Bad address / EFAULT).

Fix: pass device_name="ascend" so the engine activates its Ascend backend.
"""
import shutil

C = "/opt/mamba/lib/python3.13/site-packages/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py"
shutil.copy2(C, C + ".bak_device_name")

src = open(C).read()

old = 'ret_value = self.engine.initialize(self.hostname, "P2PHANDSHAKE", protocol, "")'
new = 'ret_value = self.engine.initialize(self.hostname, "P2PHANDSHAKE", protocol, "ascend")'

assert old in src, "anchor MISSING"
src = src.replace(old, new, 1)

with open(C, "w") as f:
    f.write(src)

print("PATCHED: device_name now = 'ascend'")
print("Backup:", C + ".bak_device_name")