#!/usr/bin/env python3
"""
Fix transformers 4.51.0 -> vLLM 0.23.0 compatibility.
Patches _import_structure AND _class_to_module at runtime, then launches baseline.
"""
import sys, os

print("[fix] applying transformers->vLLM compatibility patches", flush=True)

# ---- Step 0: Import transformers and apply patches ----
import transformers

# 0.1: Gemma3Config
from transformers.models.gemma3.configuration_gemma3 import Gemma3Config
transformers.Gemma3Config = Gemma3Config

# 0.2: ALLOWED_ATTENTION_LAYER_TYPES / ALLOWED_LAYER_TYPES
import transformers.configuration_utils as _cu
for _n in ("ALLOWED_ATTENTION_LAYER_TYPES", "ALLOWED_LAYER_TYPES"):
    if not hasattr(_cu, _n):
        setattr(_cu, _n, [])

# 0.3: AutoVideoProcessor -- critical: must update BOTH _import_structure AND _class_to_module
from transformers.models.auto.processing_auto import AutoProcessor
import transformers.models.auto as _auto_mod
_auto_mod.AutoVideoProcessor = AutoProcessor

_auto_key = "models.auto"
_name = "AutoVideoProcessor"

# Update _import_structure
_istruct = transformers._import_structure
if _auto_key in _istruct and _name not in _istruct[_auto_key]:
    _istruct[_auto_key].append(_name)

# Update _class_to_module (the lazy loader uses this for individual class lookups)
_ctm = getattr(transformers, "_class_to_module", None)
if _ctm is not None and _name not in _ctm:
    _ctm[_name] = _auto_key

# Also update for Gemma3Config
_g3_name = "Gemma3Config"
if _auto_key in _istruct and _g3_name not in _istruct[_auto_key]:
    _istruct[_auto_key].append(_g3_name)
if _ctm is not None and _g3_name not in _ctm:
    _ctm[_g3_name] = _auto_key

print("[fix] _import_structure and _class_to_module patched", flush=True)

# Quick validation
try:
    from transformers import AutoVideoProcessor
    print("[fix]   AutoVideoProcessor OK (type={})".format(type(AutoVideoProcessor)), flush=True)
except Exception as e:
    print("[fix]   AutoVideoProcessor FAILED: {}".format(e), flush=True)

try:
    from transformers import Gemma3Config as _gc
    print("[fix]   Gemma3Config OK (type={})".format(type(_gc)), flush=True)
except Exception as e:
    print("[fix]   Gemma3Config FAILED: {}".format(e), flush=True)

print("[fix] all patches applied", flush=True)

# ---- Step 1: Launch baseline server ----
print("[fix] launching baseline server...", flush=True)
_this_dir = os.path.dirname(os.path.abspath(__file__))
_start_script = os.path.join(_this_dir, "start_server_baseline.py")
print("[fix] start_script: {}".format(_start_script), flush=True)

sys.argv = [_start_script]
import runpy
runpy.run_path(_start_script, run_name="__main__")