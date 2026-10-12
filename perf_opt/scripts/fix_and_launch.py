#!/usr/bin/env python3
"""
Fix transformers 4.51.0 -> vLLM 0.23.0 compatibility.
Dual approach: edit __init__.py for persistent fix + runtime patch for current process.
"""
import sys, os, site

print("[fix] applying transformers->vLLM compatibility patches", flush=True)

# ---- Step 0a: Persistent fix - edit transformers/__init__.py ----
_site = site.getsitepackages()[0]
_init_path = os.path.join(_site, "transformers", "__init__.py")
_marker = "# === AUTO-PATCH vLLM-0.23.0 compat ==="

if os.path.exists(_init_path):
    with open(_init_path, "r") as f:
        content = f.read()
    if _marker not in content:
        # Append a compat shim that runs on transformers import
        shim = """
{}
# Compat shim for vLLM 0.23.0 on transformers 4.51.0
import sys as _fsys
if not _fsys.modules.get("transformers.models.auto"):
    import transformers.models.auto as _auto
else:
    _auto = _fsys.modules["transformers.models.auto"]

# Gemma3Config
try:
    from transformers.models.gemma3.configuration_gemma3 import Gemma3Config as _GC
except Exception:
    _GC = type("Gemma3Config", (), {{}})

# ALLOWED_ATTENTION_LAYER_TYPES
try:
    import transformers.configuration_utils as _tcu
    for _n in ("ALLOWED_ATTENTION_LAYER_TYPES", "ALLOWED_LAYER_TYPES"):
        if not hasattr(_tcu, _n):
            setattr(_tcu, _n, [])
except Exception:
    pass

# AutoVideoProcessor
if not hasattr(_auto, "AutoVideoProcessor"):
    try:
        from transformers.models.auto.processing_auto import AutoProcessor as _AVP
    except Exception:
        _AVP = type("AutoVideoProcessor", (), {{}})
    _auto.AutoVideoProcessor = _AVP
""".format(_marker)
        with open(_init_path, "a") as f:
            f.write(shim)
        print("[fix] transformers/__init__.py patched (persistent)", flush=True)
    else:
        print("[fix] transformers/__init__.py already patched, skip", flush=True)

# ---- Step 0b: Runtime patch for current process ----
import transformers

# Gemma3Config
from transformers.models.gemma3.configuration_gemma3 import Gemma3Config
transformers.Gemma3Config = Gemma3Config

# ALLOWED_ATTENTION_LAYER_TYPES
import transformers.configuration_utils as _cu
for _n in ("ALLOWED_ATTENTION_LAYER_TYPES", "ALLOWED_LAYER_TYPES"):
    if not hasattr(_cu, _n):
        setattr(_cu, _n, [])

# AutoVideoProcessor - set on BOTH transformers namespace AND models.auto
from transformers.models.auto.processing_auto import AutoProcessor
import transformers.models.auto as _auto_mod
_auto_mod.AutoVideoProcessor = AutoProcessor
# Direct attribute on transformers module for "from transformers import X"
# Must go into __dict__ to bypass __getattr__ in some versions
transformers.__dict__["AutoVideoProcessor"] = AutoProcessor

# Also extend _import_structure for lazy loader
_istruct = getattr(transformers, "_import_structure", {})
_auto_key = "models.auto"
if _auto_key in _istruct:
    for _name in ("AutoVideoProcessor", "Gemma3Config"):
        if _name not in _istruct[_auto_key]:
            _istruct[_auto_key].append(_name)

print("[fix] all patches applied", flush=True)

# ---- Step 1: Launch baseline server ----
print("[fix] launching baseline server...", flush=True)
_this_dir = os.path.dirname(os.path.abspath(__file__))
_start_script = os.path.join(_this_dir, "start_server_baseline.py")
print("[fix] start_script: {}".format(_start_script), flush=True)

sys.argv = [_start_script]
import runpy
runpy.run_path(_start_script, run_name="__main__")