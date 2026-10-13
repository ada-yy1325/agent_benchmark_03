#!/usr/bin/env python3
"""
Fix transformers 4.51.0 -> vLLM 0.23.0 compatibility.
Attacks all missing imports via sys.meta_path hook + targeted patches.
"""
import sys, os, types, importlib.abc, importlib.machinery

print("[fix] applying transformers->vLLM compatibility patches", flush=True)

# ---- Step 0a: Pre-create dummy modules for missing transformers.* modules ----
# Place finder at END of meta_path so real modules resolve first.
class _AutoGenModule(types.ModuleType):
    """Module that auto-generates dummy classes for any attribute access."""
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        cls = type(name, (), {})
        setattr(self, name, cls)
        return cls

class _DummyTransformerFinder(importlib.abc.MetaPathFinder):
    """Last-resort finder: creates _AutoGenModule for any unresolvable transformers.* import."""
    def find_spec(self, fullname, path, target=None):
        if not fullname.startswith("transformers."):
            return None
        if fullname in sys.modules:
            return None
        # All other finders failed, create an auto-generating dummy module
        print("[fix] auto-creating dummy: {}".format(fullname), flush=True)
        mod = _AutoGenModule(fullname)
        mod.__file__ = "[dummy]"
        mod.__path__ = []
        spec = importlib.machinery.ModuleSpec(fullname, None)
        mod.__spec__ = spec
        sys.modules[fullname] = mod
        return spec

# Append to END of meta_path (last resort)
sys.meta_path.append(_DummyTransformerFinder())

# ---- Step 0b: Import transformers and apply targeted patches ----
import transformers

# Patch 1: Gemma3Config
from transformers.models.gemma3.configuration_gemma3 import Gemma3Config
transformers.Gemma3Config = Gemma3Config

# Patch 2: ALLOWED_ATTENTION_LAYER_TYPES
import transformers.configuration_utils as _cu
for _n in ("ALLOWED_ATTENTION_LAYER_TYPES", "ALLOWED_LAYER_TYPES"):
    if not hasattr(_cu, _n):
        setattr(_cu, _n, [])

# Patch 3: AutoVideoProcessor in _import_structure AND _class_to_module
from transformers.models.auto.processing_auto import AutoProcessor
import transformers.models.auto as _auto_mod
_auto_mod.AutoVideoProcessor = AutoProcessor

_auto_key = "models.auto"
for _name in ("AutoVideoProcessor", "Gemma3Config"):
    _istruct = transformers._import_structure
    if _auto_key in _istruct and _name not in _istruct[_auto_key]:
        _istruct[_auto_key].append(_name)
    _ctm = getattr(transformers, "_class_to_module", None)
    if _ctm is not None and _name not in _ctm:
        _ctm[_name] = _auto_key

# Patch 4: Pre-populate known missing classes into dummy modules
# These classes are in transformers 5.x but not 4.51.0
_known_missing = {
    "transformers.video_processing_utils": ["BaseVideoProcessor"],
    "transformers.video_utils": [],
    "transformers.image_utils": [],
}
for _mod_name, _classes in _known_missing.items():
    if _mod_name not in sys.modules:
        _mod = types.ModuleType(_mod_name)
        _mod.__file__ = "[dummy]"
        _mod.__path__ = []
        _mod.__spec__ = importlib.machinery.ModuleSpec(_mod_name, None)
        sys.modules[_mod_name] = _mod
    else:
        _mod = sys.modules[_mod_name]
    for _cls_name in _classes:
        if not hasattr(_mod, _cls_name):
            setattr(_mod, _cls_name, type(_cls_name, (), {}))

print("[fix] all patches applied", flush=True)

# ---- Step 1: Launch baseline server ----
print("[fix] launching baseline server...", flush=True)
_this_dir = os.path.dirname(os.path.abspath(__file__))
_start_script = os.path.join(_this_dir, "start_server_baseline.py")
print("[fix] start_script: {}".format(_start_script), flush=True)

sys.argv = [_start_script]
import runpy
runpy.run_path(_start_script, run_name="__main__")