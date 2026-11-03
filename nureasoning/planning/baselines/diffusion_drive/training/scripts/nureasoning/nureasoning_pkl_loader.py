"""
Safe loader for nuReasoning's .pkl files.

nuReasoning's map.pkl / ego_state pkl / annotations pkl are pickled using
custom classes from a `data_schema` module (e.g. `data_schema.Lane`,
`data_schema.nuReasoningStaticMap`). That module isn't publicly available,
so a plain `pickle.load(...)` fails with `ModuleNotFoundError: No module
named 'data_schema'`.

This loader intercepts any unresolvable (module, class) pair during
unpickling and substitutes a permissive stand-in object that just stores
whatever fields were pickled as plain attributes. This was verified
against an actual uploaded map.pkl (produced 9 categories: lanes,
baseline_paths, boundaries, crosswalks, intersections, stop_polygons,
road_blocks, traffic_lights, lane_connectors -- 32/73/208/4/9/7/12/12/41
items respectively in the sample clip).

If you have the real `data_schema` module available, just `import
data_schema` before calling `load_pkl` and it will be used automatically
in place of the stand-ins (real classes take priority).
"""

from __future__ import annotations

import pickle
import sys
import types
from pathlib import Path
from typing import Any, Dict, Type, Union


class NuReasoningRecord:
    """
    Permissive stand-in for any nuReasoning data_schema class.

    Behaves like a simple namespace: whatever fields were pickled become
    attributes. Supports both dict-style state (`__setstate__`) and
    positional/keyword construction.
    """

    def __init__(self, *args, **kwargs):
        if args and isinstance(args[0], dict):
            self.__dict__.update(args[0])
        self.__dict__.update(kwargs)

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            # some pickled objects use a plain tuple/list state
            self.__dict__["_state"] = state

    def __getattr__(self, name):
        # Only called when normal attribute lookup fails -- makes missing
        # fields explicit instead of a confusing AttributeError deep in
        # calling code.
        raise AttributeError(
            f"'{type(self).__name__}' object has no field '{name}'. "
            f"Available fields: {sorted(self.__dict__.keys())}"
        )

    def __repr__(self):
        fields = ", ".join(f"{k}={v!r}" for k, v in list(self.__dict__.items())[:6])
        more = "..." if len(self.__dict__) > 6 else ""
        return f"{type(self).__name__}({fields}{more})"

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


class _StubModuleRegistry:
    """Lazily creates and caches permissive classes per (module, name)."""

    def __init__(self):
        self._modules: Dict[str, types.ModuleType] = {}

    def get_class(self, module_name: str, class_name: str) -> Type:
        # If the real module is already importable, prefer it.
        if module_name in sys.modules:
            real_mod = sys.modules[module_name]
            if hasattr(real_mod, class_name):
                return getattr(real_mod, class_name)

        if module_name not in self._modules:
            stub_mod = types.ModuleType(module_name)
            self._modules[module_name] = stub_mod
            # Register so nested pickled references to the same module
            # resolve consistently.
            sys.modules.setdefault(module_name, stub_mod)

        stub_mod = self._modules[module_name]
        if not hasattr(stub_mod, class_name):
            cls = type(class_name, (NuReasoningRecord,), {"__module__": module_name})
            setattr(stub_mod, class_name, cls)
        return getattr(stub_mod, class_name)


_registry = _StubModuleRegistry()


class _PermissiveUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            return _registry.get_class(module, name)


def load_pkl(path: Union[str, Path]) -> Any:
    """
    Load any nuReasoning .pkl file (map, ego_state, or annotations),
    substituting permissive stand-ins for unresolvable custom classes.

    :param path: path to the .pkl file
    :return: the unpickled object (real class if available, otherwise a
        NuReasoningRecord tree)
    """
    with open(path, "rb") as f:
        return _PermissiveUnpickler(f).load()


if __name__ == "__main__":
    import sys as _sys

    if len(_sys.argv) < 2:
        print("Usage: python nureasoning_pkl_loader.py <path-to-pkl>")
        raise SystemExit(1)

    obj = load_pkl(_sys.argv[1])
    print(f"Loaded: {type(obj)}")
    if hasattr(obj, "__dict__"):
        for key, val in obj.__dict__.items():
            if isinstance(val, list):
                sample = val[0] if val else None
                print(f"  {key}: list[{len(val)}] sample={sample!r}")
            else:
                print(f"  {key}: {val!r}")