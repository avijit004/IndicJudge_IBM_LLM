"""
Compatibility shim: HF `datasets` (as of 2.21/3.6) ships a custom dill
Pickler subclass (`datasets.utils._dill.Pickler`) used only for computing
cache fingerprints. Its `_batch_setitems(self, items)` override predates a
CPython 3.14 change where `pickle._Pickler.save_dict` now calls
`self._batch_setitems(items, obj)` (extra `obj` arg). On Python 3.14 this
raises:
    TypeError: Pickler._batch_setitems() takes 2 positional arguments but 3 were given
for every load_dataset() call, before any real work happens.

This module must be imported before `datasets` triggers any hashing/caching
(i.e. before the first `load_dataset` / `get_dataset_config_names` call).
It patches the override to accept (and forward) the extra `obj` argument.
Safe no-op on Python < 3.14 or once upstream fixes it.
"""
import sys

if sys.version_info >= (3, 14):
    try:
        import datasets.utils._dill as _dill_mod
        import dill as _dill

        def _patched_batch_setitems(self, items, obj=None):
            if getattr(self, "_legacy_no_dict_keys_sorting", False):
                return _dill.Pickler._batch_setitems(self, items, obj)
            try:
                items = sorted(items)
            except Exception:
                from datasets.fingerprint import Hasher

                items = sorted(items, key=lambda x: Hasher.hash(x[0]))
            return _dill.Pickler._batch_setitems(self, items, obj)

        _dill_mod.Pickler._batch_setitems = _patched_batch_setitems
    except Exception as e:  # pragma: no cover
        print(f"[_compat] warning: could not patch datasets/dill for py3.14: {e}", file=sys.stderr)
