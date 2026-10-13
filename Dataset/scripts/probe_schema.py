#!/usr/bin/env python
"""
Small utility to inspect a HF dataset before committing to a field mapping.

Usage:
    python scripts/probe_schema.py <dataset_id> [config_name] [split]
    python scripts/probe_schema.py <dataset_id> --configs        # just list configs

Examples:
    python scripts/probe_schema.py ai4bharat/IndicQA --configs
    python scripts/probe_schema.py ai4bharat/IndicQA indicqa.hi validation
"""
import sys
import json

import _compat  # noqa: F401  (py3.14 / datasets dill fingerprint patch)
from datasets import get_dataset_config_names, load_dataset


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    dataset_id = sys.argv[1]

    print(f"=== {dataset_id} ===")
    try:
        configs = get_dataset_config_names(dataset_id, trust_remote_code=True)
        print(f"configs ({len(configs)}): {configs}")
    except Exception as e:
        print(f"[!] could not list configs: {e}")
        configs = []

    if "--configs" in sys.argv:
        return

    config_name = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else None
    split = sys.argv[3] if len(sys.argv) > 3 else None

    kwargs = {}
    if config_name:
        kwargs["name"] = config_name

    try:
        ds = load_dataset(dataset_id, streaming=True, trust_remote_code=True, **kwargs)
        if split is None:
            # streaming dataset dict -> pick first split
            if hasattr(ds, "keys"):
                split = list(ds.keys())[0]
                ds = ds[split]
        else:
            ds = ds[split]
        print(f"split used: {split}")
        it = iter(ds)
        ex = next(it)
        print("keys:", list(ex.keys()))
        print("example:")
        print(json.dumps(ex, indent=2, ensure_ascii=False, default=str)[:3000])
    except Exception as e:
        print(f"[!] failed to load/stream: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
