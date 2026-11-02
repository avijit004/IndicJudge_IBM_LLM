"""Offline subprocess adapter: validate existing generation files without changing them."""
import argparse
import csv
import json
from pathlib import Path
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', type=Path, required=True)
    args = parser.parse_args()
    source = args.source_dir.resolve()
    sys.path.insert(0, str(source))
    csv.field_size_limit(16 * 1024 * 1024)
    import common
    import data
    if Path(common.__file__).resolve().parent != source:
        raise RuntimeError('Wrong generation validation module')
    manifest = data.load_manifest(source/'data/selection.json')
    config = common.load_config(source/'config.json')
    candidates = {k: common.candidate_rows(source/'outputs', k, manifest, config) for k in common.CANDIDATES}
    print(json.dumps({'manifest': manifest, 'candidates': candidates}, ensure_ascii=False))
