"""Freeze 2,000 samples from the Dataset folder. Never changes source files."""
import argparse
import csv
import json
from pathlib import Path
from data import ROOT, DEFAULT_DATASET, build_manifest, validate_manifest, canonical


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--check-only', action='store_true', help='Verify source data and frozen selection without writing')
    args = parser.parse_args(argv)
    manifest = validate_manifest(build_manifest(args.dataset_dir, 1, 50))
    target = ROOT / 'data' / 'selection.json'
    if target.exists():
        previous = json.loads(target.read_text(encoding='utf-8'))
        if validate_manifest(previous) != manifest:
            raise ValueError('Dataset differs from the frozen selection. Existing experiment is unchanged. '
                             'Use a separate experiment folder for a new dataset selection.')
        print('Existing frozen selection matches the source dataset.')
    elif args.check_only:
        print('Source data valid; selection has not yet been written.')
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents replacing another run\'s selection.
        with target.open('x', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
            f.write('\n')
    review = ROOT / 'data' / 'selection.csv'
    if not args.check_only and not review.exists():
        fields = ['sample_id', 'task', 'language', 'language_code', 'source_id', 'source_file',
                  'source_row', 'sample_hash', 'input_json', 'reference_json']
        with review.open('x', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
            for sample in manifest['samples']:
                row = {k: sample[k] for k in fields if k in sample}
                row.update(input_json=canonical(sample['input']), reference_json=canonical(sample['reference']))
                writer.writerow(row)
    print('40 source CSVs validated. Selection: 2,000 inputs; 4,000 candidate responses; 12,000 judgments.')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, csv.Error) as exc:
        raise SystemExit(f'Data preparation failed: {exc}')
