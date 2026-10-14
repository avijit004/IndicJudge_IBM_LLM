"""Read-only coverage report; counts distinct validated results, never contacts models."""
import argparse
from common import (ROOT, CANDIDATES, JUDGES, candidate_rows, judged_rows, load_config, cli_entry)
from data import load_manifest
from prompts import LANGUAGES, TASKS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=ROOT / 'config.json')
    parser.add_argument('--manifest', default=ROOT / 'data' / 'selection.json')
    parser.add_argument('--output-dir', default=ROOT / 'outputs')
    parser.add_argument('--detail', action='store_true')
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest); config = load_config(args.config)
    candidates = {}; groups = []
    for model in CANDIDATES:
        rows = candidate_rows(args.output_dir, model, manifest, config)
        candidates.update({(r['sample_id'], model): r for r in rows})
        groups.append(('generation', model, rows, 200))
    for model in JUDGES:
        groups.append(('judging', model, judged_rows(args.output_dir, model, manifest, config, candidates), 400))
    for stage, model, rows, target in groups:
        print(f'{stage:10} {model:12} {len(rows):4}/{target} completed; {target-len(rows)} remaining')
        if args.detail:
            for code, language in LANGUAGES.items():
                for task in TASKS:
                    done = sum(r['language_code'] == code and r['task'] == task for r in rows)
                    cell_target = 5 if stage == 'generation' else 10
                    print(f'  {language:10} {task:14} {done}/{cell_target}; remaining {cell_target-done}')
    return 0


if __name__ == '__main__':
    raise SystemExit(cli_entry(main))
