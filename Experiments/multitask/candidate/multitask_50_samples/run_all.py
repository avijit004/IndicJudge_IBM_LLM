"""Run a generation or judging stage sequentially; defaults to all 10 languages and four tasks."""
import argparse
import subprocess
import sys
from common import ROOT, CANDIDATES, JUDGES, cli_entry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('generate', 'judge'))
    parser.add_argument('--models', nargs='+', help='Model keys; defaults to every model in this stage')
    args, flags = parser.parse_known_args(argv)
    supported = CANDIDATES if args.stage == 'generate' else JUDGES
    models = list(dict.fromkeys(args.models or supported))
    if any(m not in supported for m in models):
        parser.error(f'{args.stage} model keys must be from {supported}')
    failures = []
    for model in models:
        result = subprocess.run([sys.executable, '-u', str(ROOT / (args.stage + '.py')), '--model', model, *flags])
        if result.returncode:
            failures.append(model)
    if failures:
        print('Incomplete/failed models: ' + ', '.join(failures))
        return 1
    print('All requested commands succeeded.')
    return 0


if __name__ == '__main__':
    raise SystemExit(cli_entry(main))
