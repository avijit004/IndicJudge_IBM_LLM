"""Run languages sequentially, continuing after failures and returning their status."""
import argparse
from pathlib import Path
import subprocess
import sys

LANGUAGES = ('bengali', 'gujarati', 'hindi', 'kannada', 'malayalam',
             'marathi', 'odia', 'punjabi', 'tamil', 'telugu')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # Sarvam retains its previous no-pilot interface.
    # Data validation runs in each standalone script.
    parser.add_argument('--languages', nargs='+', choices=LANGUAGES, default=LANGUAGES)
    args = parser.parse_args()
    flags = []
    failed = []
    for language in dict.fromkeys(args.languages):
        script = Path(__file__).resolve().with_name(f'sarvam_reasoning_judge_{language}.py')
        result = subprocess.run([sys.executable, '-u', str(script), *flags])
        if result.returncode:
            failed.append(language)
    if failed:
        print('Incomplete/failed languages: ' + ', '.join(failed))
        return 1
    print('All requested ' + 'language runs' + ' completed.')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nStopped. Saved judgments remain available for resumption.', file=sys.stderr)
        raise SystemExit(130)
