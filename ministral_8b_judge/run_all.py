"""Run the five-question Ministral evaluation across selected languages."""
import argparse
from pathlib import Path
import subprocess
import sys

LANGUAGES = ('bengali', 'gujarati', 'hindi', 'kannada', 'malayalam',
             'marathi', 'odia', 'punjabi', 'tamil', 'telugu')
DEFAULT_SOURCE_MODEL = 'both'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-data', action='store_true')
    parser.add_argument('--languages', nargs='+', choices=LANGUAGES, default=LANGUAGES)
    parser.add_argument('--question-ids', nargs=5, default=('1', '2', '3', '4', '5'), metavar='ID')
    parser.add_argument('--source-model', choices=('both', '26b', '31b'), default=DEFAULT_SOURCE_MODEL)
    args = parser.parse_args()
    if len(set(args.question_ids)) != 5 or any(not q.strip() for q in args.question_ids):
        parser.error('Choose exactly five distinct, nonempty question IDs.')
    languages = tuple(dict.fromkeys(args.languages))
    per_language = 50 if args.source_model == 'both' else 25
    print(f'Scope: {per_language} judgments/language; {len(languages) * per_language} total.', flush=True)
    flags = ['--question-ids', *args.question_ids, '--source-model', args.source_model]
    if args.check_data:
        flags.append('--check-data')
    failed = []
    for language in languages:
        script = Path(__file__).resolve().with_name(f'ministral_8b_judge_{language}.py')
        result = subprocess.run([sys.executable, '-B', '-u', str(script), *flags])
        if result.returncode:
            failed.append(language)
    if failed:
        print('Incomplete/failed languages: ' + ', '.join(failed))
        return 1
    print('All selected ' + ('data checks' if args.check_data else 'five-question evaluations') + ' completed.')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nStopped. Saved judgments will be reused on rerun.', file=sys.stderr)
        raise SystemExit(130)
