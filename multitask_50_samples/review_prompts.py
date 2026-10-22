"""Print a selected sample's actual messages offline. No generation or judging occurs."""
import argparse
import json
from common import ROOT, CANDIDATES, JUDGES, load_config, payload_for, cli_entry
from data import load_manifest, select_samples
from prompts import LANGUAGES, TASKS, generation_messages, judge_messages


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--language', choices=tuple(LANGUAGES), default='bn')
    parser.add_argument('--task', choices=TASKS, default='reasoning')
    parser.add_argument('--sample', type=int, choices=range(1, 6), default=1)
    parser.add_argument('--model', choices=CANDIDATES + JUDGES, default='sarvam')
    parser.add_argument('--config', default=ROOT / 'config.json')
    parser.add_argument('--manifest', default=ROOT / 'data' / 'selection.json')
    args = parser.parse_args(argv)
    sample = select_samples(load_manifest(args.manifest), [args.language], [args.task])[args.sample-1]
    model = load_config(args.config)['models'][args.model]
    if args.model in CANDIDATES:
        messages = generation_messages(sample)
    else:
        messages = judge_messages(sample, {'response': '[Candidate visible response will be inserted here.]',
                                          'reasoning_content': '[Candidate internal reasoning, if available.]'})
    print(json.dumps(payload_for(messages, model, args.model in JUDGES), ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(cli_entry(main))
