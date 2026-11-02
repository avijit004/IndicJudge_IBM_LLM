"""Convenient commands for the complete 4,000-response reference-free experiment."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
ACTIVE=ROOT/'Experiments/multitask/judges'
CANDIDATES=ROOT/'Experiments/multitask/candidate'
SNAPSHOT=ACTIVE/'data/inputs_4000.json'
OUTPUT=ACTIVE/'outputs/judges_4000'
ANALYSIS=ACTIVE/'analysis_4000'


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    for name in ('candidates','check','status','analyze','dashboard'):sub.add_parser(name)
    judge=sub.add_parser('judge');judge.add_argument('model',choices=('sarvam','qwen','ministral','all'))
    args,extra=p.parse_known_args(argv)
    reserved=('--snapshot','--output-dir','--analysis-dir')
    if any(x==k or x.startswith(k+'=') for x in extra for k in reserved):
        p.error('This runner selects the complete 4,000-response run. Use experiment scripts directly for a different run.')
    if args.command=='candidates':
        if extra:p.error('candidates takes no additional arguments')
        csv.field_size_limit(16*1024*1024)
        for model in ('gemma26b','gemma31b'):
            path=CANDIDATES/f'{model}_responses.csv'
            with path.open(encoding='utf-8-sig',newline='') as f:count=sum(1 for _ in csv.DictReader(f))
            print(f'{model}: {count} responses\n  {path}')
        return 0
    snapshot=json.loads(SNAPSHOT.read_text())
    if len(snapshot.get('items',[]))!=4000 or snapshot.get('missing_candidates'):
        p.error('The complete 4,000-response snapshot is missing or incomplete; run the replacement merge first.')
    common=['--snapshot',str(SNAPSHOT),'--output-dir',str(OUTPUT)]
    if args.command=='dashboard':
        env=dict(os.environ,RF_ANALYSIS_DIR=str(ANALYSIS),RF_SNAPSHOT=str(SNAPSHOT))
        command=[sys.executable,'-m','streamlit','run',str(ACTIVE/'dashboard.py'),*extra]
        return subprocess.call(command,cwd=ROOT,env=env)
    scripts={'check':'run_all.py','status':'status.py','analyze':'analyze.py'}
    if args.command=='judge':script='run_all.py' if args.model=='all' else f'judge_{args.model}.py'
    else:script=scripts[args.command]
    command=[sys.executable,'-u',str(ACTIVE/script),*common]
    if args.command=='check':command+=['--check-data']
    if args.command=='analyze':command+=['--analysis-dir',str(ANALYSIS)]
    return subprocess.call(command+extra,cwd=ROOT)

if __name__=='__main__':
    try:raise SystemExit(main())
    except KeyboardInterrupt:raise SystemExit(130)
    except (OSError,ValueError) as exc:print(f'ERROR: {exc}',file=sys.stderr);raise SystemExit(1)
