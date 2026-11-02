"""Read-only validated counts; no model calls."""
import argparse
from collections import Counter
from core import add_args,load_config,load_snapshot,validated_results
from utils import JUDGES
from transport import cli_entry

def main():
    p=argparse.ArgumentParser(description=__doc__);add_args(p);p.add_argument('--detail',action='store_true');args=p.parse_args()
    s=load_snapshot(args.snapshot);c=load_config(args.config)
    print(f'Frozen responses: {len(s["items"])}; missing/excluded candidates: {len(s["missing_candidates"])}')
    target=Counter((r['language_code'],r['task']) for r in s['items'])
    for name in JUDGES:
        rows=validated_results(args.output_dir,s,c,name)
        print(f'{name}: {len(rows)}/{len(s["items"])}; remaining {len(s["items"])-len(rows)}')
        if args.detail:
            counts=Counter((r['language_code'],r['task']) for r in rows)
            for cell,n in sorted(target.items()):print(f'  {cell[0]} {cell[1]}: {counts[cell]}/{n}')
    return 0
if __name__=='__main__':raise SystemExit(cli_entry(main))
