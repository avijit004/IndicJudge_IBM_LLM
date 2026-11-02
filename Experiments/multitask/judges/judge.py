"""Run one judge against the immutable reference-free candidate snapshot."""
import argparse
from pathlib import Path
import time
import requests
from core import (add_args, load_config, load_snapshot, select_items, results_path, validated_results,
                  request_for, run_fingerprint, FIELDS)
from utils import JUDGES, CANDIDATES, canonical, atomic_json
from rubric import LANGUAGES, TASKS
from transport import Client, output_lock, atomic_rows, now, record_error, parse_judgment, cli_entry


def run(args, judge, client_factory=Client):
    snapshot=load_snapshot(args.snapshot);config=load_config(args.config)
    items=select_items(snapshot,args)
    if args.limit is not None and args.limit<1:raise ValueError('--limit must be positive')
    if not items:raise ValueError('No candidate responses match this selection')
    existing=validated_results(args.output_dir,snapshot,config,judge)
    for item in items:request_for(item,config,judge)
    print(f'{judge}: {len(items)} selected; {len(snapshot["items"])} frozen candidates; {len(snapshot["missing_candidates"])} excluded from the original target.')
    if args.check_data:
        print('Offline input/payload/checkpoint validation passed; no API calls.');return 0
    path=results_path(args.output_dir,judge)
    with output_lock(path):
        existing=validated_results(args.output_dir,snapshot,config,judge)
        done={r['evaluation_id'] for r in existing}
        todo=[r for r in items if r['evaluation_id'] not in done]
        succeeded=failed=0
        if todo:
            client=client_factory(config['models'][judge],config);client.preflight()
            fingerprint=run_fingerprint(snapshot,config,judge)
            atomic_json(Path(args.output_dir)/f'{judge}_run.json',{'run_fingerprint':fingerprint,'snapshot_hash':snapshot['snapshot_hash'],
                'judge_model':config['models'][judge],'model_record':config.get('model_records',{}).get(judge,{}),'updated_at':now()})
            for item in todo[:args.limit]:
                print(f'{judge}: {item["sample_id"]} / {item["source_model"]}',flush=True)
                payload=request_for(item,config,judge);start=time.monotonic()
                try:
                    result=client.complete(payload);verdict=parse_judgment(result['response'])
                    if result['finish_reason']!='stop':raise ValueError('Truncated judgment')
                    row={k:item[k] for k in ('evaluation_id','sample_id','source_model','task','language_code','item_hash')}
                    row.update(snapshot_hash=snapshot['snapshot_hash'],judge_model=judge,run_fingerprint=fingerprint,
                        request_json=canonical(payload),overall_score=verdict['overall_score'],justification=verdict['justification'],
                        raw_judge_response=result['response'],judge_reasoning=result['reasoning_content'],finish_reason=result['finish_reason'],
                        usage_json=result['usage_json'],elapsed_seconds=f'{time.monotonic()-start:.3f}',created_at=now())
                except (ValueError,RuntimeError,requests.RequestException) as exc:
                    failed+=1;record_error(path,item['evaluation_id'],judge,exc);print(f'FAILED: {exc}',flush=True);continue
                atomic_rows(path,FIELDS,existing+[row]);existing.append(row);succeeded+=1
        remaining=len(todo)-succeeded
        print(f'Previously completed: {len(items)-len(todo)} | Saved: {succeeded} | Failed: {failed} | Remaining in selection: {remaining}')
        return int(remaining>0)


def main(argv=None,fixed_model=None):
    p=argparse.ArgumentParser(description=__doc__);add_args(p)
    if fixed_model is None:p.add_argument('--model',required=True,choices=JUDGES)
    p.add_argument('--languages',nargs='+',choices=tuple(LANGUAGES))
    p.add_argument('--tasks',nargs='+',choices=TASKS)
    p.add_argument('--source-models',choices=('both',)+CANDIDATES,default='both')
    p.add_argument('--limit',type=int)
    p.add_argument('--check-data',action='store_true')
    args=p.parse_args(argv)
    return run(args,fixed_model or args.model)

if __name__=='__main__':raise SystemExit(cli_entry(main))
