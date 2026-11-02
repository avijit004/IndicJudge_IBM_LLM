"""Two matched replacement inputs, four generations, and an auditable 4,000-response snapshot."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
SOURCE=PROJECT/'multitask_50_samples'
sys.path.insert(0,str(SOURCE))
import common as legacy
import data as source_data
from prompts import generation_messages
sys.path.insert(0,str(PROJECT.parent/'judges'))
from utils import atomic_json
from prepare import freeze

SWAPS={'kn':('crosssum-kn-215','crosssum-kn-490'),'ta':('crosssum-ta-174','crosssum-ta-116')}
PLAN=ROOT/'data/replacements.json'
SNAPSHOT=PROJECT.parent/'judges/data/inputs_4000.json'
COMBINED=PROJECT


def write_once(path,value):
    with legacy.output_lock(path):
        if path.exists():
            if source_data.canonical(json.loads(path.read_text()))!=source_data.canonical(value):raise ValueError(f'{path} already exists with different content')
        else:atomic_json(path,value)


def prepare():
    original=source_data.load_manifest(SOURCE/'data/selection.json')
    original_ids={s['sample_id'] for s in original['samples']}
    samples=[]
    for lang,(old,new) in SWAPS.items():
        relative=f'summarization/{lang}.csv';path=PROJECT.parents[2]/'Dataset/dataset_csv'/relative
        if hashlib.sha256(path.read_bytes()).hexdigest()!=original['source_file_hashes'][relative]:
            raise ValueError(f'Dataset changed since original selection: {path}')
        with path.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        row=rows[50]
        if row['id']!=new or f'summarization:{lang}:{old}' not in original_ids or f'summarization:{lang}:{new}' in original_ids:
            raise ValueError('Replacement row/ID does not match the approved swap')
        samples.append(source_data.normalize(row,'summarization',lang,relative,51))
    config=legacy.load_config(SOURCE/'config.json')
    body={'version':1,'source_manifest_hash':original['manifest_hash'],'swaps':SWAPS,'samples':samples,
          'candidate_models':{m:config['models'][m] for m in legacy.CANDIDATES},
          'generation_messages':[generation_messages(s) for s in samples]}
    write_once(PLAN,dict(body,plan_hash=source_data.digest(body)))
    print('Prepared two inputs; generate both with each Gemma (four responses total).')


def plan():
    value=json.loads(PLAN.read_text())
    if value['plan_hash']!=source_data.digest({k:v for k,v in value.items() if k!='plan_hash'}):raise ValueError('Replacement plan changed')
    if value['swaps']!={k:list(v) for k,v in SWAPS.items()}:raise ValueError('Unexpected replacements')
    if value['generation_messages']!=[generation_messages(s) for s in value['samples']]:raise ValueError('Generation prompts changed')
    for s in value['samples']:
        if s['sample_hash']!=source_data.digest({k:v for k,v in s.items() if k!='sample_hash'}):raise ValueError('Changed replacement input')
    return value


def fingerprint(p,model):return source_data.digest({'plan_hash':p['plan_hash'],'model':model,'messages':p['generation_messages']})


def models(p,key):
    model=p['candidate_models'][key]
    return [model,legacy.length_recovery_model(model),*(legacy.format_recovery_model(model,n) for n in (44,45,46))]


def saved(p,key):
    rows=legacy.read_rows(legacy.generation_path(ROOT/'outputs',key),legacy.GEN_FIELDS)
    samples={s['sample_id']:s for s in p['samples']};variants={fingerprint(p,m):m for m in models(p,key)};seen=set()
    for r in rows:
        s=samples.get(r['sample_id']);m=variants.get(r['run_fingerprint'])
        if not s or m is None or r['sample_id'] in seen:raise ValueError('Unknown or duplicate replacement response')
        if any(r[k]!=v for k,v in legacy.metadata(s,key).items()) or r['api_model']!=m['alias']:raise ValueError('Replacement metadata differs')
        if r['request_json']!=source_data.canonical(legacy.payload_for(generation_messages(s),m)):raise ValueError('Replacement request changed')
        if r['finish_reason']!='stop' or not r['response'].strip() or r['response_hash']!=legacy.response_hash(r):raise ValueError('Invalid replacement response')
        seen.add(r['sample_id'])
    return rows


def generate(key,retry_length=False,retry_format=False,check=False,client_factory=legacy.Client):
    p=plan();config=legacy.load_config(SOURCE/'config.json')
    if config['models'][key]!=p['candidate_models'][key]:raise ValueError('Candidate settings differ from the frozen replacement plan')
    path=legacy.generation_path(ROOT/'outputs',key)
    rows=saved(p,key)
    if check:
        print(f'{key}: {len(rows)}/2; offline checks passed.');return 0
    with legacy.output_lock(path):
        rows=saved(p,key);done={r['sample_id'] for r in rows};pending=[s for s in p['samples'] if s['sample_id'] not in done]
        length_ids=legacy.length_failure_ids(path,key) if retry_length else set()
        format_ids=legacy.format_failure_ids(path,key) if retry_format else set()
        if pending:
            client=client_factory(config['models'][key],config);client.preflight()
            for s in pending:
                model=p['candidate_models'][key]
                if s['sample_id'] in format_ids:model=legacy.format_recovery_model(model)
                elif s['sample_id'] in length_ids:model=legacy.length_recovery_model(model)
                payload=legacy.payload_for(generation_messages(s),model)
                print(f'{key}: {s["sample_id"]}',flush=True)
                try:
                    result=client.complete(payload)
                    if result['finish_reason']!='stop' or not result['response'].strip():raise ValueError('Incomplete replacement')
                    row=dict(legacy.metadata(s,key),run_fingerprint=fingerprint(p,model),api_model=model['alias'],
                        request_json=source_data.canonical(payload),created_at=legacy.now(),**result)
                    row['response_hash']=legacy.response_hash(row)
                except (ValueError,RuntimeError,legacy.requests.RequestException) as exc:
                    legacy.record_error(path,s['sample_id'],key,exc);print(f'FAILED: {exc}',flush=True);continue
                legacy.atomic_rows(path,legacy.GEN_FIELDS,rows+[row]);rows.append(row)
    print(f'{key}: {len(rows)}/2 saved')
    return int(len(rows)!=2)


def combined_export():
    p=plan();original=source_data.load_manifest(SOURCE/'data/selection.json');config=legacy.load_config(SOURCE/'config.json')
    if original['manifest_hash']!=p['source_manifest_hash']:raise ValueError('Original selection changed')
    replacements={m:saved(p,m) for m in legacy.CANDIDATES}
    if any(len(v)!=2 for v in replacements.values()):raise ValueError('Generate all four replacement responses before merging')
    removed={f'summarization:{lang}:{ids[0]}' for lang,ids in SWAPS.items()}
    body={k:v for k,v in original.items() if k!='manifest_hash'}
    body['samples']=[s for s in original['samples'] if s['sample_id'] not in removed]+p['samples']
    body['selection_method']='First 50 records, replacing two failed-input pairs with approved row 51 samples'
    body['replacement_plan_hash']=p['plan_hash']
    manifest=dict(body,manifest_hash=source_data.digest(body));candidates={}
    for m in legacy.CANDIDATES:
        original_rows=legacy.candidate_rows(SOURCE/'outputs',m,original,config)
        retained=[r for r in original_rows if r['sample_id'] not in removed]
        candidates[m]=retained+replacements[m]
        if len(candidates[m])!=2000:raise ValueError(f'{m}: expected 1,998 retained + 2 new; found {len(candidates[m])}')
        if {r['sample_id'] for r in candidates[m]}!={s['sample_id'] for s in manifest['samples']}:raise ValueError('Candidate inputs are not matched')
    return {'manifest':manifest,'candidates':candidates}


def merge():
    export=combined_export();snapshot=freeze(export)
    if len(snapshot['items'])!=4000 or snapshot['missing_candidates']:raise ValueError('Incomplete merged snapshot')
    write_once(SNAPSHOT,snapshot)
    combined_dir=COMBINED
    for model, rows in export['candidates'].items():
        rows=sorted(rows,key=lambda r:(r['task'],r['language_code'],r['sample_id']))
        path=combined_dir/f'{model}_responses.csv'
        with legacy.output_lock(path):
            if path.exists():
                if legacy.read_rows(path,legacy.GEN_FIELDS)!=rows:
                    raise ValueError(f'{path} already exists with different content; not overwriting')
            else:
                legacy.atomic_rows(path,legacy.GEN_FIELDS,rows)
        print(f'Combined CSV: {path} ({len(rows)} responses)')
    write_once(ROOT/'data/merge_audit.json',{'plan_hash':plan()['plan_hash'],'snapshot_hash':snapshot['snapshot_hash'],
        'source_manifest_hash':plan()['source_manifest_hash'],'replacement_manifest_hash':export['manifest']['manifest_hash'],
        'swaps':SWAPS,'candidate_counts':{m:len(r) for m,r in export['candidates'].items()},
        'note':'Both original input pairs excluded. Original files unchanged. All four replacements included.'})
    print(f'Saved {SNAPSHOT}: 4,000 matched responses. Use a separate outputs/judges_4000 directory.')


def export_references():
    export=combined_export();snapshot=freeze(export)
    if json.loads(SNAPSHOT.read_text())!=snapshot:raise ValueError('Merged snapshot differs; stop before review')
    samples={s['sample_id']:s for s in export['manifest']['samples']}
    references=[{'evaluation_id':i['evaluation_id'],'task':i['task'],'reference':samples[i['sample_id']]['reference']} for i in snapshot['items']]
    path=PROJECT.parent/'judges/review_4000/references_for_review.json'
    write_once(path,references);print(f'Offline human-review references saved to {path}; never sent to judges.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','generate','status','merge','export-references'))
    parser.add_argument('--model',choices=legacy.CANDIDATES)
    parser.add_argument('--check-data',action='store_true')
    parser.add_argument('--retry-length',action='store_true');parser.add_argument('--retry-format',action='store_true')
    args=parser.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='generate':
        if not args.model:parser.error('generate requires --model')
        return generate(args.model,args.retry_length,args.retry_format,args.check_data)
    elif args.command=='status':
        p=plan()
        for m in legacy.CANDIDATES:print(f'{m}: {len(saved(p,m))}/2')
    elif args.command=='merge':merge()
    else:export_references()
    return 0
if __name__=='__main__':raise SystemExit(legacy.cli_entry(main))
