"""Snapshot, configuration and result validation for the separate reference-free run."""
import argparse
import json
from pathlib import Path
from utils import ROOT, CANDIDATES, JUDGES, canonical, digest
from rubric import TASKS, LANGUAGES, RUBRIC_VERSION, RUBRICS, JUDGE_PROMPT, JUDGMENT_SCHEMA, task_input, judge_messages
from transport import read_rows, payload_for, parse_judgment

FIELDS = ['evaluation_id','sample_id','source_model','task','language_code','item_hash','snapshot_hash',
          'judge_model','run_fingerprint','request_json','overall_score','justification','raw_judge_response',
          'judge_reasoning','finish_reason','usage_json','elapsed_seconds','created_at']
ITEM_FIELDS = {'evaluation_id','sample_id','source_model','task','language_code','task_input','candidate_response',
               'candidate_reasoning','source_response_hash','source_generation_fingerprint','generation_parameters','item_hash'}

def load_snapshot(path):
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    body={k:v for k,v in data.items() if k!='snapshot_hash'}
    if set(data) != {'version','source_manifest_hash','original_candidate_target','missing_candidates','items','snapshot_hash'} or data['version']!=1 or digest(body)!=data['snapshot_hash']:
        raise ValueError('Invalid or changed snapshot')
    keys=set(); ids=set()
    for item in data['items']:
        if set(item)!=ITEM_FIELDS or item['task'] not in TASKS or item['language_code'] not in LANGUAGES or item['source_model'] not in CANDIDATES:
            raise ValueError('Invalid snapshot item fields')
        key=(item['sample_id'],item['source_model'])
        if key in keys or item['evaluation_id'] in ids or item['evaluation_id']!=digest(list(key))[:24]:
            raise ValueError('Duplicate/invalid candidate identity')
        if item['item_hash']!=digest({k:v for k,v in item.items() if k!='item_hash'}):
            raise ValueError('Candidate content changed')
        if task_input(item['task'],item['task_input'])!=item['task_input']:
            raise ValueError('Unexpected task fields: possible reference leakage')
        if not isinstance(item['candidate_response'],str) or not item['candidate_response'].strip() or not isinstance(item['candidate_reasoning'],str):
            raise ValueError('Invalid candidate text')
        if item['task']!='reasoning' and item['candidate_reasoning']:
            raise ValueError('Internal reasoning must be absent for non-reasoning tasks')
        keys.add(key);ids.add(item['evaluation_id'])
    missing_keys=[]
    for row in data['missing_candidates']:
        if set(row)!={'sample_id','source_model','task','language_code'} or row['source_model'] not in CANDIDATES or row['task'] not in TASKS or row['language_code'] not in LANGUAGES:
            raise ValueError('Invalid missing candidate record')
        missing_keys.append((row['sample_id'],row['source_model']))
    if len(set(missing_keys))!=len(missing_keys) or keys.intersection(missing_keys) or len(keys)+len(missing_keys)!=data['original_candidate_target']:
        raise ValueError('Invalid exclusion accounting')
    return data


def load_config(path):
    c=json.loads(Path(path).read_text())
    if set(c['models'])!=set(JUDGES):raise ValueError('Config must define all three judges')
    if type(c['minimum_context_tokens']) is not int or c['minimum_context_tokens']<1 or type(c['timeout_seconds']) is not int or c['timeout_seconds']<1:
        raise ValueError('Invalid context or timeout')
    for name,model in c['models'].items():
        if model['role']!='judge' or not model['alias'] or model['api_url_env']!='RF_'+name.upper()+'_API_URL':raise ValueError('Invalid judge identity')
        params=model['parameters']
        if set(params)&{'messages','model','response_format','stream'}:raise ValueError('Parameters cannot replace payload/schema')
        if type(params.get('max_tokens')) is not int or not 0<params['max_tokens']<c['minimum_context_tokens']:raise ValueError('Invalid output limit')
        if type(params.get('seed')) is not int:raise ValueError('An explicit integer seed is required')
        if type(params.get('reasoning_budget_tokens')) is not int or not 0<=params['reasoning_budget_tokens']<params['max_tokens']:raise ValueError('Invalid thinking budget')
    return c


def run_fingerprint(snapshot, config, judge):
    return digest({'snapshot':snapshot['snapshot_hash'],'rubric_version':RUBRIC_VERSION,'rules':JUDGE_PROMPT,
                   'rubrics':RUBRICS,'schema':JUDGMENT_SCHEMA,'rubric_source':digest((ROOT/'rubric.py').read_text(encoding='utf-8')),'model':config['models'][judge],
                   'model_record':config.get('model_records',{}).get(judge,{})})

def request_for(item,config,judge):
    return payload_for(judge_messages(item),config['models'][judge],judging=True)

def results_path(output,judge):return Path(output)/f'{judge}_results.csv'

def validated_results(output,snapshot,config,judge):
    rows=read_rows(results_path(output,judge),FIELDS)
    items={r['evaluation_id']:r for r in snapshot['items']};seen=set()
    fingerprint=run_fingerprint(snapshot,config,judge)
    for row in rows:
        item=items.get(row['evaluation_id'])
        if item is None or row['evaluation_id'] in seen:raise ValueError('Unknown/duplicate judgment')
        seen.add(row['evaluation_id'])
        if any(row[k]!=item[k] for k in ('sample_id','source_model','task','language_code','item_hash')):raise ValueError('Judgment identity/content mismatch')
        if row['snapshot_hash']!=snapshot['snapshot_hash'] or row['run_fingerprint']!=fingerprint or row['judge_model']!=judge:
            raise ValueError('Judgment settings/rubric/snapshot changed; use a separate experiment')
        if row['request_json']!=canonical(request_for(item,config,judge)):raise ValueError('Judgment request changed')
        parsed=parse_judgment(row['raw_judge_response'])
        if row['finish_reason']!='stop' or row['overall_score']!=str(parsed['overall_score']) or row['justification']!=parsed['justification']:
            raise ValueError('Invalid saved score')
    return rows


def add_args(p):
    p.add_argument('--snapshot',type=Path,default=ROOT/'data/inputs.json')
    p.add_argument('--config',type=Path,default=ROOT/'config.json')
    p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/judges')

def select_items(snapshot,args):
    return [r for r in snapshot['items'] if (not args.languages or r['language_code'] in args.languages)
            and (not args.tasks or r['task'] in args.tasks)
            and (args.source_models=='both' or r['source_model']==args.source_models)]
