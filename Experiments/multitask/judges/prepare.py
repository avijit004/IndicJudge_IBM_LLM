"""Freeze available validated responses. Reference fields never enter the judge snapshot."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
from utils import ROOT, CANDIDATES, digest, atomic_json
from rubric import task_input, TASKS, LANGUAGES
from transport import cli_entry, output_lock


def load_source(source):
    result = subprocess.run([sys.executable, '-B', str(ROOT/'source_export.py'), '--source-dir', str(source)], capture_output=True, text=True)
    if result.returncode:
        raise ValueError('Source validation failed:\n' + result.stderr[-4000:] + result.stdout[-1000:])
    return json.loads(result.stdout)


def freeze(export):
    manifest = export['manifest']
    samples = {s['sample_id']: s for s in manifest['samples']}
    if len(samples) != 2000 or Counter((s['task'], s['language_code']) for s in samples.values()) != Counter({(t,l):50 for t in TASKS for l in LANGUAGES}):
        raise ValueError('This experiment requires the 2,000-input, ten-language, four-task selection')
    items = []; keys = set()
    for model in CANDIDATES:
        for row in export['candidates'][model]:
            sample = samples[row['sample_id']]
            key = (sample['sample_id'], model)
            if key in keys or row['source_model'] != model or row['finish_reason'] != 'stop' or not row['response'].strip():
                raise ValueError('Duplicate or invalid source candidate')
            keys.add(key)
            item = {
                'evaluation_id': digest(list(key))[:24], 'sample_id': sample['sample_id'],
                'source_model': model, 'task': sample['task'], 'language_code': sample['language_code'],
                'task_input': task_input(sample['task'], sample['input']),
                'candidate_response': row['response'],
                'candidate_reasoning': row['reasoning_content'] if sample['task']=='reasoning' else '',
                'source_response_hash': row['response_hash'],
                'source_generation_fingerprint': row['run_fingerprint'],
                'generation_parameters': {k:v for k,v in json.loads(row['request_json']).items() if k not in ('messages',)},
            }
            item['item_hash'] = digest(item)
            items.append(item)
    items.sort(key=lambda r: r['evaluation_id'])  # Same mixed, deterministic order for every judge.
    missing = [{'sample_id':s, 'source_model':m, 'task':samples[s]['task'], 'language_code':samples[s]['language_code']} for s in sorted(samples) for m in CANDIDATES if (s,m) not in keys]
    body = {'version':1, 'source_manifest_hash':manifest['manifest_hash'], 'original_candidate_target':4000,
            'missing_candidates':missing, 'items':items}
    return dict(body, snapshot_hash=digest(body))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',type=Path,default=ROOT.parent/'candidate/multitask_50_samples')
    p.add_argument('--snapshot',type=Path,default=ROOT/'data/inputs.json')
    p.add_argument('--expected-responses',type=int,default=3998)
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args(argv)
    snapshot=freeze(load_source(args.source_dir))
    n=len(snapshot['items'])
    print(f'Validated responses: {n}/4000; missing: {len(snapshot["missing_candidates"])}; expected for this run: {args.expected_responses}')
    if not 1 <= args.expected_responses <= 4000 or n != args.expected_responses:
        raise ValueError('Unexpected coverage. Run preparation on the machine with the completed candidate files; no snapshot written.')
    if len(snapshot['missing_candidates']) <= 20:
        for missing in snapshot['missing_candidates']:
            print(f"Excluded: {missing['source_model']} / {missing['sample_id']}")
    if args.check_only: return 0
    with output_lock(args.snapshot):
        if args.snapshot.exists():
            if json.loads(args.snapshot.read_text()) != snapshot:
                raise ValueError('Snapshot already exists with different candidates; use a separate experiment copy')
            print('Existing snapshot matches; unchanged.'); return 0
        atomic_json(args.snapshot, snapshot)
    print(f'Saved reference-free snapshot: {args.snapshot}. Judges will each score {n} responses.')
    return 0

if __name__=='__main__': raise SystemExit(cli_entry(main))
