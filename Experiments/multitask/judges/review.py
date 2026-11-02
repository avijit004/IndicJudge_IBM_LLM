"""Offline human review export; references are optional and never read by judge.py."""
import argparse
import json
from pathlib import Path
from core import load_snapshot
from utils import ROOT,atomic_json
from transport import atomic_rows,output_lock,cli_entry
from rubric import evaluation_record,JUDGE_PROMPT,RUBRICS
from prepare import load_source

HUMAN_FIELDS=['evaluation_id','item_hash','human_score','status','reviewer_ids','protocol','notes']
PROTOCOLS=('human_reference_free','human_reference_assisted')

def export(snapshot,directory,source=None):
    directory=Path(directory)
    with output_lock(directory/'export'):
        names=['reading_pack.json','reviewer_1.csv','reviewer_2.csv','adjudicated.csv','rubric.txt']
        if source is not None:names+=['references_for_review.json']
        if any((directory/n).exists() for n in names):raise ValueError('Review files already exist; refusing to overwrite human work. Use a new --review-dir.')
        references=None
        if source is not None:
            exported=load_source(source)
            if exported['manifest']['manifest_hash']!=snapshot['source_manifest_hash']:raise ValueError('Reference selection differs from frozen judging selection')
            samples={s['sample_id']:s for s in exported['manifest']['samples']}
            hashes={(r['sample_id'],m):r['response_hash'] for m,rows in exported['candidates'].items() for r in rows}
            references=[]
            for item in snapshot['items']:
                if hashes.get((item['sample_id'],item['source_model']))!=item['source_response_hash']:raise ValueError('Original candidate changed since freeze')
                references.append({'evaluation_id':item['evaluation_id'],'task':item['task'], 'reference':samples[item['sample_id']]['reference']})
        reading=[dict(evaluation_id=i['evaluation_id'],language_code=i['language_code'],**evaluation_record(i)) for i in snapshot['items']]
        atomic_json(directory/'reading_pack.json',reading)
        fields=['evaluation_id','item_hash','score','reviewer_id','status','notes']
        rows=[{k:(i[k] if k in ('evaluation_id','item_hash') else '') for k in fields} for i in snapshot['items']]
        for n in (1,2):atomic_rows(directory/f'reviewer_{n}.csv',fields,rows)
        rows=[{k:(i[k] if k in ('evaluation_id','item_hash') else '') for k in HUMAN_FIELDS} for i in snapshot['items']]
        atomic_rows(directory/'adjudicated.csv',HUMAN_FIELDS,rows)
        (directory/'rubric.txt').write_text(JUDGE_PROMPT+'\n\n'+'\n\n'.join(RUBRICS.values()),encoding='utf-8')
        if references is not None:atomic_json(directory/'references_for_review.json',references)
    print(f'Blank review package saved to {directory}. No scores have been supplied or inferred.')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',type=Path,default=ROOT/'data/inputs.json')
    p.add_argument('--review-dir',type=Path,default=ROOT/'review')
    p.add_argument('--include-references',action='store_true')
    p.add_argument('--source-dir',type=Path,default=ROOT.parent/'candidate/multitask_50_samples')
    args=p.parse_args();export(load_snapshot(args.snapshot),args.review_dir,args.source_dir if args.include_references else None);return 0
if __name__=='__main__':raise SystemExit(cli_entry(main))
