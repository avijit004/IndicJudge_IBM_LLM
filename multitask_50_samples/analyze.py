"""Export validated scores and descriptive comparisons. No model calls."""
import json
from pathlib import Path
from itertools import combinations
from common import (ROOT,CANDIDATES,JUDGES,candidate_rows,judged_rows,load_config,atomic_rows,output_lock,cli_entry)
from data import load_manifest
from prompts import TASKS


def main():
    manifest=load_manifest();config=load_config();lookup={}
    counts={}
    for model in CANDIDATES:
        rows=candidate_rows(ROOT/'outputs',model,manifest,config)
        counts[model]=len(rows);lookup.update({(r['sample_id'],model):r for r in rows})
    judgments={m:judged_rows(ROOT/'outputs',m,manifest,config,lookup) for m in JUDGES}
    scores={(r['sample_id'],r['source_model'],m):int(r['overall_score']) for m,rows in judgments.items() for r in rows}
    samples={s['sample_id']:s for s in manifest['samples']}
    wide=[]
    for sample in manifest['samples']:
        for model in CANDIDATES:
            key=(sample['sample_id'],model)
            wide.append(dict(sample_id=key[0],source_model=model,task=sample['task'],language_code=sample['language_code'],
                             source_row=sample['source_row'],candidate_saved=key in lookup,
                             **{m:scores.get((*key,m),'') for m in JUDGES}))
    means=[];pairs=[]
    for task in ('all',)+tuple(TASKS):
        subset=[r for r in wide if task=='all' or r['task']==task]
        for model in JUDGES:
            values=[r[model] for r in subset if r[model]!='']
            means.append(dict(task=task,judge_model=model,completed=len(values),expected=len(subset),
                              mean_score=sum(values)/len(values) if values else ''))
        for a,b in combinations(JUDGES,2):
            matched=[(r[a],r[b]) for r in subset if r[a]!='' and r[b]!='']
            n=len(matched)
            pairs.append(dict(task=task,judge_a=a,judge_b=b,matched=n,expected=len(subset),
                              exact_agreement=sum(x==y for x,y in matched)/n if n else '',
                              mean_absolute_score_difference=sum(abs(x-y) for x,y in matched)/n if n else '',
                              within_one_point=sum(abs(x-y)<=1 for x,y in matched)/n if n else ''))
    audit_path=ROOT/'outputs'/'import_provenance.json'
    audit=json.loads(audit_path.read_text()) if audit_path.exists() else {}
    report={'selected_inputs':len(samples),'candidate_counts':counts,'judge_counts':{m:len(r) for m,r in judgments.items()},
            'candidate_target_each':len(samples),'judge_target_each':len(wide),
            'complete':all(n==len(samples) for n in counts.values()) and all(len(r)==len(wide) for r in judgments.values()),
            'import_provenance':audit,'interpretation':'Partial means use available ratings; pairwise statistics use matched ratings only. Agreement is not accuracy. Fractions range from 0 to 1.'}
    target=ROOT/'analysis'
    with output_lock(target/'export'):
        atomic_rows(target/'scores_by_sample.csv',list(wide[0]),wide)
        atomic_rows(target/'mean_scores.csv',list(means[0]),means)
        atomic_rows(target/'pairwise_agreement.csv',list(pairs[0]),pairs)
        (target/'coverage.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'Analysis exported for {len(wide)} candidate keys; complete={report["complete"]}. Missing scores are blank.')
    return 0

if __name__=='__main__':raise SystemExit(cli_entry(main))
