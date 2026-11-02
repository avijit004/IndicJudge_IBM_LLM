"""Separate descriptive agreement, candidate scores, and human-validated judge rankings."""
import argparse
from collections import Counter,defaultdict
from itertools import combinations
import json
import hashlib
from pathlib import Path
from core import add_args,load_config,load_snapshot,validated_results,run_fingerprint
from utils import ROOT,JUDGES,CANDIDATES,atomic_json
from transport import atomic_rows,read_rows,output_lock,cli_entry
from metrics import score_metrics,mean_ci
from review import HUMAN_FIELDS,PROTOCOLS

AGREEMENT_FIELDS=['language_code','task','judge_a','judge_b','n','n_clusters','eligible_responses','mean_absolute_score_difference','quadratic_kappa','exact_agreement','within_one_point']
QUALITY_FIELDS=['language_code','task','judge','descriptive_rank_by_mae','n','n_clusters','eligible_responses','human_labeled','protocol','mae','mae_ci_low','mae_ci_high','quadratic_kappa','exact_agreement','within_one_point','signed_bias','status']
DIFFERENCE_FIELDS=['language_code','task','judge_a','judge_b','n','mae_a_minus_b','ci_low','ci_high']
CANDIDATE_FIELDS=['language_code','task','source_model','judge','n','matched_samples','mean_score']


def human_scores(path,snapshot):
    if path is None:return {},None
    rows=read_rows(path,HUMAN_FIELDS)
    if not Path(path).exists():raise ValueError('Human ratings file does not exist')
    items={i['evaluation_id']:i for i in snapshot['items']};seen=set();scores={};protocols=set()
    for r in rows:
        key=r['evaluation_id']
        if key in seen or key not in items or r['item_hash']!=items[key]['item_hash']:raise ValueError('Duplicate, unknown or changed human-review item')
        seen.add(key)
        if r['status']!='adjudicated':continue
        if r['human_score'] not in tuple(str(x) for x in range(6)):raise ValueError('Adjudicated scores must be integers 0–5')
        if r['protocol'] not in PROTOCOLS or len(set(x.strip() for x in r['reviewer_ids'].split('|') if x.strip()))<2 or not r['notes'].strip():
            raise ValueError('Adjudication requires protocol, at least two reviewer IDs separated by |, and a note')
        scores[key]=int(r['human_score']);protocols.add(r['protocol'])
    if len(protocols)>1:raise ValueError('Do not mix reference-free and reference-assisted human protocols in one analysis')
    return scores,next(iter(protocols),None)


def groups(items):
    result={('all','all'):items}
    for i in items:
        for key in ((i['language_code'],'all'),('all',i['task']),(i['language_code'],i['task'])):
            result.setdefault(key,[]).append(i)
    return sorted(result.items())


def compute(snapshot,results,human=None,protocol=None,repetitions=1000,min_clusters=30):
    human=human or {}; scores={j:{r['evaluation_id']:int(r['overall_score']) for r in rows} for j,rows in results.items()}
    agreement=[];quality=[];differences=[];candidates=[]
    for (lang,task),eligible in groups(snapshot['items']):
        base={'language_code':lang,'task':task}
        # Identical keys for all three judges; never compare means on different subsets.
        matched=[i for i in eligible if all(i['evaluation_id'] in scores[j] for j in JUDGES)]
        for a,b in combinations(JUDGES,2):
            m=score_metrics([scores[a][i['evaluation_id']] for i in matched],[scores[b][i['evaluation_id']] for i in matched])
            agreement.append(dict(base,judge_a=a,judge_b=b,n=len(matched),n_clusters=len({i['sample_id'] for i in matched}),eligible_responses=len(eligible),
                mean_absolute_score_difference=m['mae'],**{k:m[k] for k in ('quadratic_kappa','exact_agreement','within_one_point')}))
        by_sample=defaultdict(dict)
        for i in matched:by_sample[i['sample_id']][i['source_model']]=i
        paired={k:v for k,v in by_sample.items() if set(v)==set(CANDIDATES)}
        for model in CANDIDATES:
            for judge in JUDGES:
                values=[scores[judge][v[model]['evaluation_id']] for v in paired.values()]
                candidates.append(dict(base,source_model=model,judge=judge,n=len(values),matched_samples=len(paired),mean_score=sum(values)/len(values) if values else None))
        rated=[i for i in matched if i['evaluation_id'] in human]
        if not rated:continue
        nclusters=len({i['sample_id'] for i in rated});gold=[human[i['evaluation_id']] for i in rated]
        group_rows=[];losses={}
        for judge in JUDGES:
            predicted=[scores[judge][i['evaluation_id']] for i in rated]
            m=score_metrics(predicted,gold);losses[judge]=[abs(a-b) for a,b in zip(predicted,gold)]
            low,high=mean_ci(losses[judge],rated,repetitions)
            group_rows.append(dict(base,judge=judge,descriptive_rank_by_mae=None,n=len(rated),n_clusters=nclusters,
                eligible_responses=len(eligible),human_labeled=sum(i['evaluation_id'] in human for i in eligible),protocol=protocol,
                **m,mae_ci_low=low,mae_ci_high=high,status='too_few_clusters' if nclusters<min_clusters else 'descriptive_only'))
        rank_values=sorted({r['mae'] for r in group_rows})
        for r in group_rows:
            if nclusters>=min_clusters:r['descriptive_rank_by_mae']=rank_values.index(r['mae'])+1
        quality.extend(group_rows)
        for a,b in combinations(JUDGES,2):
            delta=[x-y for x,y in zip(losses[a],losses[b])];low,high=mean_ci(delta,rated,repetitions)
            differences.append(dict(base,judge_a=a,judge_b=b,n=len(rated),mae_a_minus_b=sum(delta)/len(delta),ci_low=low,ci_high=high))
    return agreement,quality,differences,candidates


def main():
    p=argparse.ArgumentParser(description=__doc__);add_args(p)
    p.add_argument('--human-ratings',type=Path)
    p.add_argument('--analysis-dir',type=Path,default=ROOT/'analysis')
    p.add_argument('--bootstrap',type=int,default=1000)
    p.add_argument('--min-clusters',type=int,default=30)
    args=p.parse_args()
    if args.bootstrap<100 or args.min_clusters<2:raise ValueError('Use bootstrap >=100 and min-clusters >=2')
    s=load_snapshot(args.snapshot);c=load_config(args.config)
    results={j:validated_results(args.output_dir,s,c,j) for j in JUDGES}
    gold,protocol=human_scores(args.human_ratings,s)
    tables=compute(s,results,gold,protocol,args.bootstrap,args.min_clusters)
    names=['judge_agreement.csv','judge_leaderboard.csv','paired_mae_differences.csv','candidate_scores.csv']
    fields=[AGREEMENT_FIELDS,QUALITY_FIELDS,DIFFERENCE_FIELDS,CANDIDATE_FIELDS]
    report={'snapshot_hash':s['snapshot_hash'],'candidate_count':len(s['items']),'excluded_candidates':s['missing_candidates'],
        'judge_counts':{j:len(v) for j,v in results.items()},'human_adjudicated':len(gold),'human_protocol':protocol,
        'matched_for_all_judges':len(set.intersection(*[{r['evaluation_id'] for r in rows} for rows in results.values()])),
        'judge_run_fingerprints':{j:run_fingerprint(s,c,j) for j in JUDGES},
        'human_ratings_sha256':hashlib.sha256(args.human_ratings.read_bytes()).hexdigest() if args.human_ratings else None,
        'bootstrap_repetitions':args.bootstrap,'bootstrap_seed':42,'min_clusters_for_descriptive_rank':args.min_clusters,
        'primary_metric':'MAE against adjudicated human overall_score; smaller is better',
        'weighting':'Response-weighted within each reported group; not equal-cell macro averaging.',
        'limitations':'First 50 source-ordered records, two related candidate responses per sample, no random population sample. Cross-language alignment unknown. 95% bootstrap intervals are pointwise and not multiple-comparison corrected. No automatic best-judge declaration.'}
    args.analysis_dir.mkdir(parents=True,exist_ok=True)
    with output_lock(args.analysis_dir/'export'):
        for name,field,rows in zip(names,fields,tables):atomic_rows(args.analysis_dir/name,field,rows)
        atomic_json(args.analysis_dir/'coverage.json',report)
    print(f'Analysis saved to {args.analysis_dir}; adjudicated human ratings: {len(gold)}.')
    if not gold:print('Judge-quality leaderboard is empty until human ratings exist. Agreement and candidate scores are descriptive only.')
    return 0
if __name__=='__main__':raise SystemExit(cli_entry(main))
