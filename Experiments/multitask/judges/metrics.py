"""Ordinal-score metrics and paired, stratified sample-cluster bootstrap."""
from collections import defaultdict
import warnings
import numpy as np
from sklearn.metrics import cohen_kappa_score,mean_absolute_error

def score_metrics(a,b):
    if len(a)!=len(b):raise ValueError('Unmatched score arrays')
    if not len(a):return {k:None for k in ('mae','quadratic_kappa','exact_agreement','within_one_point','signed_bias')}
    a=np.asarray(a);b=np.asarray(b)
    if any(type(v) not in (int,np.int64,np.int32) or not 0<=v<=5 for v in [*a,*b]):raise ValueError('Metrics require integer scores 0–5')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        k=cohen_kappa_score(a,b,labels=list(range(6)),weights='quadratic')
    return {'mae':float(mean_absolute_error(a,b)),'quadratic_kappa':float(k) if np.isfinite(k) else None,
            'exact_agreement':float(np.mean(a==b)),'within_one_point':float(np.mean(np.abs(a-b)<=1)),
            'signed_bias':float(np.mean(a-b))}

def mean_ci(values,items,repetitions=1000,seed=42):
    """95% percentile CI. Keep both Gemma responses together within a sample.

    Resample sample IDs within each language/task cell. Assumes distinct sample
    IDs are independent within cells; unknown cross-language alignment is not modeled.
    """
    if len(values)!=len(items):raise ValueError('Bootstrap values/keys mismatch')
    if repetitions<100:raise ValueError('Use at least 100 bootstrap repetitions')
    if len({i['sample_id'] for i in items})<2:return (None,None)
    strata=defaultdict(lambda:defaultdict(list))
    for value,item in zip(values,items):strata[(item['language_code'],item['task'])][item['sample_id']].append(value)
    rng=np.random.default_rng(seed);totals=np.zeros(repetitions);counts=np.zeros(repetitions)
    for clusters in strata.values():
        sums=np.array([sum(v) for v in clusters.values()]);sizes=np.array([len(v) for v in clusters.values()])
        indices=rng.integers(0,len(sums),size=(repetitions,len(sums)))
        totals+=sums[indices].sum(axis=1);counts+=sizes[indices].sum(axis=1)
    lo,hi=np.quantile(totals/counts,[.025,.975])
    return float(lo),float(hi)
