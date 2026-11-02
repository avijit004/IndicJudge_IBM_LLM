"""All inference is mocked; outputs live only in temporary directories."""
import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import core,judge,rubric,prepare,review,analyze,metrics,transport
from utils import ROOT,CANDIDATES,JUDGES,digest,atomic_json,canonical


def fixture():
    samples=[];rows={m:[] for m in CANDIDATES}
    for task in rubric.TASKS:
        for lang in rubric.LANGUAGES:
            for n in range(50):
                inputs={'qa':{'context':'A person has four books.','question':'How many books?'},
                    'reasoning':{'premise':'It rained.','relation':'effect','question':'What happened?','options':{'1':'Ground got wet.','2':'Ground dried.'}},
                    'summarization':{'document':'A school opened. It welcomes children.'},
                    'translation':{'source_lang':lang,'target_lang':'en','source_text':'Source sentence.'}}[task]
                s={'sample_id':f'{task}:{lang}:{n}','task':task,'language_code':lang,'input':inputs,
                   'reference':{'answer':'FORBIDDEN_GOLD_SENTINEL'}};samples.append(s)
                if n==0:
                    for m in CANDIDATES:
                        rows[m].append(dict(sample_id=s['sample_id'],source_model=m,finish_reason='stop',response='Candidate answer.',
                            reasoning_content='Candidate explanation.',response_hash=digest([s['sample_id'],m]),run_fingerprint='fixture',
                            request_json=json.dumps({'messages':[{'content':'Hidden generation instructions'}],'seed':42})))
    return {'manifest':{'samples':samples,'manifest_hash':'fixture_manifest'},'candidates':rows}

class FakeClient:
    calls=[];invalid=False
    def __init__(self,model,config):pass
    def preflight(self):pass
    def complete(self,payload):
        self.calls.append(payload)
        text='{"overall_score":4,"justification":"The source supports the main claim. A minor detail is unclear."}'
        if self.invalid:text='{"overall_score":true,"justification":"Invalid"}'
        return dict(response=text,reasoning_content='Reasoning',finish_reason='stop',usage_json='{}')

class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.source=fixture();cls.snapshot=prepare.freeze(cls.source)
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.out=Path(self.temp.name)
        self.path=self.out/'inputs.json';atomic_json(self.path,self.snapshot)
        self.args=argparse.Namespace(snapshot=self.path,config=ROOT/'config.json',output_dir=self.out/'judges',
            languages=['bn'],tasks=['qa'],source_models='both',limit=None,check_data=False)
        FakeClient.calls=[];FakeClient.invalid=False
    def tearDown(self):self.temp.cleanup()
    def run_judge(self,model='qwen'):
        with contextlib.redirect_stdout(io.StringIO()):return judge.run(self.args,model,FakeClient)
    def test_reference_and_identity_leakage_all_tasks_languages_models(self):
        config=core.load_config(ROOT/'config.json')
        self.assertEqual(len(self.snapshot['items']),80)
        self.assertNotIn('FORBIDDEN_GOLD_SENTINEL',json.dumps(self.snapshot))
        for item in self.snapshot['items']:
            for j in JUDGES:
                payload=core.request_for(item,config,j)
                record=json.loads(payload['messages'][-1]['content'].split('Evaluation record:\n')[1])
                self.assertEqual(set(record['task_input']),set(rubric.INPUT_FIELDS[item['task']]))
                self.assertNotIn('source_model',record);self.assertNotIn('sample_id',record)
                self.assertNotIn('reference',record);self.assertNotIn('answers',record)
                self.assertNotIn('FORBIDDEN_GOLD_SENTINEL',json.dumps(payload))
                self.assertNotIn('Hidden generation instructions',json.dumps(payload))
                self.assertEqual('candidate_reasoning' in record,item['task']=='reasoning')
    def test_allowlist_strips_extra_labels(self):
        i=copy.deepcopy(self.snapshot['items'][0]);i['task_input']['answers']='FORBIDDEN'
        self.assertNotIn('FORBIDDEN',json.dumps(rubric.judge_messages(i)))
    def test_prepare_missing_coverage_and_two_distinct_candidates(self):
        self.assertEqual(len(self.snapshot['missing_candidates']),3920)
        self.assertEqual(len({r['evaluation_id'] for r in self.snapshot['items']}),80)
        self.assertEqual(core.load_snapshot(self.path),self.snapshot)
    def test_modified_snapshot_fails(self):
        s=copy.deepcopy(self.snapshot);s['items'][0]['candidate_response']='Changed';atomic_json(self.path,s)
        with self.assertRaises(ValueError):core.load_snapshot(self.path)
    def test_leaking_input_rejected_even_if_rehashed(self):
        s=copy.deepcopy(self.snapshot);i=s['items'][0];i['task_input']['reference']='secret'
        i['item_hash']=digest({k:v for k,v in i.items() if k!='item_hash'})
        s['snapshot_hash']=digest({k:v for k,v in s.items() if k!='snapshot_hash'});atomic_json(self.path,s)
        with self.assertRaises(ValueError):core.load_snapshot(self.path)
    def test_resume_no_calls_and_saved_bytes_preserved(self):
        self.assertEqual(self.run_judge(),0);self.assertEqual(len(FakeClient.calls),2)
        p=core.results_path(self.args.output_dir,'qwen');before=p.read_bytes()
        self.assertEqual(self.run_judge(),0);self.assertEqual(len(FakeClient.calls),2);self.assertEqual(p.read_bytes(),before)
    def test_failure_not_saved_and_retry(self):
        FakeClient.invalid=True;self.assertEqual(self.run_judge(),1)
        self.assertFalse(core.results_path(self.args.output_dir,'qwen').exists())
        FakeClient.invalid=False;self.assertEqual(self.run_judge(),0)
    def test_check_only_no_http_or_output(self):
        self.args.check_data=True;self.assertEqual(self.run_judge(),0)
        self.assertEqual(FakeClient.calls,[]);self.assertFalse(self.args.output_dir.exists())
    def test_limit_and_invalid_limit(self):
        self.args.limit=1;self.assertEqual(self.run_judge(),1);self.assertEqual(len(FakeClient.calls),1)
        self.args.limit=-1
        with self.assertRaises(ValueError):self.run_judge()
    def test_changed_config_rejected_after_success(self):
        self.run_judge();c=core.load_config(ROOT/'config.json');c['models']['qwen']['parameters']['seed']=99
        config=self.out/'changed.json';atomic_json(config,c);self.args.config=config
        with self.assertRaises(ValueError):self.run_judge()
    def test_reference_review_is_offline_separate_and_no_overwrite(self):
        directory=self.out/'review'
        with patch.object(review,'load_source',return_value=self.source):review.export(self.snapshot,directory,'fixture')
        self.assertIn('FORBIDDEN_GOLD_SENTINEL',(directory/'references_for_review.json').read_text())
        self.assertNotIn('FORBIDDEN_GOLD_SENTINEL',(directory/'reading_pack.json').read_text())
        self.assertNotIn('source_model',(directory/'reading_pack.json').read_text())
        gold,protocol=analyze.human_scores(directory/'adjudicated.csv',self.snapshot)
        self.assertEqual(gold,{})
        with self.assertRaises(ValueError):review.export(self.snapshot,directory)
    def test_no_rank_without_human_ratings(self):
        results={j:[dict(evaluation_id=i['evaluation_id'],overall_score='4') for i in self.snapshot['items']] for j in JUDGES}
        agreement,quality,difference,candidates=analyze.compute(self.snapshot,results,repetitions=100)
        self.assertEqual(quality,[]);self.assertEqual(difference,[])
        self.assertEqual(agreement[0]['mean_absolute_score_difference'],0)
        self.assertTrue(all(r['mean_score']==4 for r in candidates))
    def test_all_three_judges_and_candidate_matching(self):
        items=[i for i in self.snapshot['items'] if i['task']=='qa' and i['language_code']=='bn']
        results={j:[dict(evaluation_id=i['evaluation_id'],overall_score='4') for i in items] for j in JUDGES}
        results['sarvam']=results['sarvam'][:1]
        agreements,_,_,candidates=analyze.compute(dict(items=items),results,repetitions=100)
        self.assertTrue(all(r['n']==1 for r in agreements))
        self.assertTrue(all(r['n']==0 for r in candidates))
    def test_human_ranking_and_paired_intervals(self):
        items=self.snapshot['items'];human={i['evaluation_id']:4 for i in items}
        results={j:[dict(evaluation_id=i['evaluation_id'],overall_score=str(score)) for i in items] for j,score in [('sarvam',4),('qwen',3),('ministral',1)]}
        _,quality,pairs,_=analyze.compute(self.snapshot,results,human,'human_reference_assisted',100,2)
        allrows={r['judge']:r for r in quality if r['language_code']=='all' and r['task']=='all'}
        self.assertEqual(allrows['sarvam']['descriptive_rank_by_mae'],1)
        self.assertEqual(allrows['ministral']['mae'],3)
        self.assertTrue(any(r['ci_high']==-1 for r in pairs))
        cell=[r for r in quality if r['language_code']=='bn' and r['task']=='qa']
        self.assertTrue(all(r['descriptive_rank_by_mae'] is None for r in cell))
    def test_human_invalid_duplicates_and_protocols(self):
        items=self.snapshot['items'][:2];p=self.out/'human.csv'
        rows=[dict(evaluation_id=i['evaluation_id'],item_hash=i['item_hash'],human_score='4',status='adjudicated',reviewer_ids='A|B',protocol='human_reference_free',notes='Agreed after review.') for i in items]
        transport.atomic_rows(p,review.HUMAN_FIELDS,rows)
        self.assertEqual(len(analyze.human_scores(p,self.snapshot)[0]),2)
        rows[1]['protocol']='human_reference_assisted';transport.atomic_rows(p,review.HUMAN_FIELDS,rows)
        with self.assertRaises(ValueError):analyze.human_scores(p,self.snapshot)
        transport.atomic_rows(p,review.HUMAN_FIELDS,[rows[0],rows[0]])
        with self.assertRaises(ValueError):analyze.human_scores(p,self.snapshot)

class ValidationMetricsTests(unittest.TestCase):
    def test_score_schema_strict(self):
        for value in (True,4.0,'4',-1,6):
            with self.assertRaises(ValueError):transport.parse_judgment(json.dumps(dict(overall_score=value,justification='x')))
        with self.assertRaises(ValueError):transport.parse_judgment('{"overall_score":1,"overall_score":2,"justification":"x"}')
    def test_truncation_is_rejected(self):
        with self.assertRaises(ValueError):transport.parse_completion({'choices':[{'finish_reason':'length','message':{'content':'{"overall_score":4}'}}]})
    def test_metrics_known_cases_and_undefined_kappa(self):
        m=metrics.score_metrics([0,1,2,3,4,5],[0,1,2,3,4,5]);self.assertEqual(m['mae'],0);self.assertEqual(m['quadratic_kappa'],1)
        m=metrics.score_metrics([0,5],[5,0]);self.assertEqual(m['mae'],5);self.assertEqual(m['quadratic_kappa'],-1)
        self.assertIsNone(metrics.score_metrics([3,3],[3,3])['quadratic_kappa'])
        self.assertIsNone(metrics.score_metrics([],[])['mae'])
    def test_cluster_bootstrap_keeps_both_candidates(self):
        items=[dict(language_code='bn',task='qa',sample_id=s) for s in ('x','x','y','y')]
        self.assertEqual(metrics.mean_ci([0,2,0,2],items,100),(1.,1.))
    def test_snapshot_no_references_independent_of_source_gold(self):
        original=fixture();a=prepare.freeze(original)
        for sample in original['manifest']['samples']:sample['reference']={'answer':'different gold'}
        self.assertEqual(a,prepare.freeze(original))

if __name__=='__main__':unittest.main()
