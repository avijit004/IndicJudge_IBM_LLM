import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import workflow as w
from core import load_snapshot

class FakeClient:
    calls=[];fail=False
    def __init__(self,*args):pass
    def preflight(self):pass
    def complete(self,payload):
        self.calls.append(payload)
        if self.fail:raise ValueError('Mock incomplete generation')
        return {'response':'A synthetic summary for tests only.','reasoning_content':'','finish_reason':'stop','usage_json':'{}'}

class ReplacementTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.patches=[patch.object(w,'COMBINED',self.root/'combined'),patch.object(w,'ROOT',self.root),patch.object(w,'PLAN',self.root/'data/replacements.json'),patch.object(w,'SNAPSHOT',self.root/'inputs_4000.json')]
        for p in self.patches:p.start()
        FakeClient.calls=[];FakeClient.fail=False
        with contextlib.redirect_stdout(io.StringIO()):w.prepare()
    def tearDown(self):
        for p in self.patches:p.stop()
        self.tmp.cleanup()
    def gen(self,model):
        with contextlib.redirect_stdout(io.StringIO()):return w.generate(model,client_factory=FakeClient)
    def test_preparation_is_repeatable_and_ids_are_correct(self):
        before=w.PLAN.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()):w.prepare()
        self.assertEqual(before,w.PLAN.read_bytes())
        self.assertEqual({s['source_id'] for s in w.plan()['samples']},{'crosssum-kn-490','crosssum-ta-116'})
        self.assertTrue(all(s['source_row']==51 for s in w.plan()['samples']))
    def test_generation_resume_and_same_baseline_settings(self):
        self.assertEqual(self.gen('gemma26b'),0)
        path=w.legacy.generation_path(w.ROOT/'outputs','gemma26b');before=path.read_bytes()
        self.assertEqual(self.gen('gemma26b'),0)
        self.assertEqual(len(FakeClient.calls),2);self.assertEqual(path.read_bytes(),before)
        self.assertTrue(all(p['seed']==42 and p['max_tokens']==8192 for p in FakeClient.calls))
        for p in FakeClient.calls:
            self.assertEqual(set(json.loads(p['messages'][-1]['content'].split('task data.\n')[1])),{'document'})
    def test_failed_generation_not_saved(self):
        FakeClient.fail=True;self.assertEqual(self.gen('gemma31b'),1)
        self.assertEqual(w.saved(w.plan(),'gemma31b'),[])
        FakeClient.fail=False;self.assertEqual(self.gen('gemma31b'),0)
    def test_merge_refuses_incomplete_replacements(self):
        self.gen('gemma26b')
        with self.assertRaises(ValueError):w.merge()
        self.assertFalse(w.SNAPSHOT.exists())
    def test_merge_4000_is_matched_and_originals_unchanged(self):
        paths=[w.SOURCE/'outputs/candidates'/f'{m}_responses.csv' for m in w.legacy.CANDIDATES]
        hashes=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
        for m in w.legacy.CANDIDATES:self.gen(m)
        with contextlib.redirect_stdout(io.StringIO()):w.merge();w.merge()
        snapshot=load_snapshot(w.SNAPSHOT)
        for model in w.legacy.CANDIDATES:
            path=w.COMBINED/f'{model}_responses.csv'
            rows=w.legacy.read_rows(path,w.legacy.GEN_FIELDS)
            self.assertEqual(len(rows),2000)
            self.assertEqual(len({r['sample_id'] for r in rows}),2000)
            from collections import Counter
            self.assertEqual(set(Counter((r['task'],r['language_code']) for r in rows).values()),{50})
        self.assertEqual(len(snapshot['items']),4000);self.assertEqual(snapshot['missing_candidates'],[])
        sets={m:{i['sample_id'] for i in snapshot['items'] if i['source_model']==m} for m in w.legacy.CANDIDATES}
        self.assertEqual(sets['gemma26b'],sets['gemma31b']);self.assertEqual(len(sets['gemma26b']),2000)
        for lang,(old,new) in w.SWAPS.items():
            self.assertNotIn(f'summarization:{lang}:{old}',sets['gemma26b'])
            self.assertIn(f'summarization:{lang}:{new}',sets['gemma26b'])
        self.assertEqual(hashes,[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths])
    def test_corrupt_replacement_is_rejected(self):
        self.gen('gemma26b');p=w.plan();rows=w.saved(p,'gemma26b');rows[0]['response']='Tampered'
        w.legacy.atomic_rows(w.legacy.generation_path(w.ROOT/'outputs','gemma26b'),w.legacy.GEN_FIELDS,rows)
        with self.assertRaises(ValueError):w.saved(p,'gemma26b')

if __name__=='__main__':unittest.main()
