import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import common,data,generate
from test_workflow import FakeClient
ERROR='HTTP 500 from http://localhost/v1/chat/completions: The model produced output that does not match the expected peg-gemma4 format'

class FormatRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.out=Path(self.temp.name)
        self.manifest=data.load_manifest();self.config=common.load_config()
        self.samples=self.manifest['samples'][:3]
        self.args=argparse.Namespace(config=ROOT/'config.json',manifest=ROOT/'data/selection.json',output_dir=self.out,languages=None,tasks=None,limit=None,check_data=False,source_models='both',retry_length=True,retry_format=True)
        FakeClient.calls=[];FakeClient.invalid_next=False
    def tearDown(self):self.temp.cleanup()
    def run_with(self,samples):
        with patch.object(generate,'select_samples',return_value=samples),contextlib.redirect_stdout(io.StringIO()):return generate.run(self.args,'gemma26b',FakeClient)
    def test_targeted_profile_preservation_and_resume(self):
        self.assertEqual(self.run_with(self.samples[:1]),0)
        path=common.generation_path(self.out,'gemma26b')
        before=common.candidate_rows(self.out,'gemma26b',self.manifest,self.config)[0]
        for sample in self.samples[:2]:common.record_error(path,sample['sample_id'],'gemma26b',ERROR)
        common.record_error(path,self.samples[1]['sample_id'],'gemma26b',"finish_reason='length'")
        self.assertEqual(self.run_with(self.samples),0)
        self.assertEqual([p['seed'] for p in FakeClient.calls],[42,44,42])
        self.assertEqual(FakeClient.calls[1]['max_tokens'],16384)
        rows=common.candidate_rows(self.out,'gemma26b',self.manifest,self.config)
        self.assertEqual(rows[0],before)
        self.assertEqual(self.run_with(self.samples),0)
        self.assertEqual(len(FakeClient.calls),3)
    def test_format_failure_is_not_saved(self):
        path=common.generation_path(self.out,'gemma26b');path.parent.mkdir(parents=True)
        common.record_error(path,self.samples[0]['sample_id'],'gemma26b',ERROR)
        FakeClient.invalid_next=True
        self.assertEqual(self.run_with(self.samples[:1]),1)
        self.assertFalse(path.exists())
        self.assertEqual(self.run_with(self.samples[:1]),0)
    def test_selected_seed_is_recorded_and_resumable(self):
        path=common.generation_path(self.out,'gemma26b');path.parent.mkdir(parents=True)
        common.record_error(path,self.samples[0]['sample_id'],'gemma26b',ERROR)
        self.args.format_retry_seed=45
        self.assertEqual(self.run_with(self.samples[:2]),0)
        self.assertEqual([p['seed'] for p in FakeClient.calls],[45,42])
        before=path.read_bytes()
        rows=common.candidate_rows(self.out,'gemma26b',self.manifest,self.config)
        self.assertEqual(json.loads(rows[0]['request_json'])['seed'],45)
        self.args.retry_format=False
        self.assertEqual(self.run_with(self.samples[:2]),0)
        self.assertEqual(path.read_bytes(),before)
        self.assertEqual(len(FakeClient.calls),2)
    def test_legacy_profile_and_seed_validation(self):
        model=self.config['models']['gemma26b']
        expected=common.length_recovery_model(model)
        expected['recovery_profile']='gemma-format-v1'
        expected['parameters']['seed']=44
        self.assertEqual(common.format_recovery_model(model),expected)
        with self.assertRaises(ValueError):common.format_recovery_model(model,999)
        self.assertEqual(common.format_recovery_model(model,46)['parameters']['seed'],46)
    def test_only_matching_errors_and_model(self):
        path=common.generation_path(self.out,'gemma26b');path.parent.mkdir(parents=True)
        for model,error,sample in [('gemma26b',ERROR,'yes'),('gemma31b',ERROR,'other-model'),('gemma26b','HTTP 500 unrelated','other-error')]:common.record_error(path,sample,model,error)
        self.assertEqual(common.format_failure_ids(path,'gemma26b'),{'yes'})
        with Path(str(path)+'.errors.jsonl').open('a') as f:f.write('broken json\n')
        with self.assertRaises(ValueError):common.format_failure_ids(path,'gemma26b')

if __name__=='__main__':unittest.main()
