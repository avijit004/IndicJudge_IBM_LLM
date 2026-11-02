"""Full selection and historical checkpoint scope tests; never calls a server."""
import argparse
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import common,data,generate,judge,prompts
from test_workflow import FakeClient

class FullSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest=data.load_manifest()
        cls.config=common.load_config()

    def test_all_fifty_source_records(self):
        self.assertEqual(len(self.manifest['samples']),2000)
        self.assertEqual(self.manifest,data.build_manifest(data.DEFAULT_DATASET,1,50))
        for task in prompts.TASKS:
            for code in prompts.LANGUAGES:
                self.assertEqual([s['source_row'] for s in data.select_samples(self.manifest,[code],[task])],list(range(1,51)))

    def test_imported_hashes_are_scoped_to_original_samples(self):
        model=self.config['models']['gemma26b']
        original=data.load_manifest(ROOT/'data/checkpoint_origins/first_five.json')
        scopes=common.fingerprint_scopes(self.manifest,model,'generation')
        fp=common.fingerprint(original,model,'generation')
        self.assertEqual(len(scopes[fp]),200)
        self.assertNotIn(next(s['sample_id'] for s in self.manifest['samples'] if s['source_row']==6),scopes[fp])

    def test_old_fingerprint_cannot_validate_new_row(self):
        model=self.config['models']['gemma26b']
        sample=next(s for s in self.manifest['samples'] if s['source_row']==6)
        origin=data.load_manifest(ROOT/'data/checkpoint_origins/first_five.json')
        row=dict(common.metadata(sample,'gemma26b'),run_fingerprint=common.fingerprint(origin,model,'generation'),api_model=model['alias'],
                 request_json=data.canonical(common.payload_for(prompts.generation_messages(sample),model)),response='Visible answer',reasoning_content='',finish_reason='stop',usage_json='{}',created_at=common.now())
        row['response_hash']=common.response_hash(row)
        with tempfile.TemporaryDirectory() as temp:
            common.atomic_rows(common.generation_path(temp,'gemma26b'),common.GEN_FIELDS,[row])
            with self.assertRaisesRegex(ValueError,'settings/prompts changed'):
                common.candidate_rows(temp,'gemma26b',self.manifest,self.config)

    def test_imported_results_and_new_results_resume_together(self):
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()):
            target=Path(temp)
            # A single original candidate and judgment, retaining their historical hashes.
            candidates=common.candidate_rows(ROOT/'outputs','gemma26b',self.manifest,self.config)
            original=candidates[0]
            lookup={(r['sample_id'],'gemma26b'):r for r in candidates}
            # Build full lookup for validation of all original judge rows.
            other=common.candidate_rows(ROOT/'outputs','gemma31b',self.manifest,self.config)
            lookup.update({(r['sample_id'],'gemma31b'):r for r in other})
            existing=next(r for r in common.judged_rows(ROOT/'outputs','qwen',self.manifest,self.config,lookup) if (r['sample_id'],r['source_model'])==(original['sample_id'],'gemma26b'))
            common.atomic_rows(common.generation_path(target,'gemma26b'),common.GEN_FIELDS,[original])
            common.atomic_rows(common.judgment_path(target,'qwen'),common.JUDGE_FIELDS,[existing])
            samples=[next(s for s in self.manifest['samples'] if s['sample_id']==original['sample_id']),next(s for s in self.manifest['samples'] if s['source_row']==6)]
            args=argparse.Namespace(config=ROOT/'config.json',manifest=ROOT/'data/selection.json',output_dir=target,languages=None,tasks=None,limit=None,check_data=False,source_models='gemma26b',retry_length=False)
            FakeClient.calls=[];FakeClient.invalid_next=False
            with patch.object(generate,'select_samples',return_value=samples),patch.object(judge,'select_samples',return_value=samples):
                self.assertEqual(generate.run(args,'gemma26b',FakeClient),0)
                self.assertEqual(judge.run(args,'qwen',FakeClient),0)
                self.assertEqual(len(FakeClient.calls),2)
                self.assertEqual(generate.run(args,'gemma26b',FakeClient),0)
                self.assertEqual(judge.run(args,'qwen',FakeClient),0)
                self.assertEqual(len(FakeClient.calls),2)
            saved=common.candidate_rows(target,'gemma26b',self.manifest,self.config)
            self.assertEqual(saved[0],original)
            self.assertEqual(len(saved),2)

if __name__=='__main__':unittest.main()
