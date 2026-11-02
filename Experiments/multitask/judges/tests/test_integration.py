"""CLI analysis, transport boundary and dashboard checks, without live inference."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import analyze,core,judge,transport
from utils import ROOT,JUDGES,atomic_json
from test_reference_free import fixture,prepare,FakeClient

class TransportTests(unittest.TestCase):
    def config(self):return core.load_config(ROOT/'config.json')
    def client(self):
        c=self.config();c['endpoints']={'RF_QWEN_API_URL':'http://localhost:8000/v1/chat/completions'}
        return transport.Client(c['models']['qwen'],c,session=Mock())
    def test_long_multilingual_csv_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'long.csv'
            text='বাংলা,தமிழ்\n' * 20000
            transport.atomic_rows(p,['content'],[{'content':text}])
            self.assertEqual(transport.read_rows(p,['content']),[{'content':text}])
    def test_wrong_alias_stops(self):
        c=self.client();c.request=Mock(return_value={'data':[{'id':'wrong-model'}]})
        with self.assertRaises(ValueError):c.preflight()
    def test_small_context_stops(self):
        c=self.client();c.request=Mock(side_effect=[{'data':[{'id':'Qwen3-14B'}]},{'default_generation_settings':{'n_ctx':8192}}])
        with self.assertRaises(ValueError):c.preflight()
    def test_overlong_input_never_calls_completion(self):
        c=self.client();c.context=32768;c.request=Mock(side_effect=[{'prompt':'Rendered'},{'tokens':[0]*25000}])
        with self.assertRaises(ValueError):c.complete({'messages':[],'max_tokens':8192})
        self.assertEqual(c.request.call_count,2)
    def test_corrupt_result_rejected_without_overwrite(self):
        from test_reference_free import WorkflowTests
        t=WorkflowTests('test_resume_no_calls_and_saved_bytes_preserved');t.snapshot=prepare.freeze(fixture());t.setUp()
        try:
            t.run_judge();p=core.results_path(t.args.output_dir,'qwen');s=p.read_text(encoding='utf-8-sig').replace('The source supports','Changed evidence')
            p.write_text(s,encoding='utf-8-sig')
            # Change only parsed justification, leaving raw response unchanged.
            import csv
            with p.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
            rows[0]['overall_score']='0';transport.atomic_rows(p,core.FIELDS,rows)
            before=p.read_bytes()
            with self.assertRaises(ValueError):t.run_judge()
            self.assertEqual(before,p.read_bytes())
        finally:t.tearDown()

class AnalysisIntegration(unittest.TestCase):
    def test_cli_exports_real_columns_and_dashboard(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);s=prepare.freeze(fixture());atomic_json(d/'inputs.json',s)
            import argparse
            args=argparse.Namespace(snapshot=d/'inputs.json',config=ROOT/'config.json',output_dir=d/'judges',
                languages=None,tasks=None,source_models='both',limit=None,check_data=False)
            FakeClient.invalid=False;FakeClient.calls=[]
            with contextlib.redirect_stdout(io.StringIO()):
                for j in JUDGES:self.assertEqual(judge.run(args,j,FakeClient),0)
            self.assertEqual(len(FakeClient.calls),240)
            result=subprocess.run([sys.executable,'-B',str(ROOT/'analyze.py'),'--snapshot',str(d/'inputs.json'),
                '--config',str(ROOT/'config.json'),'--output-dir',str(d/'judges'),'--analysis-dir',str(d/'analysis'),'--bootstrap','100'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr+result.stdout)
            self.assertEqual(json.loads((d/'analysis/coverage.json').read_text())['matched_for_all_judges'],80)
            self.assertEqual(len((d/'analysis/judge_leaderboard.csv').read_text().splitlines()),1)
            # Optional GUI dependency: exercise the populated dashboard when available.
            try:from streamlit.testing.v1 import AppTest
            except ImportError:return
            shutil.copyfile(ROOT/'dashboard.py',d/'dashboard.py')
            at=AppTest.from_file(str(d/'dashboard.py')).run(timeout=20)
            self.assertEqual(len(at.exception),0,str(at.exception))
            self.assertEqual(len(at.dataframe),2)
            at.selectbox[0].select('bn').run();at.selectbox[1].select('qa').run()
            self.assertEqual(len(at.exception),0,str(at.exception))

if __name__=='__main__':unittest.main()
