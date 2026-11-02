"""Offline tests: no live API calls, and all generated results stay in temporary directories."""
import argparse
import contextlib
import copy
import csv
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import common
import data
import generate
import judge
import prompts


def completion(content='A candidate response.', reasoning='', finish='stop'):
    return {'choices': [{'finish_reason': finish, 'message': {'content': content, 'reasoning_content': reasoning}}],
            'usage': {'completion_tokens': 12}}


class FakeClient:
    calls = []
    invalid_next = False

    def __init__(self, model, config):
        self.model = model

    def preflight(self):
        pass

    def complete(self, payload):
        FakeClient.calls.append(payload)
        if FakeClient.invalid_next:
            FakeClient.invalid_next = False
            raise ValueError('Mock incomplete generation')
        text = 'A visible answer with a brief explanation.'
        if self.model['role'] == 'judge':
            text = json.dumps({'overall_score': 4, 'justification': 'The main relationship is supported. A minor explanation is missing.'})
        return common.parse_completion(completion(text, 'Internal reasoning.'))


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = data.build_manifest(os.environ.get('MULTITASK_TEST_DATASET', str(data.DEFAULT_DATASET)))
        cls.config = common.load_config()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.manifest_path = self.directory / 'selection.json'
        self.manifest_path.write_text(json.dumps(self.manifest), encoding='utf-8')
        self.args = argparse.Namespace(config=ROOT/'config.json', manifest=self.manifest_path,
            output_dir=self.directory/'results', languages=['bn'], tasks=['reasoning'], check_data=False,
            limit=None, source_models='both')
        FakeClient.calls = []; FakeClient.invalid_next = False

    def tearDown(self):
        self.temp.cleanup()

    def run_quiet(self, function, *args):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            code = function(*args)
        return code, output.getvalue()

    def test_selection_has_exactly_200_unique_inputs(self):
        self.assertEqual(len(data.validate_manifest(self.manifest)['samples']), 200)
        self.assertEqual(len(self.manifest['source_file_hashes']), 40)
        for task in prompts.TASKS:
            for code in prompts.LANGUAGES:
                selected = data.select_samples(self.manifest, [code], [task])
                self.assertEqual([s['source_row'] for s in selected], [1,2,3,4,5])
        self.assertEqual(data.build_manifest(os.environ.get('MULTITASK_TEST_DATASET', str(data.DEFAULT_DATASET))), self.manifest)

    def test_missing_source_fails_before_api(self):
        with self.assertRaises(FileNotFoundError):
            data.build_manifest(self.directory)

    def test_changed_manifest_rejected(self):
        changed = copy.deepcopy(self.manifest)
        changed['samples'][0]['input']['question'] = 'Changed'
        with self.assertRaises(ValueError):
            data.validate_manifest(changed)

    def test_invalid_qa_reference_and_copa_label(self):
        row = {'id':'x', 'language':'Bengali', 'context':'c', 'question':'q', 'answers':'[false]'}
        with self.assertRaises(ValueError): data.normalize(row, 'qa', 'bn', 'qa/bn.csv', 1)
        row = {'id':'x', 'language':'Bengali', 'premise':'p', 'question':'cause', 'choice1':'a', 'choice2':'b', 'label':'2'}
        with self.assertRaises(ValueError): data.normalize(row, 'reasoning', 'bn', 'reasoning/bn.csv', 1)

    def test_all_prompts_and_model_payloads(self):
        count_generation = count_judging = 0
        for sample in self.manifest['samples']:
            for model_key in common.CANDIDATES:
                payload = common.payload_for(prompts.generation_messages(sample), self.config['models'][model_key])
                body = json.loads(payload['messages'][-1]['content'].split('task data.\n',1)[1])
                self.assertEqual(body, sample['input'])
                self.assertNotIn('reference', body)
                count_generation += 1
            for model_key in common.JUDGES:
                for source in common.CANDIDATES:
                    messages = prompts.judge_messages(sample, {'response':'Answer {question} {{}}',
                                                              'reasoning_content':'Reasoning {response}'})
                    payload = common.payload_for(messages, self.config['models'][model_key], True)
                    record = json.loads(payload['messages'][-1]['content'].split('Evaluation record:\n',1)[1])
                    self.assertNotIn('source_model', record)
                    self.assertNotIn('generation_prompt', record)
                    self.assertEqual(record['candidate_response'], 'Answer {question} {{}}')
                    if sample['task'] == 'reasoning':
                        self.assertNotIn('reference', record)
                        self.assertNotIn('correct_option', record['task_input'])
                        self.assertNotIn('label', record['task_input'])
                        self.assertEqual(record['candidate_reasoning'], 'Reasoning {response}')
                        self.assertIn('cause' if sample['input']['relation']=='cause' else 'effect', record['task_input']['question'])
                    else:
                        self.assertNotIn('candidate_reasoning', record)
                    if sample['task'] == 'translation':
                        self.assertEqual(record['requested_output_language'], 'English')
                    count_judging += 1
        self.assertEqual((count_generation,count_judging),(400,1200))

    def test_generation_does_not_leak_gold_text(self):
        for sample in self.manifest['samples']:
            clone=copy.deepcopy(sample)
            clone['reference']={'answers':['GOLD_CANARY'], 'summary':'GOLD_CANARY', 'target_text':'GOLD_CANARY', 'correct_option':'GOLD_CANARY'}
            self.assertNotIn('GOLD_CANARY', prompts.generation_messages(clone)[0]['content'])
            if sample['task']=='reasoning':
                self.assertNotIn('GOLD_CANARY', prompts.judge_messages(clone, {'response':'x','reasoning_content':'y'})[0]['content'])

    def test_strict_judgments(self):
        for score in range(6):
            self.assertEqual(common.parse_judgment(json.dumps({'overall_score':score,'justification':'Evidence.'}))['overall_score'],score)
        for score in (3.9, 3.0, True, False, '4', -1, 6, None):
            with self.subTest(score=score), self.assertRaises(ValueError):
                common.validate_judgment({'overall_score':score,'justification':'Evidence.'})
        for justification in ('', ' ', [], {}, None, 'x'*1201):
            with self.subTest(justification=repr(justification)[:30]), self.assertRaises(ValueError):
                common.validate_judgment({'overall_score':4,'justification':justification})
        for text in ('[]','null','{bad}', '```json\n{}\n```', '{"overall_score":4,"overall_score":5,"justification":"x"}',
                     '{"overall_score":4,"justification":"x","extra":1}'):
            with self.subTest(text=text),self.assertRaises(ValueError): common.parse_judgment(text)

    def test_completion_length_and_empty_reasoning_are_not_saved(self):
        for raw in (completion('x', finish='length'), completion('', reasoning='answer only here'),
                    completion('<think>not closed'), {'choices':[]}):
            with self.assertRaises(ValueError): common.parse_completion(raw)
        parsed=common.parse_completion(completion('[THINK]private[/THINK]final'))
        self.assertEqual(parsed['reasoning_content'],'private')
        self.assertEqual(parsed['response'],'final')
        literal='{"overall_score":4,"justification":"Literal <think> tag."}'
        self.assertEqual(common.parse_completion(completion(literal))['response'],literal)

    def test_full_mocked_pipeline_and_resume(self):
        for model in common.CANDIDATES:
            self.assertEqual(self.run_quiet(generate.run,self.args,model,FakeClient)[0],0)
        for model in common.JUDGES:
            self.assertEqual(self.run_quiet(judge.run,self.args,model,FakeClient)[0],0)
            rows=common.read_rows(common.judgment_path(self.args.output_dir,model),common.JUDGE_FIELDS)
            self.assertEqual(len(rows),10)
        calls=len(FakeClient.calls)
        for model in common.CANDIDATES:
            self.assertEqual(self.run_quiet(generate.run,self.args,model,FakeClient)[0],0)
        for model in common.JUDGES:
            self.assertEqual(self.run_quiet(judge.run,self.args,model,FakeClient)[0],0)
        self.assertEqual(len(FakeClient.calls),calls)

    def test_failed_generation_retries_and_exit_status(self):
        FakeClient.invalid_next=True
        code,text=self.run_quiet(generate.run,self.args,'gemma26b',FakeClient)
        self.assertEqual(code,1);self.assertIn('Failed: 1 | Remaining: 1',text)
        self.assertEqual(len(common.read_rows(common.generation_path(self.args.output_dir,'gemma26b'),common.GEN_FIELDS)),4)
        code,text=self.run_quiet(generate.run,self.args,'gemma26b',FakeClient)
        self.assertEqual(code,0);self.assertIn('Previously completed: 4 | Successful: 1',text)

    def test_failed_judgment_retry_and_missing_candidates(self):
        self.run_quiet(generate.run,self.args,'gemma26b',FakeClient)
        self.args.source_models='gemma26b'
        FakeClient.invalid_next=True
        code,text=self.run_quiet(judge.run,self.args,'sarvam',FakeClient)
        self.assertEqual(code,1);self.assertIn('Failed: 1',text)
        code,text=self.run_quiet(judge.run,self.args,'sarvam',FakeClient)
        self.assertEqual(code,0);self.assertIn('Previously completed: 4 | Successful: 1',text)
        self.args.source_models='both'
        code,text=self.run_quiet(judge.run,self.args,'sarvam',FakeClient)
        self.assertEqual(code,1);self.assertIn('Missing candidates: 5',text)

    def test_malformed_judgment_never_saved(self):
        self.run_quiet(generate.run,self.args,'gemma26b',FakeClient)
        self.args.source_models='gemma26b'
        with patch.object(FakeClient,'complete',return_value=common.parse_completion(completion('{"overall_score":3.9,"justification":"x"}'))):
            code,_=self.run_quiet(judge.run,self.args,'sarvam',FakeClient)
        self.assertEqual(code,1)
        self.assertFalse(common.judgment_path(self.args.output_dir,'sarvam').exists())

    def test_check_data_never_calls_api(self):
        self.args.check_data=True
        factory=Mock(side_effect=AssertionError('No API access allowed'))
        self.assertEqual(self.run_quiet(generate.run,self.args,'gemma26b',factory)[0],0)
        self.assertEqual(self.run_quiet(judge.run,self.args,'sarvam',factory)[0],0)
        factory.assert_not_called()

    def test_checkpoint_changes_stop_without_overwriting(self):
        self.run_quiet(generate.run,self.args,'gemma26b',FakeClient)
        path=common.generation_path(self.args.output_dir,'gemma26b')
        rows=common.read_rows(path,common.GEN_FIELDS);rows[0]['response']='Edited answer'
        common.atomic_rows(path,common.GEN_FIELDS,rows);before=path.read_bytes()
        with self.assertRaises(ValueError): common.candidate_rows(self.args.output_dir,'gemma26b',self.manifest,self.config)
        self.assertEqual(path.read_bytes(),before)

    def test_atomic_write_and_lock(self):
        path=self.directory/'file.csv';common.atomic_rows(path,['x'],[{'x':'old'}]);before=path.read_bytes()
        with patch('common.os.replace',side_effect=OSError('disk failure')):
            with self.assertRaises(OSError):common.atomic_rows(path,['x'],[{'x':'new'}])
        self.assertEqual(path.read_bytes(),before)
        with common.output_lock(path):
            with self.assertRaises(ValueError):
                with common.output_lock(path): pass

    def test_limits_and_working_directory(self):
        self.args.limit=1
        original=Path.cwd()
        try:
            os.chdir(self.directory)
            code,text=self.run_quiet(generate.run,self.args,'gemma26b',FakeClient)
        finally:
            os.chdir(original)
        self.assertEqual(code,1);self.assertIn('Remaining: 4',text)
        self.assertTrue(common.generation_path(self.args.output_dir,'gemma26b').exists())
        self.assertTrue(data.ROOT.is_absolute())

    def test_length_recovery_only_retries_failed_record_and_resumes(self):
        original_complete = FakeClient.complete
        first = [True]
        def once_length(client, payload):
            if first[0]:
                first[0] = False
                raise ValueError("Incomplete generation: finish_reason='length'; usage={}")
            return original_complete(client, payload)
        with patch.object(FakeClient, 'complete', once_length):
            self.assertEqual(self.run_quiet(generate.run, self.args, 'gemma26b', FakeClient)[0], 1)
        path = common.generation_path(self.args.output_dir, 'gemma26b')
        before = common.read_rows(path, common.GEN_FIELDS)
        self.assertEqual(len(before), 4)
        self.args.retry_length = True
        FakeClient.calls = []
        self.assertEqual(self.run_quiet(generate.run, self.args, 'gemma26b', FakeClient)[0], 0)
        self.assertEqual(len(FakeClient.calls), 1)
        request = FakeClient.calls[0]
        self.assertEqual(request['seed'], 43)
        self.assertEqual(request['max_tokens'], 16384)
        self.assertEqual(request['reasoning_budget_tokens'], 4096)
        after = common.candidate_rows(self.args.output_dir, 'gemma26b', self.manifest, self.config)
        self.assertEqual(after[:4], before)
        self.assertNotEqual(after[0]['run_fingerprint'], after[-1]['run_fingerprint'])
        self.args.retry_length = False
        calls = len(FakeClient.calls)
        self.assertEqual(self.run_quiet(generate.run, self.args, 'gemma26b', FakeClient)[0], 0)
        self.assertEqual(len(FakeClient.calls), calls)
        self.args.source_models = 'gemma26b'
        self.assertEqual(self.run_quiet(judge.run, self.args, 'sarvam', FakeClient)[0], 0)
        # Known profile fingerprints do not permit arbitrary decoding changes.
        payload = json.loads(after[-1]['request_json']); payload['seed'] = 44
        after[-1]['request_json'] = data.canonical(payload)
        after[-1]['response_hash'] = common.response_hash(after[-1])
        common.atomic_rows(path, common.GEN_FIELDS, after)
        with self.assertRaises(ValueError):
            common.candidate_rows(self.args.output_dir, 'gemma26b', self.manifest, self.config)

    def test_retry_length_does_not_change_fresh_samples(self):
        self.args.retry_length = True
        self.assertEqual(self.run_quiet(generate.run, self.args, 'gemma26b', FakeClient)[0], 0)
        for request in FakeClient.calls:
            self.assertEqual(request['seed'], 42)
            self.assertEqual(request['max_tokens'], 8192)
            self.assertNotIn('reasoning_budget_tokens', request)

    def test_length_recovery_failure_stays_retryable(self):
        path = common.generation_path(self.args.output_dir, 'gemma26b')
        path.parent.mkdir(parents=True)
        sample = data.select_samples(self.manifest, ['bn'], ['reasoning'])[0]
        common.record_error(path, sample['sample_id'], 'gemma26b', "finish_reason='length'")
        self.args.retry_length = True
        self.args.limit = 1
        with patch.object(FakeClient, 'complete', side_effect=ValueError("finish_reason='length'")):
            self.assertEqual(self.run_quiet(generate.run, self.args, 'gemma26b', FakeClient)[0], 1)
        self.assertFalse(path.exists())

    def client(self):
        model=self.config['models']['qwen']
        with patch.dict(os.environ,{model['api_url_env']:'http://example.invalid:8080/v1/chat/completions'}):
            return common.Client(model,self.config,Mock())

    def test_preflight_alias_and_context(self):
        client=self.client()
        with patch.object(client,'request',return_value={'data':[{'id':'wrong'}]}):
            with self.assertRaises(ValueError): client.preflight()
        with patch.object(client,'request',side_effect=[{'data':[{'id':'Qwen3-14B'}]}, {'default_generation_settings':{'n_ctx':8192}}]):
            with self.assertRaises(ValueError): client.preflight()
        with patch.object(client,'request',side_effect=[{'data':[{'id':'Qwen3-14B'}]}, {'default_generation_settings':{'n_ctx':32768}}]):
            self.run_quiet(client.preflight)
        self.assertEqual(client.context,32768)

    def test_context_budget_failure_before_completion_call(self):
        client=self.client();client.context=32768
        payload=common.payload_for([{'role':'user','content':'text'}],self.config['models']['qwen'],True)
        with patch.object(client,'request',side_effect=[{'prompt':'rendered'}, {'tokens':[1]*25000}]) as request:
            with self.assertRaises(ValueError):client.complete(payload)
            self.assertEqual(request.call_count,2)
        with patch.object(client,'request',side_effect=[{'prompt':'rendered'}, {'tokens':[1]*100}, completion('{"overall_score":5,"justification":"Valid."}')]):
            self.assertEqual(common.parse_judgment(client.complete(payload)['response'])['overall_score'],5)


if __name__ == '__main__':
    unittest.main()


class PromptFingerprintTests(unittest.TestCase):
    def setUp(self):
        self.manifest = data.load_manifest(ROOT / 'data' / 'selection.json')
        self.model = common.load_config()['models']['gemma26b']
        self.archive = json.loads((ROOT / 'data' / 'legacy_prompt_fingerprint.json').read_text())

    def legacy(self, model=None):
        return data.digest({'selection': self.manifest['manifest_hash'], 'stage': 'generation',
                            'model': model or self.model, 'prompts_code': self.archive['prompts_code']})

    def test_source_formatting_does_not_change_fingerprint(self):
        expected = common.fingerprint(self.manifest, self.model, 'generation')
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'prompts.py'
            source.write_text('# Different formatting only\n')
            with patch.object(prompts, '__file__', str(source)):
                self.assertEqual(expected, common.fingerprint(self.manifest, self.model, 'generation'))
                self.assertIn(self.legacy(), common.compatible_fingerprints(self.manifest, self.model, 'generation'))

    def test_actual_prompt_change_rejects_legacy(self):
        original = prompts.generation_messages
        def changed(sample):
            messages = copy.deepcopy(original(sample))
            messages[0]['content'] += 'Changed instructions.'
            return messages
        expected = common.fingerprint(self.manifest, self.model, 'generation')
        with patch.object(prompts, 'generation_messages', changed):
            self.assertNotEqual(expected, common.fingerprint(self.manifest, self.model, 'generation'))
            self.assertNotIn(self.legacy(), common.compatible_fingerprints(self.manifest, self.model, 'generation'))

    def test_actual_model_change_rejects_legacy(self):
        changed = copy.deepcopy(self.model)
        changed['parameters']['temperature'] = 0.123
        self.assertNotIn(self.legacy(), common.compatible_fingerprints(self.manifest, changed, 'generation'))


class EndpointConfigurationTests(unittest.TestCase):
    def test_defaults_and_environment_override(self):
        config = common.load_config()
        for name, port in [('qwen', 8080), ('ministral', 33937)]:
            model = config['models'][name]
            with patch.dict(os.environ, {}, clear=True):
                client = common.Client(model, config)
                self.assertEqual(client.url, f'http://172.30.1.83:{port}/v1/chat/completions')
            with patch.dict(os.environ, {model['api_url_env']: 'http://localhost:12345/v1/chat/completions'}):
                self.assertEqual(common.Client(model, config).url, 'http://localhost:12345/v1/chat/completions')

    def test_endpoint_changes_preserve_fingerprint(self):
        config = common.load_config()
        manifest = data.load_manifest(ROOT / 'data' / 'selection.json')
        before = common.fingerprint(manifest, config['models']['ministral'], 'judging')
        config['endpoints']['MULTITASK_MINISTRAL_API_URL'] = 'http://localhost:9999/v1/chat/completions'
        self.assertEqual(before, common.fingerprint(manifest, config['models']['ministral'], 'judging'))
