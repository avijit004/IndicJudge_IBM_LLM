import contextlib
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parent


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def response(content=None, finish='stop'):
    if content is None:
        content = json.dumps(dict(answer_correct=True, reasoning_quality=4,
                                  follows_prompt_intent=True, justification='Checked all three dimensions.'))
    return Mock(ok=True, json=Mock(return_value={'choices': [{'finish_reason': finish, 'message': {'content': content}}]}))


class MinistralTests(unittest.TestCase):
    def test_scope_ground_truth_and_rubric_all_languages(self):
        total = 0
        for p in sorted(ROOT.glob('ministral_8b_judge_*.py')):
            m = load(p)
            with self.subTest(language=m.LANGUAGE), contextlib.redirect_stdout(io.StringIO()):
                cs = m.load_candidate_responses()
                gt = m.load_ground_truth()
                selected = m.select_candidates(cs, m.DEFAULT_QUESTION_IDS, 'both')
                self.assertEqual(len(selected), 50)
                m.validate_candidates(selected, gt)
                total += len(selected)
                qwen = load(Path(m.BASE_DIR) / 'qwen3_14b_judge' / f'qwen3_14b_judge_{m.LANGUAGE}.py')
                self.assertEqual(m.JUDGE_PROMPT, qwen.JUDGE_PROMPT)
                self.assertEqual(m.FIELDNAMES, qwen.FIELDNAMES)
                self.assertEqual(gt, qwen.load_ground_truth())
                self.assertEqual(cs, qwen.load_candidate_responses())
                for mode in ('26b', '31b'):
                    subset = m.select_candidates(list(reversed(cs)), ('5', '3', '1', '4', '2'), mode)
                    self.assertEqual(len(subset), 25)
                    self.assertEqual({c['source_model'] for c in subset}, {'Gemma-4-' + mode.upper()})
                    for q in ('1','2','3','4','5'):
                        self.assertEqual({c['prompt_id'] for c in subset if c['question_id'] == q}, set(m.PROMPT_INTENTS))
                with patch.object(m.requests, 'get') as get, patch.object(m.requests, 'post') as post:
                    self.assertEqual(m.main(check_data=True), 0)
                    get.assert_not_called(); post.assert_not_called()
        self.assertEqual(total, 500)

    def test_all_selected_rows_save_and_resume(self):
        for p in sorted(ROOT.glob('ministral_8b_judge_*.py')):
            m = load(p)
            with self.subTest(language=m.LANGUAGE), tempfile.TemporaryDirectory() as directory, \
                 patch.object(m, 'OUTPUT_FILE', str(Path(directory)/'results.csv')), \
                 patch.object(m, 'verify_server_context'), \
                 patch.object(m.requests, 'post', return_value=response()) as post, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(m.main(), 0)
                self.assertEqual(post.call_count, 50)
                payload = post.call_args.kwargs['json']
                self.assertEqual(payload['temperature'], .7)
                self.assertEqual(payload['model'], 'Ministral-3-8B-Reasoning')
                self.assertNotIn('chat_template_kwargs', payload)
                self.assertEqual(payload['messages'][0]['role'], 'system')
                self.assertIn('[THINK]', payload['messages'][0]['content'])
                with open(m.OUTPUT_FILE, encoding='utf-8-sig', newline='') as f:
                    rows = list(csv.DictReader(f))
                self.assertEqual(len(rows), 50)
                self.assertTrue(all(m.checkpoint_row_is_complete(r) for r in rows))
                post.reset_mock()
                self.assertEqual(m.main(), 0)
                post.assert_not_called()
                with open(m.OUTPUT_FILE, 'a') as f: f.write('"unfinished')
                original = Path(m.OUTPUT_FILE).read_bytes()
                self.assertEqual(len(m.load_completed_keys()), 50)
                backups = list(Path(directory).glob('*.bak'))
                self.assertEqual(backups[0].read_bytes(), original)

    def test_selection_errors_before_requests(self):
        m = load(ROOT/'ministral_8b_judge_tamil.py')
        with contextlib.redirect_stdout(io.StringIO()): cs = m.load_candidate_responses()
        for ids in (('1',)*5, ('1','2'), ('1','2','3','4','999')):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                m.select_candidates(cs, ids, 'both')
        with self.assertRaisesRegex(ValueError, 'Missing selected candidate'):
            m.select_candidates(cs[1:], m.DEFAULT_QUESTION_IDS, 'both')
        with patch.object(m, 'load_ground_truth', return_value={}), \
             patch.object(m.requests, 'post') as post, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'No reference'): m.main()
            post.assert_not_called()

    def test_response_parsing_all_languages(self):
        good = dict(answer_correct=True, reasoning_quality=4, follows_prompt_intent=False,
                    justification='The candidate quotes [THINK] and [/THINK] literally.')
        args = ('question {answer}', 'తెలుగు தமிழ் {response}'*2000, 'reasoning {question}', '18', 'P1', 'Solve {generation_prompt}')
        for p in sorted(ROOT.glob('ministral_8b_judge_*.py')):
            m = load(p)
            for prefix in ('', '[THINK]analysis[/THINK]', '<think>analysis</think>', 'analysis[/THINK]'):
                with self.subTest(language=m.LANGUAGE, prefix=prefix), patch.object(m.requests, 'post', return_value=response(prefix+json.dumps(good))) as post:
                    result = m.judge_response(*args)
                    self.assertEqual(result['justification'], good['justification'])
                    prompt = post.call_args.kwargs['json']['messages'][1]['content']
                    for text in (args[0],args[1],args[2],args[5]): self.assertIn(text, prompt)
            for bad in (dict(good, reasoning_quality=True), dict(good, reasoning_quality=3.5),
                        dict(good, reasoning_quality=6), dict(good, justification=[]), dict(good, justification=''),
                        dict(good, justification='x'*1201), dict(good, answer_correct=1), [], dict(good, extra=1)):
                with patch.object(m.requests, 'post', return_value=response(json.dumps(bad))), self.assertRaises(ValueError):
                    m.judge_response(*args)
            for text in ('[THINK]unfinished', 'garbage '+json.dumps(good), '{invalid'):
                with patch.object(m.requests, 'post', return_value=response(text)), self.assertRaises(ValueError):
                    m.judge_response(*args)
            with patch.object(m.requests, 'post', return_value=response(finish='length')), self.assertRaisesRegex(ValueError, 'will not be saved'):
                m.judge_response(*args)

    def test_failed_rows_retry(self):
        m = load(ROOT/'ministral_8b_judge_telugu.py')
        with tempfile.TemporaryDirectory() as directory, patch.object(m,'OUTPUT_FILE',str(Path(directory)/'out.csv')), \
             patch.object(m,'verify_server_context'), contextlib.redirect_stdout(io.StringIO()):
            with patch.object(m.requests,'post',side_effect=[response(finish='length')]+[response() for _ in range(24)]) as post:
                self.assertEqual(m.main(source_model='26b'),1)
                self.assertEqual(post.call_count,25)
            with patch.object(m.requests,'post',return_value=response()) as post:
                self.assertEqual(m.main(source_model='26b'),0)
                self.assertEqual(post.call_count,1)

    def test_preflight(self):
        m = load(ROOT/'ministral_8b_judge_tamil.py')
        with patch.object(m,'API_URL',''), patch.object(m.requests,'get') as get:
            with self.assertRaisesRegex(ValueError,'Set MINISTRAL_API_URL'):m.verify_server_context()
            get.assert_not_called()
        for model,context,ok in (('Qwen3-14B',32768,False),(m.JUDGE_MODEL,8192,False),(m.JUDGE_MODEL,32768,True)):
            models=Mock(json=Mock(return_value={'data':[{'id':model}]}))
            props=Mock(ok=True,json=Mock(return_value={'default_generation_settings':{'n_ctx':context},'total_slots':1}))
            with patch.object(m,'API_URL','http://example.test:9999/v1/chat/completions'), \
                 patch.object(m.requests,'get',side_effect=[models,props]), contextlib.redirect_stdout(io.StringIO()):
                if ok:m.verify_server_context()
                else:
                    with self.assertRaises(RuntimeError):m.verify_server_context()

    def test_runner_forwards_scope_and_continues(self):
        m=load(ROOT/'run_all.py')
        with patch('sys.argv',['run_all.py','--languages','tamil','telugu','--source-model','31b','--question-ids','6','7','8','9','10','--check-data']), \
             patch.object(m.subprocess,'run',side_effect=[Mock(returncode=1),Mock(returncode=0)]) as run, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(m.main(),1)
            self.assertEqual(run.call_count,2)
            self.assertEqual(run.call_args.args[0][-9:],['--question-ids','6','7','8','9','10','--source-model','31b','--check-data'])


if __name__=='__main__':unittest.main()
