"""Shared storage and llama.cpp transport. Importing this module never calls a server."""
import argparse
import copy
import csv
import fcntl
import json
import os
import re
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
import requests
from data import ROOT, canonical, digest
from prompts import LANGUAGES, TASKS, JUDGMENT_SCHEMA

CANDIDATES = ('gemma26b', 'gemma31b')
JUDGES = ('sarvam', 'qwen', 'ministral')
META_FIELDS = ['sample_id', 'sample_hash', 'task', 'language_code', 'language', 'source_id', 'source_model']
GEN_FIELDS = META_FIELDS + ['run_fingerprint', 'api_model', 'request_json', 'response', 'reasoning_content',
                           'response_hash', 'finish_reason', 'usage_json', 'created_at']
JUDGE_FIELDS = META_FIELDS + ['judge_model', 'api_model', 'run_fingerprint', 'candidate_hash', 'request_json',
                             'overall_score', 'justification', 'raw_judge_response', 'judge_reasoning',
                             'finish_reason', 'usage_json', 'created_at']


def now():
    return datetime.now(timezone.utc).isoformat()


def load_config(path=None):
    config = json.loads(Path(path or ROOT / 'config.json').read_text(encoding='utf-8'))
    if type(config.get('minimum_context_tokens')) is not int or config['minimum_context_tokens'] < 1:
        raise ValueError('minimum_context_tokens must be a positive integer')
    if type(config.get('timeout_seconds')) is not int or config['timeout_seconds'] < 1:
        raise ValueError('timeout_seconds must be a positive integer')
    if set(config.get('models', {})) != set(CANDIDATES + JUDGES):
        raise ValueError('config.json must define both candidates and all three judges')
    for key, model in config['models'].items():
        if model.get('role') != ('candidate' if key in CANDIDATES else 'judge'):
            raise ValueError(f'Incorrect role for {key}')
        if not model.get('alias') or not model.get('api_url_env'):
            raise ValueError(f'Missing alias or endpoint environment variable for {key}')
        params = model.get('parameters', {})
        if set(params) & {'model', 'messages', 'response_format', 'stream'}:
            raise ValueError('Model parameters cannot override the request or output schema')
        if type(params.get('max_tokens')) is not int or not 0 < params['max_tokens'] < config['minimum_context_tokens']:
            raise ValueError(f'Invalid max_tokens for {key}')
        budget = params.get('reasoning_budget_tokens', 0)
        if type(budget) is not int or not 0 <= budget < params['max_tokens']:
            raise ValueError(f'Invalid reasoning budget for {key}')
    return config


def messages_for_model(messages, model):
    if model.get('system_prompt'):
        return [{'role': 'system', 'content': model['system_prompt']}] + messages
    return messages


def payload_for(messages, model, judging=False):
    payload = dict(model['parameters'], model=model['alias'],
                   messages=messages_for_model(messages, model), stream=False)
    if judging:
        payload['response_format'] = {'type': 'json_object', 'schema': JUDGMENT_SCHEMA}
    return payload


def rendered_prompt_content(manifest, stage):
    """Fingerprint instructions as sent, independent of Python source formatting."""
    import prompts
    if stage == 'generation':
        return [prompts.generation_messages(s) for s in manifest['samples']]
    if stage != 'judging':
        raise ValueError(f'Unknown fingerprint stage: {stage}')
    candidate = {'response': '{candidate_response}', 'reasoning_content': '{candidate_reasoning}'}
    return {'schema': prompts.JUDGMENT_SCHEMA,
            'messages': [prompts.judge_messages(s, candidate) for s in manifest['samples']]}


def fingerprint(manifest, model, stage):
    return digest({'selection': manifest['manifest_hash'], 'stage': stage, 'model': model,
                   'rendered_prompts': rendered_prompt_content(manifest, stage)})


def compatible_fingerprints(manifest, model, stage):
    """Accept known historical hashes only for identical rendered instructions.

    Each saved row must also pass exact request, settings and integrity checks.
    Saved CSVs and response hashes are never rewritten.
    """
    import prompts
    result = {fingerprint(manifest, model, stage)}
    # Compatibility with checkpoints made using the current source-hash format.
    sources = [Path(prompts.__file__).read_text(encoding='utf-8')]
    archive = json.loads((ROOT / 'data' / 'legacy_prompt_fingerprint.json').read_text(encoding='utf-8'))
    if digest(rendered_prompt_content(manifest, stage)) == archive['rendered_hashes'].get(stage):
        sources.append(archive['prompts_code'])
    for source in sources:
        result.add(digest({'selection': manifest['manifest_hash'], 'stage': stage,
                           'model': model, 'prompts_code': source}))
    return result


def validate_judgment(value):
    if not isinstance(value, dict) or set(value) != {'overall_score', 'justification'}:
        raise ValueError('Expected a JSON object with exactly overall_score and justification')
    if type(value['overall_score']) is not int or value['overall_score'] not in range(6):
        raise ValueError('overall_score must be an integer from 0 to 5, not a Boolean, float or string')
    text = value['justification']
    if not isinstance(text, str) or not text.strip() or len(text) > 1200:
        raise ValueError('justification must be a nonempty string of at most 1200 characters')
    # The 2-4 sentence rule is prompted; sentence counts are not heuristically enforced.
    return value


def parse_judgment(text):
    def unique_keys(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError(f'Duplicate JSON key: {key}')
            obj[key] = value
        return obj
    if not isinstance(text, str):
        raise ValueError('Judge final content must be text')
    try:
        return validate_judgment(json.loads(text, object_pairs_hook=unique_keys))
    except json.JSONDecodeError as exc:
        raise ValueError('Judge final content must be a complete JSON object without Markdown or commentary') from exc


def parse_completion(data):
    if not isinstance(data, dict) or not isinstance(data.get('choices'), list) or len(data['choices']) != 1:
        raise ValueError('Expected exactly one completion choice')
    choice = data['choices'][0]
    if not isinstance(choice, dict) or choice.get('finish_reason') != 'stop':
        raise ValueError(f'Incomplete generation: finish_reason={choice.get("finish_reason") if isinstance(choice, dict) else None!r}; usage={data.get("usage")}')
    message = choice.get('message')
    if not isinstance(message, dict):
        raise ValueError('Missing completion message')
    content, reasoning = message.get('content'), message.get('reasoning_content') or ''
    if not isinstance(content, str) or not isinstance(reasoning, str):
        raise ValueError('Completion content/reasoning must be strings')
    # Only an explicit leading thinking block is removed; never scan or replace inside the answer.
    stripped = content.lstrip()
    for opening, closing in (('<think>', '</think>'), ('[THINK]', '[/THINK]')):
        if stripped.startswith(opening):
            if closing not in stripped:
                raise ValueError('Unclosed thinking block; no visible response')
            internal, content = stripped[len(opening):].split(closing, 1)
            reasoning = reasoning or internal.strip()
            break
    if not content.strip():
        raise ValueError('Empty visible response; reasoning is not substituted for it')
    return {'response': content, 'reasoning_content': reasoning, 'finish_reason': 'stop',
            'usage_json': canonical(data.get('usage', {}))}


class Client:
    def __init__(self, model, config, session=None):
        self.model, self.config = model, config
        # Environment overrides configured defaults; endpoints do not affect checkpoints.
        default_url = config.get('endpoints', {}).get(model['api_url_env'], '')
        self.url = os.environ.get(model['api_url_env'], default_url).strip().rstrip('/')
        parsed = urlsplit(self.url)
        if parsed.scheme not in ('http', 'https') or not parsed.netloc or not parsed.path.endswith('/v1/chat/completions') or parsed.query or parsed.fragment:
            raise ValueError(f'Set {model["api_url_env"]} to the server URL ending in /v1/chat/completions')
        self.base = self.url[:-len('/v1/chat/completions')]
        self.session = session or requests.Session()
        api_key = os.environ.get('MULTITASK_API_KEY')
        if api_key:
            self.session.headers.update({'Authorization': 'Bearer ' + api_key})
        self.context = None

    def request(self, method, url, payload=None):
        result = self.session.request(method, url, json=payload, timeout=(10, self.config['timeout_seconds']))
        if not result.ok:
            raise RuntimeError(f'HTTP {result.status_code} from {url}: {result.text[:1000]}')
        value = result.json()
        if not isinstance(value, dict):
            raise ValueError(f'Expected JSON object from {url}')
        return value

    def preflight(self):
        models = self.request('GET', self.base + '/v1/models')
        aliases = [m.get('id') for m in models.get('data', []) if isinstance(m, dict)]
        if self.model['alias'] not in aliases:
            raise ValueError(f'Expected alias {self.model["alias"]!r}; server advertises {aliases!r}')
        props = self.request('GET', self.base + '/props')
        context = props.get('default_generation_settings', {}).get('n_ctx')
        if type(context) is not int or context < self.config['minimum_context_tokens']:
            raise ValueError(f'Insufficient or unknown context per slot: {context!r}; need at least {self.config["minimum_context_tokens"]}')
        self.context = context
        print(f'Server ready: {self.model["alias"]}, context per slot {context}.')

    def complete(self, payload):
        if self.context is None:
            raise RuntimeError('Server preflight has not passed')
        template_request = {'messages': payload['messages'], 'add_generation_prompt': True}
        if 'chat_template_kwargs' in payload:
            template_request['chat_template_kwargs'] = payload['chat_template_kwargs']
        prompt = self.request('POST', self.base + '/apply-template', template_request).get('prompt')
        if not isinstance(prompt, str) or not prompt:
            raise ValueError('Server did not return a rendered chat template')
        tokens = self.request('POST', self.base + '/tokenize',
                              {'content': prompt, 'add_special': True, 'parse_special': True}).get('tokens')
        if not isinstance(tokens, list):
            raise ValueError('Server did not return token IDs')
        # Conservative margin for server/version-specific chat formatting. Never truncate task data.
        if len(tokens) + payload['max_tokens'] + 256 > self.context:
            raise ValueError(f'Context budget exceeded: {len(tokens)} input + {payload["max_tokens"]} output + 256 margin > {self.context}. Increase server context; text is not truncated.')
        return parse_completion(self.request('POST', self.url, payload))


def read_rows(path, fields):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f, strict=True)
        if reader.fieldnames != fields:
            raise ValueError(f'{path}: unexpected CSV header; file was not modified')
        rows = list(reader)
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError(f'{path}: malformed record; file was not modified')
    return rows


def atomic_rows(path, fields, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8-sig', newline='',
                                         dir=path.parent, prefix='.' + path.name, delete=False) as f:
            name = f.name
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
            f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


@contextmanager
def output_lock(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the inode: unlinking a lock file would allow concurrent writers to bypass it.
    with Path(str(path) + '.lock').open('a') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f'Another process is writing {path}') from exc
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def record_error(path, sample_id, model, error):
    error_path = Path(str(path) + '.errors.jsonl')
    with error_path.open('a', encoding='utf-8') as f:
        f.write(canonical({'created_at': now(), 'sample_id': sample_id, 'model': model,
                           'error': str(error)}) + '\n')


def metadata(sample, source_model):
    return {**{k: sample[k] for k in ('sample_id', 'sample_hash', 'task', 'language_code', 'language', 'source_id')},
            'source_model': source_model}


def response_hash(row):
    return digest({k: row[k] for k in ('sample_id', 'sample_hash', 'source_model', 'run_fingerprint',
                                     'request_json', 'response', 'reasoning_content')})


def generation_path(output_dir, model):
    return Path(output_dir) / 'candidates' / (model + '_responses.csv')


def judgment_path(output_dir, model):
    return Path(output_dir) / 'judges' / (model + '_results.csv')


def length_recovery_model(model):
    """Explicit, fixed recovery profile; baseline config and existing fingerprints stay intact."""
    recovered = copy.deepcopy(model)
    recovered['recovery_profile'] = 'length-v1'
    params = recovered['parameters']
    params['seed'] = 43
    params['max_tokens'] = max(params['max_tokens'], 16384)
    params['reasoning_budget_tokens'] = 4096
    params['reasoning_budget_message'] = (
        'Finish the analysis now and provide the requested task response directly.'
    )
    return recovered


def length_failure_ids(path, model_key):
    """Only opt already failed records into recovery; never change fresh requests."""
    log = Path(str(path) + '.errors.jsonl')
    if not log.exists():
        return set()
    failed = set()
    with log.open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{log}: malformed error-log record {number}; file unchanged') from exc
            if not isinstance(event, dict):
                raise ValueError(f'{log}: invalid error-log record {number}')
            if event.get('model') == model_key and "finish_reason='length'" in str(event.get('error', '')):
                sample_id = event.get('sample_id')
                if isinstance(sample_id, str):
                    failed.add(sample_id)
    return failed


def candidate_rows(output_dir, model_key, manifest, config):
    from prompts import generation_messages
    model = config['models'][model_key]
    variants = {fp: variant for variant in (model, length_recovery_model(model))
                for fp in compatible_fingerprints(manifest, variant, 'generation')}
    samples = {s['sample_id']: s for s in manifest['samples']}
    rows = read_rows(generation_path(output_dir, model_key), GEN_FIELDS)
    seen = set()
    for row in rows:
        sample = samples.get(row['sample_id'])
        if sample is None or row['sample_id'] in seen:
            raise ValueError(f'Unknown or duplicate saved candidate: {row["sample_id"]}')
        seen.add(row['sample_id'])
        if any(row[k] != v for k, v in metadata(sample, model_key).items()):
            raise ValueError('Candidate metadata differs from selection')
        saved_model = variants.get(row['run_fingerprint'])
        if saved_model is None or row['api_model'] != model['alias']:
            raise ValueError(f'Candidate settings/prompts changed for {model_key}; use a new output directory')
        if row['request_json'] != canonical(payload_for(generation_messages(sample), saved_model)):
            raise ValueError('Saved candidate request differs from the current generation prompt')
        if row['finish_reason'] != 'stop' or not row['response'].strip() or row['response_hash'] != response_hash(row):
            raise ValueError(f'Incomplete or modified saved candidate: {row["sample_id"]}')
    return rows


def judged_rows(output_dir, model_key, manifest, config, candidates):
    from prompts import judge_messages
    model = config['models'][model_key]
    expected_fps = compatible_fingerprints(manifest, model, 'judging')
    samples = {s['sample_id']: s for s in manifest['samples']}
    rows = read_rows(judgment_path(output_dir, model_key), JUDGE_FIELDS)
    seen = set()
    for row in rows:
        key = (row['sample_id'], row['source_model'])
        candidate = candidates.get(key)
        if candidate is None or key in seen:
            raise ValueError(f'Unknown/duplicate judgment or missing candidate: {key}')
        seen.add(key); sample = samples[row['sample_id']]
        if any(row[k] != v for k, v in metadata(sample, row['source_model']).items()):
            raise ValueError('Judgment metadata differs from selection')
        if row['judge_model'] != model_key or row['api_model'] != model['alias'] or row['run_fingerprint'] not in expected_fps:
            raise ValueError(f'Judge settings/rubric changed for {model_key}; use a new output directory')
        if row['candidate_hash'] != candidate['response_hash']:
            raise ValueError(f'Candidate changed after judgment: {key}')
        expected_request = payload_for(judge_messages(sample, candidate), model, judging=True)
        if row['request_json'] != canonical(expected_request):
            raise ValueError(f'Changed judge request: {key}')
        parsed = parse_judgment(row['raw_judge_response'])
        if row['overall_score'] != str(parsed['overall_score']) or row['justification'] != parsed['justification'] or row['finish_reason'] != 'stop':
            raise ValueError(f'Invalid saved judgment: {key}')
    return rows


def add_shared_args(parser):
    parser.add_argument('--config', type=Path, default=ROOT / 'config.json')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'data' / 'selection.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--languages', nargs='+', choices=tuple(LANGUAGES))
    parser.add_argument('--tasks', nargs='+', choices=TASKS)
    parser.add_argument('--check-data', action='store_true', help='Validate locally; make no HTTP requests')
    parser.add_argument('--limit', type=int, help='Attempt at most this many unsaved records; incomplete work exits nonzero')


def cli_entry(main, *args):
    try:
        return main(*args)
    except KeyboardInterrupt:
        print('\nStopped. Saved results remain available for resumption.')
        return 130
    except (ValueError, OSError, RuntimeError, csv.Error, requests.RequestException) as exc:
        print(f'ERROR: {exc}')
        return 1
