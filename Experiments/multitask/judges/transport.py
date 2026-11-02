"""Local copy of validated transport/storage helpers; no inference on import."""
import csv
import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
import requests
from utils import canonical
from rubric import JUDGMENT_SCHEMA

# Long multilingual documents and serialized requests can exceed Python's 128 KiB CSV default.
csv.field_size_limit(16 * 1024 * 1024)


def now():
    return datetime.now(timezone.utc).isoformat()


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
        api_key = os.environ.get('RF_API_KEY')
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
        try:
            return parse_completion(self.request('POST', self.url, payload))
        except ValueError as exc:
            if 'finish_reason' in str(exc):
                for t, s in ((0.1, payload.get('seed', 42) + 1), (0.2, payload.get('seed', 42) + 2)):
                    try:
                        retry_payload = dict(payload, temperature=t, seed=s)
                        return parse_completion(self.request('POST', self.url, retry_payload))
                    except ValueError:
                        continue
            raise


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


def cli_entry(main, *args):
    try:
        return main(*args)
    except KeyboardInterrupt:
        print('\nStopped. Saved results remain available for resumption.')
        return 130
    except (ValueError, OSError, RuntimeError, csv.Error, requests.RequestException) as exc:
        print(f'ERROR: {exc}')
        return 1
