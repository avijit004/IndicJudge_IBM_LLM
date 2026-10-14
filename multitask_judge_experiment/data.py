"""Read-only Dataset import; a frozen selection shared by every model."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from prompts import LANGUAGES, TASKS, LANGUAGE_NAMES

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT.parent / 'Dataset' / 'dataset_csv'
SAMPLES_PER_CELL = 5


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def need(row, *fields):
    for field in fields:
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise ValueError(f'Missing or empty {field!r} in source row {row.get("id")!r}')


def normalize(row, task, code, source_file, source_row):
    need(row, 'id', 'language')
    if row['language'].strip() != LANGUAGES[code]:
        raise ValueError(f'{source_file}: wrong language in {row["id"]}')
    if task == 'qa':
        need(row, 'context', 'question', 'answers')
        answers = json.loads(row['answers'])
        if not isinstance(answers, list) or not answers or any(not isinstance(a, str) or not a.strip() for a in answers):
            raise ValueError(f'{source_file}: invalid reference answers in {row["id"]}')
        input_data = {k: row[k] for k in ('context', 'question')}
        reference = {'answers': answers}
    elif task == 'reasoning':
        need(row, 'premise', 'question', 'choice1', 'choice2', 'label')
        if row['question'] not in ('cause', 'effect'):
            raise ValueError(f'{source_file}: expected a COPA cause/effect row, got {row["id"]}')
        if row.get('choice3', '').strip() or row.get('choice4', '').strip() or row.get('prompt', '').strip():
            raise ValueError(f'{source_file}: unsupported reasoning format in {row["id"]}; review adapter')
        if row['label'] not in ('0', '1'):
            raise ValueError(f'{source_file}: invalid zero-based COPA label in {row["id"]}')
        input_data = {'premise': row['premise'], 'relation': row['question'],
                      'question': ('What was the most plausible cause of this event?' if row['question'] == 'cause'
                                   else 'What was the most plausible effect of this event?'),
                      'options': {'1': row['choice1'], '2': row['choice2']}}
        # Retained solely for provenance. Neither generation nor reasoning judging receives this.
        reference = {'zero_based_label': int(row['label']), 'correct_option': str(int(row['label']) + 1)}
    elif task == 'summarization':
        need(row, 'document', 'summary')
        input_data = {'document': row['document']}
        reference = {'summary': row['summary']}
    elif task == 'translation':
        need(row, 'source_lang', 'target_lang', 'source_text', 'target_text')
        if row['source_lang'] != code or row['target_lang'] not in LANGUAGE_NAMES or row['target_lang'] == code:
            raise ValueError(f'{source_file}: invalid translation direction in {row["id"]}')
        input_data = {k: row[k] for k in ('source_lang', 'target_lang', 'source_text')}
        reference = {'target_text': row['target_text']}
    else:
        raise ValueError(f'Unknown task {task}')
    result = {'sample_id': f'{task}:{code}:{row["id"]}', 'source_id': row['id'], 'language_code': code,
              'language': LANGUAGES[code], 'task': task, 'source_file': source_file,
              'source_row': source_row, 'source_record_hash': digest(row),
              'input': input_data, 'reference': reference}
    result['sample_hash'] = digest(result)
    return result


def build_manifest(dataset_dir):
    dataset_dir = Path(dataset_dir)
    samples, files = [], {}
    # First five records in each source's existing order. No resampling between model runs.
    for task in TASKS:
        for code in LANGUAGES:
            relative = f'{task}/{code}.csv'
            path = dataset_dir / relative
            content = path.read_bytes()
            files[relative] = hashlib.sha256(content).hexdigest()
            with path.open(encoding='utf-8-sig', newline='') as stream:
                reader = csv.DictReader(stream, strict=True)
                if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
                    raise ValueError(f'{path}: invalid CSV header')
                rows = list(reader)
            if len(rows) < SAMPLES_PER_CELL:
                raise ValueError(f'{path}: need at least {SAMPLES_PER_CELL} records, got {len(rows)}')
            ids = set()
            for index, row in enumerate(rows, 1):
                if None in row or any(v is None for v in row.values()):
                    raise ValueError(f'{path}: malformed CSV record {index}')
                sample = normalize(row, task, code, relative, index)
                if sample['source_id'] in ids:
                    raise ValueError(f'{path}: duplicate source ID {sample["source_id"]}')
                ids.add(sample['source_id'])
                if index <= SAMPLES_PER_CELL:
                    samples.append(sample)
    body = {'schema_version': 1, 'selection_method': 'first five records per task and language, in source order',
            'samples_per_cell': SAMPLES_PER_CELL, 'source_file_hashes': files, 'samples': samples}
    return dict(body, manifest_hash=digest(body))


def validate_manifest(manifest):
    if not isinstance(manifest, dict) or manifest.get('schema_version') != 1:
        raise ValueError('Unsupported selection manifest')
    if manifest.get('manifest_hash') != digest({k: v for k, v in manifest.items() if k != 'manifest_hash'}):
        raise ValueError('Selection manifest changed: hash mismatch')
    samples = manifest['samples']
    counts = Counter((s['task'], s['language_code']) for s in samples)
    expected = Counter({(t, c): SAMPLES_PER_CELL for t in TASKS for c in LANGUAGES})
    if counts != expected or len({s['sample_id'] for s in samples}) != 200:
        raise ValueError('Selection must contain exactly five distinct samples in all 40 language/task cells')
    for sample in samples:
        if sample['sample_hash'] != digest({k: v for k, v in sample.items() if k != 'sample_hash'}):
            raise ValueError(f'Changed sample {sample["sample_id"]}')
    return manifest


def load_manifest(path=None):
    path = Path(path) if path else ROOT / 'data' / 'selection.json'
    if not path.exists():
        raise ValueError(f'No selection at {path}. Run prepare_data.py first.')
    return validate_manifest(json.loads(path.read_text(encoding='utf-8')))


def select_samples(manifest, languages=None, tasks=None):
    return [s for s in manifest['samples'] if (not languages or s['language_code'] in languages)
            and (not tasks or s['task'] in tasks)]
