"""Independently judge the same saved Gemma responses with the approved task rubrics."""
import argparse
import requests
from common import (CANDIDATES, JUDGES, Client, JUDGE_FIELDS, add_shared_args, atomic_rows, candidate_rows,
                    cli_entry, fingerprint, judged_rows, judgment_path, load_config, metadata, now,
                    output_lock, parse_judgment, payload_for, record_error)
from data import canonical, load_manifest, select_samples
from prompts import judge_messages


def run(args, model_key, client_factory=Client):
    config = load_config(args.config); manifest = load_manifest(args.manifest)
    samples = select_samples(manifest, args.languages, args.tasks)
    if args.limit is not None and args.limit < 1:
        raise ValueError('--limit must be positive')
    source_models = CANDIDATES if args.source_models == 'both' else (args.source_models,)
    all_candidates = {}
    for source in CANDIDATES:
        for row in candidate_rows(args.output_dir, source, manifest, config):
            all_candidates[(row['sample_id'], source)] = row
    desired = [(s['sample_id'], source) for s in samples for source in source_models]
    sample_by_id = {s['sample_id']: s for s in manifest['samples']}
    missing = [key for key in desired if key not in all_candidates]
    available = [key for key in desired if key in all_candidates]
    model = config['models'][model_key]
    existing = judged_rows(args.output_dir, model_key, manifest, config, all_candidates)
    payloads = {key: payload_for(judge_messages(sample_by_id[key[0]], all_candidates[key]), model, judging=True)
                for key in available}
    print(f'{model_key}: target {len(desired)}; available candidates {len(available)}; missing candidates {len(missing)}.')
    if args.check_data:
        print('Offline data/prompt/checkpoint checks passed. No API calls made. Missing candidates must be generated before full judging.')
        return 0
    path = judgment_path(args.output_dir, model_key)
    with output_lock(path):
        existing = judged_rows(args.output_dir, model_key, manifest, config, all_candidates)
        saved = {(r['sample_id'], r['source_model']) for r in existing}
        previous = len(set(desired) & saved)
        remaining = [key for key in available if key not in saved]
        succeeded = failed = 0
        if remaining:
            client = client_factory(model, config); client.preflight()
            for key in remaining[:args.limit]:
                print(f'{model_key}: {key[0]} / {key[1]}', flush=True)
                try:
                    result = client.complete(payloads[key])
                    verdict = parse_judgment(result['response'])
                    row = dict(metadata(sample_by_id[key[0]], key[1]), judge_model=model_key,
                               api_model=model['alias'], run_fingerprint=fingerprint(manifest, model, 'judging'),
                               candidate_hash=all_candidates[key]['response_hash'], request_json=canonical(payloads[key]),
                               overall_score=verdict['overall_score'], justification=verdict['justification'],
                               raw_judge_response=result['response'], judge_reasoning=result['reasoning_content'],
                               finish_reason=result['finish_reason'], usage_json=result['usage_json'], created_at=now())
                except (ValueError, RuntimeError, requests.RequestException) as exc:
                    failed += 1; print(f'  FAILED: {exc}', flush=True)
                    record_error(path, '/'.join(key), model_key, exc)
                    continue
                atomic_rows(path, JUDGE_FIELDS, existing + [row])
                existing.append(row); succeeded += 1
        unresolved = len(desired) - previous - succeeded
        print(f'Previously completed: {previous} | Successful: {succeeded} | Failed: {failed} | Missing candidates: {len(missing)} | Remaining: {unresolved}')
        print('Judging completed.' if unresolved == 0 else 'Judging incomplete; generate missing candidates or rerun failed judgments.')
        return int(unresolved != 0)


def main(argv=None, fixed_model=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_shared_args(parser)
    if fixed_model is None:
        parser.add_argument('--model', required=True, choices=JUDGES)
    parser.add_argument('--source-models', choices=('both',) + CANDIDATES, default='both')
    args = parser.parse_args(argv)
    return run(args, fixed_model or args.model)


if __name__ == '__main__':
    raise SystemExit(cli_entry(main))
