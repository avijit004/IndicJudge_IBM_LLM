"""Generate task responses with either Gemma candidate. All languages/tasks by default."""
import argparse
import requests
from common import (CANDIDATES, Client, GEN_FIELDS, add_shared_args, atomic_rows, candidate_rows,
                    cli_entry, fingerprint, generation_path, load_config, metadata, now, output_lock,
                    payload_for, record_error, response_hash, length_recovery_model, length_failure_ids)
from data import canonical, load_manifest, select_samples
from prompts import generation_messages


def run(args, model_key, client_factory=Client):
    config = load_config(args.config); manifest = load_manifest(args.manifest)
    samples = select_samples(manifest, args.languages, args.tasks)
    if args.limit is not None and args.limit < 1:
        raise ValueError('--limit must be positive')
    model = config['models'][model_key]
    path = generation_path(args.output_dir, model_key)
    recovery_ids = length_failure_ids(path, model_key) if getattr(args, 'retry_length', False) else set()
    request_models = {s['sample_id']: (length_recovery_model(model) if s['sample_id'] in recovery_ids else model)
                      for s in samples}
    # Validate all prompts, metadata and prior checkpoints before making any request.
    payloads = {s['sample_id']: payload_for(generation_messages(s), request_models[s['sample_id']]) for s in samples}
    existing = candidate_rows(args.output_dir, model_key, manifest, config)
    print(f'{model_key}: {len(samples)} selected inputs; {len(existing)} saved records in this model file.')
    if args.check_data:
        print('Offline data/prompt/checkpoint checks passed. No API calls made.')
        return 0
    path = generation_path(args.output_dir, model_key)
    with output_lock(path):
        existing = candidate_rows(args.output_dir, model_key, manifest, config)
        saved = {r['sample_id'] for r in existing}
        remaining = [s for s in samples if s['sample_id'] not in saved]
        previous = len(samples) - len(remaining)
        succeeded = failed = 0
        if remaining:
            client = client_factory(model, config); client.preflight()
            for sample in remaining[:args.limit]:
                key = sample['sample_id']; print(f'{model_key}: {key}', flush=True)
                request_model = request_models[key]
                if 'recovery_profile' in request_model:
                    print('  Length recovery: seed 43; output limit 16384; requested thinking budget 4096.', flush=True)
                try:
                    result = client.complete(payloads[key])
                    row = dict(metadata(sample, model_key), run_fingerprint=fingerprint(manifest, request_model, 'generation'),
                               api_model=model['alias'], request_json=canonical(payloads[key]),
                               created_at=now(), **result)
                    row['response_hash'] = response_hash(row)
                    # Saving errors are fatal, not an excuse to call the model again in this run.
                except (ValueError, RuntimeError, requests.RequestException) as exc:
                    failed += 1; print(f'  FAILED: {exc}', flush=True)
                    record_error(path, key, model_key, exc)
                    continue
                atomic_rows(path, GEN_FIELDS, existing + [row])
                existing.append(row); succeeded += 1
        unresolved = len(samples) - previous - succeeded
        print(f'Previously completed: {previous} | Successful: {succeeded} | Failed: {failed} | Remaining: {unresolved}')
        print('Generation completed.' if unresolved == 0 else 'Generation incomplete; rerun to resume.')
        return int(unresolved != 0)


def main(argv=None, fixed_model=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_shared_args(parser)
    parser.add_argument('--retry-length', action='store_true',
                        help='Use a recorded recovery profile for unsaved records with prior length failures only')
    if fixed_model is None:
        parser.add_argument('--model', required=True, choices=CANDIDATES)
    args = parser.parse_args(argv)
    return run(args, fixed_model or args.model)


if __name__ == '__main__':
    raise SystemExit(cli_entry(main))
