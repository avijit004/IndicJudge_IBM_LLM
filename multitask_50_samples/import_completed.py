"""Import validated first-five checkpoints without rewriting requests or hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from common import ROOT, CANDIDATES, JUDGES, candidate_rows, judged_rows, load_config, cli_entry
from data import load_manifest


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-folder',type=Path,default=ROOT.parent/'multitask_judge_experiment')
    args=parser.parse_args(argv)
    source=args.from_folder.resolve()
    if source==ROOT:raise ValueError('Source must be a separate experiment')
    if (ROOT/'outputs').exists():
        raise ValueError('outputs/ already exists. Import runs only before new work; existing results are unchanged.')
    config=load_config(); old_config=load_config(source/'config.json')
    if config['models']!=old_config['models'] or sha(ROOT/'prompts.py')!=sha(source/'prompts.py'):
        raise ValueError('Prompts or model settings differ. No checkpoints imported.')
    manifest=load_manifest(); origin=load_manifest(source/'data'/'selection.json')
    if (origin.get('start_row',1),origin.get('end_row',5))!=(1,5):
        raise ValueError('This importer expects the completed first-five selection')
    samples={s['sample_id']:s for s in manifest['samples']}
    if origin['source_file_hashes']!=manifest['source_file_hashes'] or any(samples.get(s['sample_id'])!=s for s in origin['samples']):
        raise ValueError('Original selection is not an identical subset of the full selection')
    origin_path=ROOT/'data'/'checkpoint_origins'/'first_five.json'
    origin_path.parent.mkdir(parents=True,exist_ok=True)
    if origin_path.exists() and load_manifest(origin_path)!=origin:raise ValueError('Existing origin manifest differs')
    if not origin_path.exists():
        with origin_path.open('x',encoding='utf-8') as f:json.dump(origin,f,ensure_ascii=False,indent=2)
    audit={'source_folder':str(source),'source_manifest_hash':origin['manifest_hash'],
           'destination_manifest_hash':manifest['manifest_hash'],'files':{},'counts':{}}
    with tempfile.TemporaryDirectory(prefix='.import-',dir=ROOT) as temp:
        staged=Path(temp)/'outputs';staged.mkdir()
        relative=[Path('candidates')/(m+'_responses.csv') for m in CANDIDATES]+[Path('judges')/(m+'_results.csv') for m in JUDGES]
        for rel in relative:
            original=source/'outputs'/rel
            if original.exists():
                before=sha(original);dest=staged/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(original,dest)
                if sha(dest)!=before or sha(original)!=before:raise ValueError('Source changed during import; retry after it stops writing')
                audit['files'][str(rel)]=before
        lookup={}
        for model in CANDIDATES:
            rows=candidate_rows(staged,model,manifest,config)
            lookup.update({(r['sample_id'],model):r for r in rows})
            audit['counts'][model]=len(rows)
        for model in JUDGES:
            audit['counts'][model]=len(judged_rows(staged,model,manifest,config,lookup))
        (staged/'import_provenance.json').write_text(json.dumps(audit,indent=2)+'\n')
        # Publish the fully validated set together. Never overwrite existing results.
        if (ROOT/'outputs').exists():raise ValueError('Another run created outputs during import')
        staged.rename(ROOT/'outputs')
    print('Imported original CSVs byte-for-byte. Future runs skip these completed records.')
    print(json.dumps(audit['counts'],indent=2))
    return 0

if __name__=='__main__':raise SystemExit(cli_entry(main))
