"""Print one exact judge request for offline inspection; never calls a model."""
import argparse
import json
from core import add_args,load_config,load_snapshot,request_for
from utils import JUDGES
from rubric import LANGUAGES,TASKS
from transport import cli_entry

def main():
    p=argparse.ArgumentParser(description=__doc__);add_args(p)
    p.add_argument('--judge',choices=JUDGES,default='qwen')
    p.add_argument('--language',choices=tuple(LANGUAGES),default='bn')
    p.add_argument('--task',choices=TASKS,default='qa')
    args=p.parse_args();s=load_snapshot(args.snapshot);c=load_config(args.config)
    item=next((i for i in s['items'] if i['language_code']==args.language and i['task']==args.task),None)
    if item is None:raise ValueError('No candidate matches that language/task')
    print(json.dumps(request_for(item,c,args.judge),ensure_ascii=False,indent=2));return 0
if __name__=='__main__':raise SystemExit(cli_entry(main))
