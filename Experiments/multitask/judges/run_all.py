"""Run judges sequentially. Their servers must already be running."""
import subprocess
import sys
from utils import ROOT,JUDGES

def main():
    failed=[]
    for judge in JUDGES:
        result=subprocess.run([sys.executable,'-u',str(ROOT/f'judge_{judge}.py'),*sys.argv[1:]])
        if result.returncode:failed.append(judge)
    success = 'All three judges passed offline checks; no inference was run.' if '--check-data' in sys.argv else 'All three judges completed the selected scope.'
    print('Unfinished judges: '+', '.join(failed) if failed else success)
    return int(bool(failed))
if __name__=='__main__':raise SystemExit(main())
