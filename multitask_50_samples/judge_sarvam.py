"""Run sarvam across all selected tasks and languages by default."""
from judge import main
from common import cli_entry

if __name__ == "__main__":
    raise SystemExit(cli_entry(lambda: main(fixed_model="sarvam")))
