"""Run gemma31b across all selected tasks and languages by default."""
from generate import main
from common import cli_entry

if __name__ == "__main__":
    raise SystemExit(cli_entry(lambda: main(fixed_model="gemma31b")))
