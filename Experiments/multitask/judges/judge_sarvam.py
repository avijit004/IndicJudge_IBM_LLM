from judge import main
from transport import cli_entry

if __name__ == "__main__":
    raise SystemExit(cli_entry(lambda: main(fixed_model="sarvam")))
