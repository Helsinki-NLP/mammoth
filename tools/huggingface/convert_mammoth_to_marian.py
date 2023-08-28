from transformers import MarianConfig
import sys
import argparse


def convert_mammoth_to_marian(mammoth_config_path: str, marian_config_path: str) -> None:
    """
    """
    with (
        open(mammoth_config_path, "r") if mammoth_config_path is not sys.stdin else sys.stdin as mammoth_config,
        open(marian_config_path, "w") if marian_config_path is not sys.stdout else sys.stdout as marian_config
    ):
        pass

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", "-s", dest="mammoth_config_path", default=sys.stdin)
    parser.add_argument("--tgt", "-t", dest="marian_config_path", default=sys.stdout)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_mammoth_to_marian(**args.__dict__)