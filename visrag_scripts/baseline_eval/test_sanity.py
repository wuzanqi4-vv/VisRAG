"""Explicit read-only dataset sanity command, without credential output."""
import argparse
import config
from data_loader import load_dataset


def main():
    """Reports dataset sizes without sending API requests or loading models."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file")
    args = parser.parse_args()
    settings = config.load_settings(args.env_file)
    for name in ("nation_standard_bench", "doc_bench"):
        items = load_dataset(name, settings["data_dir"])
        print(name, len(items), "unique keys:",
              len({item["question_key"] for item in items}))


if __name__ == "__main__":
    main()
