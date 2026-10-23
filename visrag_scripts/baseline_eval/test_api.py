"""Explicit API smoke command; import/collection never sends a request."""
import argparse
import json
import config
from generate_qwen import QwenGenerator


def main():
    """Tests credentials only when the user explicitly runs this script."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file")
    args = parser.parse_args()
    settings = config.load_settings(args.env_file)
    config.validate_api(settings)
    generator = QwenGenerator(
        settings["api_key"], settings["base_url"], settings["model_name"])
    print(json.dumps(generator.generate_answer(
        "What is 1 + 1? Answer with a number.", [], max_retries=1),
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
