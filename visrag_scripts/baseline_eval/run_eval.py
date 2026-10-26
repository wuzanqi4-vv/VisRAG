"""Run one resumable Top-k benchmark, or export an existing run."""
import argparse
from datetime import datetime, timezone
import json
import importlib.metadata
from pathlib import Path
import time
import uuid
from urllib.parse import urlsplit, urlunsplit

import config
from data_loader import load_dataset
from generate_qwen import PROMPT, QwenGenerator
from io_utils import (
    file_digest, fingerprint, model_identity, run_lock, write_json,
)
from state import ExperimentState


def parse_args() -> argparse.Namespace:
    """Parses server runtime arguments without importing GPU models."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[
        "nation_standard_bench", "doc_bench"])
    parser.add_argument("--retriever", choices=["visrag", "colpali"])
    parser.add_argument("--top_k", "--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--dpi", type=int, default=config.PDF_RENDER_DPI)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--thinking-mode", choices=[
        "provider_default", "enabled", "disabled"], default="provider_default")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if not args.export_only and (not args.dataset or not args.retriever):
        parser.error("--dataset and --retriever are required.")
    for name in ("top_k", "dpi", "max_retries", "max_tokens", "timeout"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive.")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive.")
    return args


def export_existing(directory: Path) -> None:
    """Regenerates artifacts without reading .env or loading a model."""
    if directory is None or not (directory / "state.sqlite3").is_file():
        raise ValueError("--run-dir must identify an existing experiment.")
    with run_lock(directory / "run.lock"):
        state = ExperimentState(directory, [])
        try:
            print(json.dumps(state.export(), indent=2))
        finally:
            state.close()


def main() -> None:
    """Runs generation only; evaluation remains a separate downstream task."""
    args = parse_args()
    if args.export_only:
        export_existing(args.run_dir)
        return
    settings = config.load_settings(args.env_file)
    items = load_dataset(args.dataset, settings["data_dir"])
    if args.limit is not None:
        items = items[:args.limit]
    model_path = settings["model_dir"] / (
        "VisRAG-Ret" if args.retriever == "visrag" else "colpali-v1.3-hf")
    model_signature = model_identity(model_path)
    pdf_hashes = {path: file_digest(path)
                  for path in {item["pdf_path"] for item in items}}
    public_items = [
        {key: value for key, value in item.items() if key != "pdf_path"}
        for item in items
    ]
    endpoint = urlsplit(settings["base_url"])
    public_endpoint = urlunsplit((endpoint.scheme,
                                 endpoint.netloc.rsplit("@", 1)[-1],
                                 endpoint.path, "", ""))
    manifest = {
        "schema": 2, "dataset": args.dataset, "retriever": args.retriever,
        "top_k": args.top_k, "dpi": args.dpi, "limit": args.limit,
        "model_identity": model_signature,
        "data_identity": fingerprint([
            public_items, sorted(pdf_hashes.values())]),
        "api_model": settings["model_name"],
        "api_root": public_endpoint, "api_format": settings["api_format"],
        "max_tokens": args.max_tokens, "thinking_mode": args.thinking_mode,
        "prompt": PROMPT,
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "transformers", "Pillow",
                         "requests", "PyMuPDF")
        },
        "code_identity": fingerprint([
            [path.name, file_digest(path)]
            for path in sorted(Path(__file__).parent.glob("*.py"))
            if not path.name.startswith("test_")
        ]),
    }
    if args.validate_only:
        config.validate_api(settings)
        print(json.dumps({
            "dataset": args.dataset, "questions": len(items),
            "unique_keys": len({item["question_key"] for item in items}),
            "model_path": str(model_path),
            "configured_model": settings["model_name"],
        }, ensure_ascii=False, indent=2))
        return

    if args.run_dir:
        directory = Path(args.run_dir)
    else:
        identity = fingerprint(manifest)
        if not args.resume:
            identity += "_" + uuid.uuid4().hex[:8]
        directory = (settings["artifact_dir"] / "runs" / args.dataset
                     / args.retriever / f"top{args.top_k}_{identity}")
    cache_root = settings["artifact_dir"] / "cache"
    with run_lock(directory / "run.lock"):
        state = ExperimentState(directory, items)
        write_json(directory / "manifest.json", manifest)
        retriever = None
        generator = None
        session_start = time.perf_counter()
        session_timestamp = datetime.now(timezone.utc).isoformat()
        stages = {"render_sec": 0.0, "retrieval_sec": 0.0,
                  "model_load_sec": 0.0, "generation_sec": 0.0}
        rendered = {}
        try:
            for item in items:
                key = item["question_key"]
                previous = state.result(key)
                if previous and previous["status"] in {"success", "truncated"}:
                    continue
                count = state.attempt_count(key)
                if (previous and previous["status"] == "failed"
                        and not args.retry_failed
                        and (not previous.get("retryable")
                             or count >= args.max_retries)):
                    continue
                # API configuration is checked before GPU/render work. Exports
                # and completely cached runs do not need valid credentials.
                if generator is None:
                    config.validate_api(settings)
                    generator = QwenGenerator(
                        settings["api_key"], settings["base_url"],
                        settings["model_name"], max_tokens=args.max_tokens,
                        timeout=args.timeout, thinking_mode=args.thinking_mode)
                retrieval = state.retrieval(key)
                if retrieval and not all(
                        Path(path).is_file()
                        for path in retrieval["image_paths"]):
                    retrieval = None
                if retrieval is None:
                    if retriever is None:
                        started = time.perf_counter()
                        if args.retriever == "visrag":
                            from retriever_visrag import VisRAGRetriever
                            retriever = VisRAGRetriever(str(model_path))
                        else:
                            from retriever_colpali import ColPaliRetriever
                            retriever = ColPaliRetriever(str(model_path))
                        stages["model_load_sec"] += (
                            time.perf_counter() - started)
                    try:
                        from render_pdf import render_pdf_to_images
                        pdf = item["pdf_path"]
                        if pdf not in rendered:
                            started = time.perf_counter()
                            rendered[pdf] = render_pdf_to_images(
                                pdf, cache_root / "pages", args.dpi)
                            stages["render_sec"] += (
                                time.perf_counter() - started)
                        images = rendered[pdf]
                        index_id = fingerprint([
                            pdf_hashes[pdf], args.dpi, model_signature,
                            manifest["code_identity"],
                            manifest["dependencies"]])
                        extension = (
                            ".npy" if args.retriever == "visrag" else ".npz")
                        cache = (cache_root / "indexes" / args.retriever
                                 / (index_id + extension))
                        started = time.perf_counter()
                        ranked = retriever.retrieve(
                            item["question"], images, args.top_k, cache)
                        elapsed = time.perf_counter() - started
                        stages["retrieval_sec"] += elapsed
                        retrieval = {
                            "pages": [row[0] for row in ranked],
                            "scores": [row[1] for row in ranked],
                            "image_paths": [row[2] for row in ranked],
                            "retrieval_time_sec": elapsed,
                            "requested_top_k": args.top_k,
                            "actual_top_k": len(ranked),
                        }
                        state.save_retrieval(key, retrieval)
                    except (OSError, ValueError, RuntimeError) as error:
                        state.save_result(key, {
                            "status": "failed", "prediction": None,
                            "stage": "retrieval", "retryable": False,
                            "error": str(error)})
                        state.export()
                        continue
                budget = (args.max_retries if args.retry_failed
                          else args.max_retries - count)
                started = time.perf_counter()
                result = generator.generate_answer(
                    item["question"], retrieval["image_paths"],
                    max_retries=budget,
                    on_attempt=lambda record: state.record_attempt(key, record))
                stages["generation_sec"] += time.perf_counter() - started
                state.save_result(key, result)
                state.export()
                print(f"{item['document_key']} / {item['question_id']}: "
                      f"{result['status']}")
                if result.get("http_status") in {401, 403}:
                    raise RuntimeError(
                        "API authentication rejected; "
                        "remaining tasks preserved.")
        finally:
            # Each execution gets an independent timing record, never replacing
            # historical elapsed time when a resume performs no new work.
            stages["session_elapsed_sec"] = time.perf_counter() - session_start
            stages["started_at"] = session_timestamp
            write_json(directory / "sessions" / (uuid.uuid4().hex + ".json"),
                       stages)
            state.export()
            state.close()
            print(f"Results: {directory}")


if __name__ == "__main__":
    main()
