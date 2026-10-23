"""Unified orchestrator for visual RAG baseline evaluation.

Usage:
    python run_eval.py --dataset nation_standard_bench --retriever visrag --top_k 5 --resume
    python run_eval.py --dataset doc_bench --retriever colpali --top_k 5 --resume
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Set

from config import (
    CACHE_IMAGES_DIR,
    CACHE_INDEX_DIR,
    COLPALI_PATH,
    DOC_BENCH_DIR,
    NATION_STANDARD_BENCH_DIR,
    PDF_RENDER_DPI,
    QWEN_API_KEY,
    QWEN_BASE_URL,
    QWEN_MODEL_NAME,
    RESULTS_DIR,
    TOP_K,
    VISRAG_RET_PATH,
)
from data_loader import load_dataset
from generate_qwen import QwenGenerator
from render_pdf import render_pdf_to_images


def load_completed_question_ids(predictions_file: Path) -> Set[str]:
    """Reads existing predictions file to determine completed questions for resume."""
    completed = set()
    if predictions_file.exists():
        with open(predictions_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        qid = record.get("question_id")
                        if qid:
                            completed.add(qid)
                    except json.JSONDecodeError:
                        continue
    return completed


def main():
    parser = argparse.ArgumentParser(description="Run Visual RAG Baseline Evaluation.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["nation_standard_bench", "doc_bench"],
        help="Dataset to evaluate on.",
    )
    parser.add_argument(
        "--retriever",
        type=str,
        required=True,
        choices=["visrag", "colpali"],
        help="Retriever model to use.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=TOP_K,
        help="Number of retrieved pages to feed to generator (default: 5).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from previous checkpoint if available.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of questions to process (for smoke testing).",
    )
    args = parser.parse_args()

    # 1. Resolve dataset directory and items
    if args.dataset == "nation_standard_bench":
        data_dir = NATION_STANDARD_BENCH_DIR
    else:
        data_dir = DOC_BENCH_DIR

    print(f"=== Starting Visual RAG Baseline Eval ===")
    print(f"Dataset:   {args.dataset} ({data_dir})")
    print(f"Retriever: {args.retriever}")
    print(f"Top-K:     {args.top_k}")
    print(f"Resume:    {args.resume}")

    items = load_dataset(args.dataset, data_dir)
    print(f"Total QA items loaded: {len(items)}")

    if not items:
        print(f"[Error] No QA items found in {data_dir}. Check paths.")
        sys.exit(1)

    if args.limit:
        items = items[: args.limit]
        print(f"Applied limit: evaluating first {args.limit} items.")

    # 2. Setup output directories and files
    out_dir = RESULTS_DIR / args.dataset / f"{args.retriever}_top{args.top_k}"
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_file = out_dir / "predictions.jsonl"
    retrieval_log_file = out_dir / "retrieved_pages.jsonl"
    summary_file = out_dir / "summary.json"

    completed_qids = load_completed_question_ids(predictions_file) if args.resume else set()
    print(f"Already completed items: {len(completed_qids)}")

    # 3. Initialize retriever
    if args.retriever == "visrag":
        from retriever_visrag import VisRAGRetriever
        retriever = VisRAGRetriever(str(VISRAG_RET_PATH))
    else:
        from retriever_colpali import ColPaliRetriever
        retriever = ColPaliRetriever(str(COLPALI_PATH))

    # 4. Initialize Qwen generator
    generator = QwenGenerator(
        api_key=QWEN_API_KEY,
        base_url=QWEN_BASE_URL,
        model_name=QWEN_MODEL_NAME,
    )

    # 5. Process questions
    processed_count = 0
    total_tokens = 0
    total_time = 0.0

    # Open append-only files for immediate persistence
    pred_f = open(predictions_file, "a", encoding="utf-8")
    ret_f = open(retrieval_log_file, "a", encoding="utf-8")

    try:
        for idx, item in enumerate(items):
            qid = item["question_id"]
            if qid in completed_qids:
                continue

            doc_id = item["doc_id"]
            pdf_path = item["pdf_path"]
            question = item["question"]

            print(f"[{idx + 1}/{len(items)}] Processing {doc_id} - {qid}...")

            # Step 5.1: Render PDF pages (cached)
            rendered_images = render_pdf_to_images(
                pdf_path=pdf_path,
                output_dir=CACHE_IMAGES_DIR / args.dataset,
                dpi=PDF_RENDER_DPI,
            )

            # Step 5.2: Document-level top-k retrieval
            pdf_stem = Path(pdf_path).stem
            index_cache = CACHE_INDEX_DIR / args.dataset / args.retriever / f"{pdf_stem}.npy"

            retrieval_start = time.time()
            retrieved = retriever.retrieve(
                query=question,
                image_paths=rendered_images,
                top_k=args.top_k,
                cache_file=index_cache,
            )
            retrieval_time = time.time() - retrieval_start

            retrieved_pages = [r[0] for r in retrieved]
            retrieved_scores = [r[1] for r in retrieved]
            retrieved_img_paths = [r[2] for r in retrieved]

            # Log retrieval
            ret_entry = {
                "doc_id": doc_id,
                "question_id": qid,
                "question": question,
                "retrieved_pages": retrieved_pages,
                "retrieved_scores": retrieved_scores,
                "retrieval_time_sec": retrieval_time,
            }
            ret_f.write(json.dumps(ret_entry, ensure_ascii=False) + "\n")
            ret_f.flush()

            # Step 5.3: Call Qwen API for Generation
            gen_res = generator.generate_answer(
                question=question,
                image_paths=retrieved_img_paths,
            )

            # Log prediction
            pred_entry = {
                "dataset": item["dataset"],
                "doc_id": doc_id,
                "question_id": qid,
                "question": question,
                "ground_truth_answer": item.get("ground_truth_answer", ""),
                "evidence_pages": item.get("evidence_pages", []),
                "retrieved_pages": retrieved_pages,
                "prediction": gen_res.get("prediction", ""),
                "prompt_tokens": gen_res.get("prompt_tokens", 0),
                "completion_tokens": gen_res.get("completion_tokens", 0),
                "total_tokens": gen_res.get("total_tokens", 0),
                "generation_latency_sec": gen_res.get("latency_sec", 0.0),
                "status": gen_res.get("status", "unknown"),
            }
            pred_f.write(json.dumps(pred_entry, ensure_ascii=False) + "\n")
            pred_f.flush()

            processed_count += 1
            total_tokens += gen_res.get("total_tokens", 0)
            total_time += gen_res.get("latency_sec", 0.0)

            print(f"    Done {qid}. Latency: {gen_res.get('latency_sec', 0.0):.2f}s, Tokens: {gen_res.get('total_tokens', 0)}")

    finally:
        pred_f.close()
        ret_f.close()

    # 6. Save summary statistics
    all_completed = load_completed_question_ids(predictions_file)
    summary = {
        "dataset": args.dataset,
        "retriever": args.retriever,
        "top_k": args.top_k,
        "total_questions": len(items),
        "completed_questions": len(all_completed),
        "newly_processed_in_this_run": processed_count,
        "total_tokens_accumulated": total_tokens,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n=== Evaluation Finished ===")
    print(f"Summary saved to: {summary_file}")
    print(f"Total Completed: {len(all_completed)} / {len(items)}")


if __name__ == "__main__":
    main()
