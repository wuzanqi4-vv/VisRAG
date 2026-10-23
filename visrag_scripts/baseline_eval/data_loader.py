"""Data loaders for NationStandardBench and DocBench.

Uniformly normalizes dataset items into standardized evaluation dictionaries:
{
    "dataset": str,
    "doc_id": str,
    "question_id": str,
    "question": str,
    "ground_truth_answer": str,
    "evidence_pages": list[int],
    "pdf_path": str,
}
"""

import json
from pathlib import Path
from typing import Any, Dict, List


def load_nation_standard_bench(dataset_dir: Path) -> List[Dict[str, Any]]:
    """Loads all QA items and associated PDF paths from NationStandardBench."""
    items: List[Dict[str, Any]] = []
    if not dataset_dir.exists():
        print(f"[Warning] Dataset path does not exist: {dataset_dir}")
        return items

    categories = ["Electrical", "Energy", "Medical", "Tobacco"]
    for category in categories:
        cat_dir = dataset_dir / category
        if not cat_dir.is_dir():
            continue

        for doc_dir in sorted(cat_dir.iterdir()):
            if not doc_dir.is_dir():
                continue

            qa_file = doc_dir / "qa.jsonl"
            pdf_candidates = list(doc_dir.glob("*.pdf"))
            if not qa_file.exists() or not pdf_candidates:
                continue

            pdf_path = str(pdf_candidates[0].resolve())
            doc_id = doc_dir.name

            with open(qa_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    items.append({
                        "dataset": "NationStandardBench",
                        "category": category,
                        "doc_id": doc_id,
                        "question_id": record.get("question_id", ""),
                        "question": record.get("question", ""),
                        "ground_truth_answer": record.get("answer", ""),
                        "evidence_pages": record.get("evidence_pages", []),
                        "pdf_path": pdf_path,
                    })

    return items


def load_doc_bench(dataset_dir: Path) -> List[Dict[str, Any]]:
    """Loads all QA items and associated PDF paths from DocBench."""
    items: List[Dict[str, Any]] = []
    if not dataset_dir.exists():
        print(f"[Warning] Dataset path does not exist: {dataset_dir}")
        return items

    for doc_dir in sorted(dataset_dir.iterdir()):
        if not doc_dir.is_dir() or not doc_dir.name.isdigit():
            continue

        doc_idx = doc_dir.name
        qa_file = doc_dir / f"{doc_idx}_qa.jsonl"
        pdf_candidates = list(doc_dir.glob("*.pdf"))

        if not qa_file.exists() or not pdf_candidates:
            continue

        pdf_path = str(pdf_candidates[0].resolve())

        with open(qa_file, "r", encoding="utf-8") as f:
            for q_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                items.append({
                    "dataset": "DocBench",
                    "doc_id": doc_idx,
                    "question_id": f"doc_{doc_idx}_q{q_idx}",
                    "question": record.get("question", ""),
                    "ground_truth_answer": record.get("answer", ""),
                    "evidence_pages": [],  # DocBench evidence is text snippet
                    "evidence_text": record.get("evidence", ""),
                    "pdf_path": pdf_path,
                })

    return items


def load_dataset(dataset_name: str, base_dir: Path) -> List[Dict[str, Any]]:
    """Convenience dispatcher to load specified dataset."""
    name_lower = dataset_name.lower().replace("-", "_").replace(" ", "_")
    if "nation" in name_lower:
        target_dir = base_dir / "NationStandardBench" if (base_dir / "NationStandardBench").exists() else base_dir
        return load_nation_standard_bench(target_dir)
    elif "doc" in name_lower:
        target_dir = base_dir / "docBench" if (base_dir / "docBench").exists() else base_dir
        return load_doc_bench(target_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Expected 'NationStandardBench' or 'DocBench'.")
