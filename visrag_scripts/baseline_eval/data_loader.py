"""Read-only dataset adapters preserving original annotations."""
import json
from pathlib import Path
from io_utils import fingerprint


def _load_documents(root: Path, dataset: str) -> list[dict]:
    """Validates document/QA pairing and assigns globally unique keys."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Missing dataset: {root}")
    documents = []
    if dataset == "nation_standard_bench":
        for category in ("Electrical", "Energy", "Medical", "Tobacco"):
            folder = root / category
            if not folder.is_dir():
                raise FileNotFoundError(f"Missing category: {folder}")
            documents.extend((p, p / "qa.jsonl")
                             for p in sorted(folder.iterdir()) if p.is_dir())
    else:
        documents = [(p, p / f"{p.name}_qa.jsonl")
                     for p in sorted(root.iterdir())
                     if p.is_dir() and p.name.isdigit()]
    items, keys = [], set()
    for folder, qa_file in documents:
        pdfs = sorted(folder.glob("*.pdf"))
        if not qa_file.is_file() or len(pdfs) != 1:
            raise ValueError(f"Expected one PDF and a QA file: {folder}")
        document_key = folder.relative_to(root).as_posix()
        for line_no, line in enumerate(
                qa_file.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            question = record.get("question")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"Invalid question: {qa_file}:{line_no}")
            if not isinstance(record.get("answer"), str):
                raise ValueError(f"Invalid answer: {qa_file}:{line_no}")
            qid = (record.get("question_id")
                   if dataset == "nation_standard_bench"
                   else fingerprint(question))
            if not qid:
                raise ValueError(f"Missing question ID: {qa_file}:{line_no}")
            key = fingerprint([dataset, document_key, qid])
            if key in keys:
                raise ValueError(f"Duplicate question: {qa_file}:{line_no}")
            keys.add(key)
            items.append({
                **record, "dataset": dataset, "document_key": document_key,
                "doc_id": record.get("doc_id", folder.name),
                "question_id": qid, "question_key": key,
                "question_type": record.get(
                    "question_type", record.get("type")),
                "ground_truth_answer": record["answer"],
                "pdf_path": str(pdfs[0].resolve()), "source_line": line_no,
            })
    if not items:
        raise ValueError(f"No questions: {root}")
    return items


def load_nation_standard_bench(dataset_dir: Path) -> list[dict]:
    """Loads NationStandardBench."""
    return _load_documents(dataset_dir, "nation_standard_bench")


def load_doc_bench(dataset_dir: Path) -> list[dict]:
    """Loads DocBench with stable question-text IDs."""
    return _load_documents(dataset_dir, "doc_bench")


def load_dataset(dataset_name: str, base_dir: Path) -> list[dict]:
    """Accepts the dataset directory or its parent."""
    names = {"nation_standard_bench": "NationStandardBench",
             "doc_bench": "docBench"}
    if dataset_name not in names:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    root = Path(base_dir)
    target = root / names[dataset_name]
    return _load_documents(target if target.is_dir() else root, dataset_name)
