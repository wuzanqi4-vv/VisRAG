"""Transactional experiment state; exports can always be rebuilt."""
import json
import csv
from pathlib import Path
import sqlite3

from io_utils import atomic_path, write_json, write_jsonl


class ExperimentState:
    """Stores references, generation attempts and final answers separately."""

    def __init__(self, directory: Path, items: list[dict]) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.directory / "state.sqlite3")
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS questions (
                key TEXT PRIMARY KEY, item TEXT NOT NULL,
                result TEXT, retrieval TEXT
            );
            CREATE TABLE IF NOT EXISTS attempts (
                id INTEGER PRIMARY KEY, key TEXT NOT NULL, record TEXT NOT NULL
            );
        """)
        with self.connection:
            for item in items:
                self.connection.execute(
                    "INSERT OR IGNORE INTO questions(key,item) VALUES (?,?)",
                    (item["question_key"],
                     json.dumps(item, ensure_ascii=False)))

    def result(self, key: str) -> dict | None:
        """Returns the last committed result, if any."""
        row = self.connection.execute(
            "SELECT result FROM questions WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row and row[0] else None

    def attempt_count(self, key: str) -> int:
        """Returns the persisted attempt count, including failed requests."""
        return self.connection.execute(
            "SELECT COUNT(*) FROM attempts WHERE key=?", (key,)).fetchone()[0]

    def save_retrieval(self, key, retrieval):
        """Commits retrieval so an interrupted generation need not redo it."""
        with self.connection:
            self.connection.execute(
                "UPDATE questions SET retrieval=? WHERE key=?",
                (json.dumps(retrieval), key))

    def retrieval(self, key):
        """Returns persisted ranked evidence and measured retrieval time."""
        row = self.connection.execute(
            "SELECT retrieval FROM questions WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row and row[0] else None

    def save_result(self, key, result):
        """Commits a terminal result without inventing an API attempt."""
        with self.connection:
            self.connection.execute(
                "UPDATE questions SET result=? WHERE key=?",
                (json.dumps(result, ensure_ascii=False), key))

    def record_attempt(self, key, record):
        """Atomically commits a response, usage and its terminal status."""
        encoded = json.dumps(record, ensure_ascii=False)
        with self.connection:
            self.connection.execute(
                "INSERT INTO attempts(key,record) VALUES (?,?)", (key, encoded))
            self.connection.execute(
                "UPDATE questions SET result=? WHERE key=?", (encoded, key))

    def export(self) -> dict:
        """Rebuilds full-run outputs; never invokes generation or a judge."""
        attempts = [
            {"attempt_id": number, "question_key": key, **json.loads(record)}
            for number, key, record in self.connection.execute(
                "SELECT id,key,record FROM attempts ORDER BY id")
        ]
        rows, retrieved = [], []
        for key, item, result, retrieval in self.connection.execute(
                "SELECT key,item,result,retrieval FROM questions ORDER BY key"):
            row = json.loads(item)
            row.update(json.loads(result) if result else {
                "status": "pending", "prediction": None})
            row["retrieval"] = json.loads(retrieval) if retrieval else None
            rows.append(row)
            if row["retrieval"]:
                retrieved.append({"question_key": key, **row["retrieval"]})
        write_jsonl(self.directory / "predictions.jsonl", rows)
        write_jsonl(self.directory / "evaluation_input.jsonl", [
            {key: row.get(key) for key in (
                "dataset", "document_key", "doc_id", "question_id",
                "question_key", "question", "ground_truth_answer",
                "answer_normalized", "question_type", "prediction", "status",
                "evidence_text", "evidence", "reference_text")}
            for row in rows
        ])
        write_jsonl(self.directory / "retrieved_pages.jsonl", retrieved)
        write_jsonl(self.directory / "api_requests.jsonl", attempts)
        columns = ("dataset", "document_key", "question_id", "question",
                   "ground_truth_answer", "prediction", "status",
                   "finish_reason", "prompt_tokens", "completion_tokens",
                   "total_tokens", "usage_source", "usage_complete",
                   "latency_sec", "generation_elapsed_sec")
        with atomic_path(self.directory / "results.csv") as temporary:
            with temporary.open(
                    "w", encoding="utf-8-sig", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=columns)
                writer.writeheader()
                writer.writerows({key: row.get(key) for key in columns}
                                 for row in rows)
        summary = {
            "total_questions": len(rows),
            "completed_questions": sum(
                row["status"] in {"success", "truncated"} for row in rows),
            "failed_questions": sum(row["status"] == "failed" for row in rows),
            "truncated_questions": sum(
                row["status"] == "truncated" for row in rows),
            "api_attempts": len(attempts),
            "usage_complete": all(
                record.get("usage_complete", False) for record in attempts),
            "missing_usage_attempts": sum(
                not record.get("usage_complete", False) for record in attempts),
            "api_latency_sum_sec": sum(
                record.get("latency_sec", 0) for record in attempts),
        }
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            summary["known_" + name] = sum(
                record[name] for record in attempts
                if record.get(name) is not None)
        sessions = [json.loads(path.read_text(encoding="utf-8"))
                    for path in (self.directory / "sessions").glob("*.json")]
        summary["execution_sessions"] = len(sessions)
        for name in ("session_elapsed_sec", "render_sec", "retrieval_sec",
                     "model_load_sec", "generation_sec"):
            summary[name + "_sum"] = sum(row.get(name, 0) for row in sessions)
        write_json(self.directory / "summary.json", summary)
        return summary

    def close(self) -> None:
        """Closes the database connection."""
        self.connection.close()
