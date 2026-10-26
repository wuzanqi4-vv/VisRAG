"""Independent, resumable binary LLM evaluation of saved benchmark answers."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time

import requests

import config
from generate_qwen import redact, usage_fields
from io_utils import file_digest, fingerprint, run_lock, write_json, write_jsonl
from state import ExperimentState

EVAL_SYSTEM_PROMPT = "You are a helpful evaluator."
EVAL_USER_PROMPT_TEMPLATE = """
Task Overview:
You are tasked with evaluating user answers based on a given question, reference
answer, and additional reference text. Your goal is to assess the correctness of
the user answer using a specific metric.

Evaluation Criteria:
1. Yes/No Questions: Verify if the user's answer aligns with the reference
answer in terms of a "yes" or "no" response.
2. Short Answers/Directives: Ensure key details such as numbers, specific
nouns/verbs, and dates match those in the reference answer.
3. Abstractive/Long Answers: The user's answer can differ in wording but must
convey the same meaning and contain the same key information as the reference
answer to be considered correct.

Evaluation Process:
1. Identify the type of question presented.
2. Apply the relevant criteria from the Evaluation Criteria.
3. Compare the user's answer against the reference answer accordingly.
4. Consult the reference text for clarification when needed.
5. Score the answer with a binary label 0 or 1, where 0 denotes wrong and
1 denotes correct.

NOTE that if the user answer is 0 or an empty string, it should get a 0 score.

Question: {question}
User Answer: {sys_ans}
Reference Answer: {ref_ans}
Reference Text: {ref_text}

Evaluation Form (score ONLY):
- Correctness:
""".strip()


def strip_think_tags(text: str) -> str:
    """Removes complete reasoning blocks, preserving the final answer."""
    return re.sub(r"<think>.*?</think>", " ", text,
                  flags=re.DOTALL | re.IGNORECASE).strip()


def extract_binary_score(text: str) -> int:
    """Accepts only an unambiguous binary answer or Correctness field.

    Raises:
        ValueError: The response is not a single binary score.
    """
    cleaned = strip_think_tags(text)
    match = re.fullmatch(
        r"(?:(?:-\s*)?Correctness\s*:\s*)?([01])", cleaned,
        flags=re.IGNORECASE)
    if not match:
        raise ValueError("Evaluator must return 0, 1 or Correctness: 0/1.")
    return int(match.group(1))


def build_eval_prompt(question: str, sys_ans: str, ref_ans: str,
                      ref_text: str = "") -> str:
    """Formats the supplied evaluation protocol without changing answers."""
    return EVAL_USER_PROMPT_TEMPLATE.format(
        question=question.strip(), sys_ans=sys_ans.strip(),
        ref_ans=ref_ans.strip(), ref_text=ref_text.strip())


def evaluate_answer(*, question: str, sys_ans: str, ref_ans: str,
                    ref_text: str = "", settings: dict, timeout: float = 120,
                    max_tokens: int = 512, max_attempts: int = 3,
                    thinking_mode: str = "provider_default",
                    on_attempt=None) -> dict:
    """Evaluates one answer and checkpoints every actual judge request.

    Args:
        question: Original question.
        sys_ans: Generated answer, excluding separately stored reasoning.
        ref_ans: Reference answer.
        ref_text: Reference evidence text, or an empty string if unavailable.
        settings: Validated Chat API settings from config.load_settings.
        timeout: Per-request timeout in seconds.
        max_tokens: Judge output budget.
        max_attempts: Maximum API attempts for this invocation.
        thinking_mode: Provider default, enabled, or disabled thinking.
        on_attempt: Durable callback for each actual API attempt.

    Returns:
        Binary score, status, raw judge response, API usage and timing.
        Failed evaluation has score=None, not score=0.
    """
    clean_answer = strip_think_tags(sys_ans)
    if clean_answer in {"", "0"}:
        return {
            "status": "success", "score": 0, "correct": False,
            "reason": "empty_or_zero_prediction", "raw": "0",
            "usage_raw": None, "usage_source": "not_called",
            "prompt_tokens": None, "completion_tokens": None,
            "total_tokens": None, "latency_sec": 0.0,
            "evaluation_elapsed_sec": 0.0,
        }
    if max_attempts <= 0:
        raise ValueError("At least one evaluation attempt is required.")
    prompt = build_eval_prompt(question, clean_answer, ref_ans, ref_text)
    payload = {
        "model": settings["model_name"], "temperature": 0,
        "max_tokens": max_tokens, "stream": False,
        "messages": [
            {"role": "system", "content": EVAL_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }
    if thinking_mode != "provider_default":
        payload["enable_thinking"] = thinking_mode == "enabled"
    overall = time.perf_counter()
    for attempt in range(max_attempts):
        started_at = datetime.now(timezone.utc).isoformat()
        started = time.perf_counter()
        raw, http_status, output = {}, None, ""
        score, error, retryable, finish_reason = None, None, False, None
        request_id, retry_after = None, 0.0
        try:
            response = requests.post(
                settings["base_url"].rstrip("/") + "/chat/completions",
                headers={"Authorization": "Bearer " + settings["api_key"]},
                json=payload, timeout=timeout)
            http_status = response.status_code
            request_id = response.headers.get("x-request-id")
            try:
                retry_after = float(response.headers.get("Retry-After", 0))
            except (ValueError, TypeError):
                pass
            decoded = response.json()
            if not isinstance(decoded, dict):
                raise ValueError("Judge response is not a JSON object.")
            raw = redact(decoded, settings["api_key"])
            if http_status != 200:
                error = f"HTTP {http_status}"
                retryable = http_status in {408, 429} or http_status >= 500
            else:
                choice = raw["choices"][0]
                finish_reason = choice.get("finish_reason")
                output = choice["message"]["content"]
                if not isinstance(output, str) or finish_reason == "length":
                    raise ValueError("Missing or truncated judge answer.")
                score = extract_binary_score(output)
        except requests.RequestException as exception:
            error, retryable = type(exception).__name__, True
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            error = "Invalid or ambiguous evaluator response"
            retryable = http_status in {None, 200, 408, 429}
            retryable = retryable or (http_status is not None
                                      and http_status >= 500)
        result = {
            "status": "success" if score is not None else "failed",
            "score": score,
            "correct": bool(score) if score is not None else None,
            "reason": error, "raw": output, "response_raw": raw,
            "http_status": http_status, "retryable": retryable,
            "outcome_unknown": http_status is None,
            "finish_reason": finish_reason,
            "request_id": raw.get("id") or request_id,
            "response_model": raw.get("model"),
            "latency_sec": time.perf_counter() - started,
            "evaluation_elapsed_sec": time.perf_counter() - overall,
            "started_at": started_at,
            **usage_fields(raw),
        }
        # Propagate checkpoint errors instead of repeating a billed request.
        if on_attempt:
            on_attempt(result)
        if score is not None or not retryable:
            return result
        if attempt + 1 < max_attempts:
            time.sleep(min(max(2 ** attempt, retry_after), 60))
    return result


class EvaluationState(ExperimentState):
    """Reuses transactional checkpoints, exporting only judge artifacts."""

    def export(self) -> dict:
        """Writes judgments and full-snapshot Accuracy using actual usage."""
        rows = []
        for item, result in self.connection.execute(
                "SELECT item,result FROM questions ORDER BY key"):
            rows.append({**json.loads(item), **(
                json.loads(result) if result else
                {"status": "pending", "score": None})})
        attempts = [
            {"attempt_id": number, "question_key": key, **json.loads(record)}
            for number, key, record in self.connection.execute(
                "SELECT id,key,record FROM attempts ORDER BY id")
        ]
        scored = [row for row in rows if row.get("score") is not None]
        correct = sum(row["score"] for row in scored)
        summary = {
            "total_questions": len(rows), "scored_questions": len(scored),
            "correct_questions": correct,
            "accuracy": (correct / len(rows)
                         if rows and len(scored) == len(rows) else None),
            "failed_evaluations": sum(
                row["status"] == "failed" for row in rows),
            "api_attempts": len(attempts),
            "missing_usage_attempts": sum(
                not row.get("usage_complete", False) for row in attempts),
            "usage_complete": all(row.get("usage_complete", False)
                                  for row in attempts),
            "api_latency_sum_sec": sum(row["latency_sec"] for row in attempts),
        }
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            summary["known_" + field] = sum(
                row[field] for row in attempts if row.get(field) is not None)
        write_jsonl(self.directory / "judgments.jsonl", rows)
        write_jsonl(self.directory / "api_requests.jsonl", attempts)
        write_json(self.directory / "summary.json", summary)
        return summary


def load_answers(path: Path) -> list[dict]:
    """Loads new or legacy JSONL predictions, preserving reference evidence."""
    rows, keys = [], set()
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = row.get("question_key") or fingerprint([
            row.get("dataset"), row.get("document_key", row.get("doc_id")),
            row.get("question_id"), row.get("question")])
        if key in keys:
            raise ValueError(f"Duplicate question key in {path}: {key}")
        keys.add(key)
        reference = row.get("ground_truth_answer", row.get("answer"))
        if not isinstance(row.get("question"), str) or not isinstance(
                reference, str):
            raise ValueError("Every row needs question and reference answer.")
        answer = row.get("prediction", row.get("predicted_answer"))
        if answer is not None and not isinstance(answer, (str, int, float)):
            raise ValueError("Prediction must be text, a number or null.")
        evidence = next((row[field] for field in (
            "ref_text", "reference_text", "evidence_text", "evidence")
            if row.get(field)), "")
        if not isinstance(evidence, str):
            evidence = json.dumps(evidence, ensure_ascii=False)
        rows.append({
            "question_key": key, "dataset": row.get("dataset"),
            "doc_id": row.get("doc_id"), "question_id": row.get("question_id"),
            "question_type": row.get("question_type", row.get("type")),
            "question": row["question"], "prediction": str(answer)
            if answer is not None else "", "reference_answer": reference,
            "reference_text": evidence,
            "generation_status": row.get("status", "success"),
        })
    if not rows:
        raise ValueError("Input file contains no answers.")
    return rows


def main() -> None:
    """Evaluates a frozen prediction snapshot using configured Chat settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--model", help="Override the Chat judge model ID.")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--thinking-mode", choices=[
        "provider_default", "enabled", "disabled"], default="provider_default")
    args = parser.parse_args()
    if min(args.timeout, args.max_tokens, args.max_attempts) <= 0:
        parser.error(
            "Timeout, output budget and attempt count must be positive.")
    rows = load_answers(args.input)
    settings = config.load_settings(args.env_file)
    if args.model:
        settings["model_name"] = args.model
    config.validate_api(settings)
    manifest = {
        "source": str(args.input.resolve()), "answers": fingerprint(rows),
        "system_prompt": EVAL_SYSTEM_PROMPT,
        "user_prompt": EVAL_USER_PROMPT_TEMPLATE,
        "model": settings["model_name"], "base_url": settings["base_url"],
        "max_tokens": args.max_tokens, "thinking_mode": args.thinking_mode,
        "protocol_version": "binary-correctness-v1",
        "evaluator_code": file_digest(__file__),
    }
    root = args.output_dir or args.input.parent / "evaluations"
    directory = root / fingerprint(manifest)
    with run_lock(directory / "run.lock"):
        state = EvaluationState(directory, rows)
        write_json(directory / "manifest.json", manifest)
        try:
            for row in rows:
                key = row["question_key"]
                previous = state.result(key)
                if previous and previous["status"] == "success":
                    continue
                if row["generation_status"] == "pending":
                    continue  # Unfinished generation is not a wrong answer.
                count = state.attempt_count(key)
                if count >= args.max_attempts and not args.retry_failed:
                    continue
                budget = (args.max_attempts if args.retry_failed
                          else args.max_attempts - count)
                result = evaluate_answer(
                    question=row["question"], sys_ans=row["prediction"],
                    ref_ans=row["reference_answer"],
                    ref_text=row["reference_text"], settings=settings,
                    timeout=args.timeout, max_tokens=args.max_tokens,
                    max_attempts=budget, thinking_mode=args.thinking_mode,
                    on_attempt=lambda record: state.record_attempt(key, record))
                state.save_result(key, result)
                state.export()
                if result.get("http_status") in {401, 403}:
                    raise RuntimeError("Judge authentication rejected.")
        finally:
            summary = state.export()
            state.close()
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Evaluation results: {directory}")


if __name__ == "__main__":
    main()
