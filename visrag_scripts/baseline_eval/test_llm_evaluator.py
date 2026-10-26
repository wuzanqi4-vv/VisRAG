"""Offline evaluator regressions; no test sends real API requests."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import requests

import llm_evaluator as evaluator


class EvaluatorTest(unittest.TestCase):
    """Verifies scoring, actual usage and independent checkpoint recovery."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.settings = {"api_key": "test-secret", "model_name": "judge",
                         "base_url": "https://api.test/v1"}

    def response(self, text, usage=None, finish_reason="stop"):
        """Builds a complete mocked Chat response."""
        response = mock.Mock(status_code=200, headers={})
        response.json.return_value = {
            "choices": [{"message": {"content": text},
                         "finish_reason": finish_reason}],
            "usage": usage, "model": "judge", "id": "request-id"}
        return response

    def evaluate(self, **overrides):
        """Calls the evaluator with a small explicit attempt budget."""
        arguments = {"question": "What is the limit?", "sys_ans": "10 mA",
                     "ref_ans": "10 mA", "ref_text": "Limit is 10 mA.",
                     "settings": self.settings, "max_attempts": 1}
        arguments.update(overrides)
        return evaluator.evaluate_answer(**arguments)

    def test_exact_binary_parsing(self):
        for text, score in [("1", 1), ("- Correctness: 0", 0),
                            ("<think>0 or 1</think>Correctness: 1", 1)]:
            self.assertEqual(evaluator.extract_binary_score(text), score)
        for text in ("0 or 1", "The answer contains 1 item", "10", ""):
            with self.assertRaises(ValueError):
                evaluator.extract_binary_score(text)

    def test_empty_and_zero_do_not_call_api(self):
        with mock.patch("requests.post") as post:
            for text in ("", "0", "<think>reasoning</think> 0"):
                result = self.evaluate(sys_ans=text)
                self.assertEqual(result["score"], 0)
                self.assertEqual(result["usage_source"], "not_called")
                self.assertIsNone(result["total_tokens"])
            post.assert_not_called()

    def test_prompt_usage_and_system_message(self):
        usage = {"prompt_tokens": 12, "completion_tokens": 1,
                 "total_tokens": 13,
                 "completion_tokens_details": {"reasoning_tokens": 0}}
        with mock.patch("requests.post", return_value=self.response(
                "- Correctness: 1", usage)) as post:
            result = self.evaluate()
        self.assertEqual(result["score"], 1)
        self.assertEqual(result["usage_raw"], usage)
        self.assertEqual(result["total_tokens"], 13)
        messages = post.call_args.kwargs["json"]["messages"]
        self.assertEqual(messages[0]["content"], evaluator.EVAL_SYSTEM_PROMPT)
        self.assertIn("Reference Text: Limit is 10 mA.", messages[1]["content"])

    def test_invalid_score_retries_and_preserves_usage(self):
        attempts = []
        with mock.patch("requests.post", side_effect=[
                self.response("0 or 1", {"total_tokens": 9}),
                self.response("0")]), mock.patch("time.sleep"):
            result = self.evaluate(max_attempts=2, on_attempt=attempts.append)
        self.assertEqual(result["score"], 0)
        self.assertEqual(len(attempts), 2)
        self.assertIsNone(attempts[0]["score"])
        self.assertEqual(attempts[0]["total_tokens"], 9)
        self.assertIsNone(attempts[1]["total_tokens"])

    def test_failure_and_truncation_are_not_wrong_answers(self):
        with mock.patch("requests.post", side_effect=requests.Timeout):
            result = self.evaluate()
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["score"])
        self.assertTrue(result["outcome_unknown"])
        with mock.patch("requests.post", return_value=self.response(
                "1", finish_reason="length")):
            self.assertIsNone(self.evaluate()["score"])

    def test_checkpoint_restart_and_accuracy(self):
        rows = [{"question_key": key, "prediction": "answer"}
                for key in ("document_a_q1", "document_b_q1")]
        state = evaluator.EvaluationState(self.root, rows)
        state.record_attempt(rows[0]["question_key"], {
            "status": "success", "score": 1, "total_tokens": 7,
            "latency_sec": 1, "usage_complete": False})
        self.assertIsNone(state.export()["accuracy"])
        state.close()
        state = evaluator.EvaluationState(self.root, rows)
        try:
            self.assertEqual(state.result(rows[0]["question_key"])["score"], 1)
            state.save_result(rows[1]["question_key"], {
                "status": "success", "score": 0,
                "usage_source": "not_called"})
            first = state.export()
            self.assertEqual(first["accuracy"], 0.5)
            self.assertEqual(first["known_total_tokens"], 7)
            self.assertEqual(state.export(), first)
        finally:
            state.close()

    def test_load_input_preserves_evidence_and_qualified_ids(self):
        path = self.root / "predictions.jsonl"
        rows = [{"doc_id": doc, "question_id": "Q001", "question": "q",
                 "prediction": "a", "ground_truth_answer": "a",
                 "evidence_text": "reference"} for doc in ("A", "B")]
        path.write_text("\n".join(json.dumps(row) for row in rows))
        loaded = evaluator.load_answers(path)
        self.assertNotEqual(
            loaded[0]["question_key"], loaded[1]["question_key"])
        self.assertEqual(loaded[0]["reference_text"], "reference")

    def test_cli_resume_does_not_repeat_judge_calls(self):
        path = self.root / "answers.jsonl"
        rows = [{"doc_id": doc, "question_id": "Q001", "question": "q",
                 "prediction": answer, "ground_truth_answer": "a"}
                for doc, answer in (("A", "a"), ("B", ""))]
        path.write_text("\n".join(json.dumps(row) for row in rows))
        settings = {**self.settings, "api_format": "chat_completions"}
        arguments = ["llm_evaluator.py", "--input", str(path)]
        with mock.patch.object(sys, "argv", arguments), \
                mock.patch.object(evaluator.config, "load_settings",
                                  return_value=settings), \
                mock.patch("requests.post", return_value=self.response(
                    "1", {"total_tokens": 15})) as post, \
                redirect_stdout(io.StringIO()):
            evaluator.main()
            evaluator.main()
        self.assertEqual(post.call_count, 1)
        summary = json.loads(next(
            self.root.glob("evaluations/*/summary.json")).read_text())
        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["known_total_tokens"], 15)


if __name__ == "__main__":
    unittest.main()
