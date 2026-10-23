"""Offline regressions. Run with unittest on the Linux server."""
import importlib
import json
import os
from pathlib import Path
import tempfile
import sys
import types
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import torch

import config
from generate_qwen import QwenGenerator, usage_fields
from io_utils import run_lock
from retriever_colpali import ColPaliRetriever
from state import ExperimentState


class BenchmarkTest(unittest.TestCase):
    """Exercises persisted responses rather than mirroring helper internals."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_chat_settings_and_no_import_api(self):
        env = self.root / "settings.env"
        env.write_text(
            'CHAT_API_KEY="secret"\nCHAT_BASE_URL=https://api.test/v1\n'
            'CHAT_MODEL=actual-model\n', encoding="utf-8")
        with mock.patch.dict(os.environ, {}, clear=True):
            settings = config.load_settings(env)
            config.validate_api(settings)
            self.assertEqual(settings["api_key"], "secret")
            self.assertEqual(settings["model_name"], "actual-model")
        with mock.patch("requests.post") as post:
            import test_api
            importlib.reload(test_api)
            post.assert_not_called()

    def test_usage_missing_is_unknown(self):
        fields = usage_fields({})
        self.assertIsNone(fields["total_tokens"])
        self.assertFalse(fields["usage_complete"])
        fields = usage_fields({"usage": {"prompt_tokens": 3}})
        self.assertEqual(fields["prompt_tokens"], 3)
        self.assertIsNone(fields["total_tokens"])

    def test_retry_and_truncation_persisted(self):
        response = mock.Mock(status_code=200, headers={})
        response.json.return_value = {
            "choices": [{"message": {"content": "partial"},
                         "finish_reason": "length"}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 2,
                      "total_tokens": 10},
        }
        first = mock.Mock(status_code=429, headers={"Retry-After": "0"})
        first.json.return_value = {"error": {"message": "retry"}}
        attempts = []
        generator = QwenGenerator("secret", "https://api.test/v1", "model")
        with mock.patch("requests.post", side_effect=[first, response]), \
                mock.patch("time.sleep"):
            result = generator.generate_answer(
                "q", [], on_attempt=attempts.append, retry_delay=0)
        self.assertEqual(result["status"], "truncated")
        self.assertEqual(len(attempts), 2)
        self.assertIsNone(attempts[0]["total_tokens"])
        self.assertEqual(attempts[1]["total_tokens"], 10)

    def test_state_restart_and_full_run_totals(self):
        items = [{"question_key": key, "question_id": "Q001",
                  "doc_id": key} for key in ("document_a", "document_b")]
        state = ExperimentState(self.root, items)
        success = {
            "status": "success", "prediction": "answer",
            "prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10,
            "usage_complete": True, "latency_sec": 1,
        }
        state.record_attempt("document_a", success)
        state.record_attempt("document_b", {
            "status": "failed", "prediction": None, "total_tokens": None,
            "latency_sec": 2})
        state.close()
        state = ExperimentState(self.root, items)
        try:
            self.assertEqual(state.result("document_b")["status"], "failed")
            self.assertEqual(state.export()["known_total_tokens"], 10)
            state.record_attempt("document_b", success)
            self.assertEqual(state.export()["completed_questions"], 2)
            self.assertEqual(state.export()["known_total_tokens"], 20)
            # A torn text export is harmless: SQLite remains authoritative.
            (self.root / "predictions.jsonl").write_text('{"broken"')
            state.export()
            rows = (self.root / "predictions.jsonl").read_text().splitlines()
            self.assertEqual(len([json.loads(row) for row in rows]), 2)
        finally:
            state.close()

    def test_colpali_numeric_cache_equal_and_variable_lengths(self):
        retriever = ColPaliRetriever.__new__(ColPaliRetriever)
        for lengths in ((3, 3), (3, 5)):
            path = self.root / f"cache_{lengths[1]}.npz"
            np.savez(path, **{
                f"page_{i}": np.ones((length, 128), dtype=np.float32)
                for i, length in enumerate(lengths)})
            vectors = retriever.encode_document_pages(["a", "b"], path)
            self.assertEqual([len(v) for v in vectors], list(lengths))
            self.assertTrue(all(v.dtype == torch.float32 for v in vectors))
            score = retriever.score_maxsim(
                torch.ones((2, 128)), vectors[0])
            self.assertEqual(score, 256)

    def test_lock_rejects_second_writer(self):
        with run_lock(self.root / "run.lock"):
            with self.assertRaises(RuntimeError):
                with run_lock(self.root / "run.lock"):
                    pass

    def test_driver_resume_and_export_without_api(self):
        import run_eval
        from contextlib import redirect_stdout
        import io

        items = [{"question_key": key, "question_id": "Q001",
                  "doc_id": key, "document_key": key, "question": "question",
                  "dataset": "nation_standard_bench", "pdf_path": "fake.pdf"}
                 for key in ("document_a", "document_b")]
        settings = {"api_key": "test-key", "base_url": "https://api.test/v1",
                    "model_name": "test-model",
                    "api_format": "chat_completions",
                    "data_dir": self.root, "model_dir": self.root,
                    "artifact_dir": self.root}
        retriever = types.ModuleType("retriever_visrag")
        retriever.VisRAGRetriever = mock.Mock()
        retriever.VisRAGRetriever.return_value.retrieve.return_value = [
            (1, 0.9, "fake.png")]
        response = mock.Mock(status_code=200, headers={})
        response.json.return_value = {
            "choices": [{"message": {"content": "answer"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 2,
                      "total_tokens": 10}}
        argv = ["run_eval.py", "--dataset", "nation_standard_bench",
                "--retriever", "visrag"]
        with mock.patch.object(sys, "argv", argv), \
                mock.patch.object(run_eval.config, "load_settings",
                                  return_value=settings), \
                mock.patch.object(
                    run_eval, "load_dataset", return_value=items), \
                mock.patch.object(
                    run_eval, "model_identity", return_value="m"), \
                mock.patch.object(
                    run_eval, "file_digest", return_value="digest"), \
                mock.patch.dict(sys.modules, {"retriever_visrag": retriever}), \
                mock.patch("render_pdf.render_pdf_to_images",
                           return_value=["fake.png"]), \
                mock.patch("generate_qwen.encode_image_to_base64",
                           return_value="data:image/png;base64,AA=="), \
                mock.patch("requests.post", return_value=response) as post, \
                redirect_stdout(io.StringIO()):
            run_eval.main()
            run_eval.main()
            self.assertEqual(post.call_count, 2)
        summary_path = next(self.root.glob("runs/**/summary.json"))
        summary = json.loads(summary_path.read_text())
        self.assertEqual(summary["completed_questions"], 2)
        self.assertEqual(summary["known_total_tokens"], 20)
        self.assertEqual(summary["execution_sessions"], 2)

    def test_pdf_cache_tracks_dpi_and_recovers_bad_pages(self):
        import fitz
        from render_pdf import render_pdf_to_images

        pdf = self.root / "document.pdf"
        with fitz.open() as document:
            document.new_page()
            document.save(pdf)
        first = render_pdf_to_images(pdf, self.root / "pages", dpi=72)
        Path(first[0]).write_bytes(b"broken")
        repaired = render_pdf_to_images(pdf, self.root / "pages", dpi=72)
        with Image.open(repaired[0]) as image:
            image.verify()
        changed = render_pdf_to_images(pdf, self.root / "pages", dpi=100)
        self.assertNotEqual(first, changed)


if __name__ == "__main__":
    unittest.main()
