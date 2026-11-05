"""Run: python -m unittest discover -s 05_Jupyter -p 'test_notebook_tools.py'."""
import json
from pathlib import Path
import tempfile
import unittest

from notebook_tools import Chunk, PublicIndex, bounded_context, extractive_answer, load_public_chunks, save_metrics, validate_metrics


class NotebookToolsTests(unittest.TestCase):
    def test_retrieval_relevance_and_abstention(self):
        index = PublicIndex([Chunk("README.md", 1, "Jupyter notebooks run Python code on managed compute."),
                             Chunk("05_Jupyter/README.md", 1, "Logistic regression classifies handwritten digit images.")])
        hits = index.search("handwritten digit images")
        self.assertEqual(hits[0][0].path, "05_Jupyter/README.md")
        self.assertIn("05_Jupyter/README.md#chunk-001", extractive_answer(hits))
        self.assertEqual(index.search("quantum zebrafish"), [])
        self.assertIn("could not find", extractive_answer([]))
        self.assertLessEqual(len(bounded_context(hits, max_chars=20)), 20)

    def test_allowlist_budgets_and_no_private_crawl(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "05_Jupyter").mkdir()
            for name in ("README.md", "05_Jupyter/README.md"):
                (root / name).write_text("public words " * 40)
            (root / "private.txt").write_text("PRIVATE_SENTINEL")
            chunks = load_public_chunks(root, chunk_words=20, overlap=5)
            self.assertTrue(all(len(c.text.split()) <= 20 for c in chunks))
            self.assertNotIn("PRIVATE_SENTINEL", " ".join(c.text for c in chunks))
            with self.assertRaises(ValueError):
                load_public_chunks(root, max_bytes=10)
            (root / "README.md").unlink()
            (root / "README.md").symlink_to(root / "private.txt")
            with self.assertRaises(ValueError):
                load_public_chunks(root)

    def test_safe_repeatable_metric_export(self):
        metrics = dict(accuracy=.9, macro_f1=.89, fit_seconds=.1, predict_seconds=.01, train_samples=100, test_samples=30)
        with tempfile.TemporaryDirectory() as directory:
            save_metrics(metrics, directory)
            save_metrics(metrics, directory)
            self.assertEqual(json.loads((Path(directory) / "metrics.json").read_text()), metrics)
            self.assertEqual(len((Path(directory) / "metrics.csv").read_text().splitlines()), 2)
        for bad in (dict(metrics, accuracy=float("nan")), dict(metrics, accuracy=1.1),
                    dict(metrics, train_samples=1.5), dict(metrics, token="secret")):
            with self.assertRaises(ValueError):
                validate_metrics(bad)


if __name__ == "__main__":
    unittest.main()
