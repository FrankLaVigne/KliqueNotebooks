"""CPU-only local regression tests. These do not exercise a Klique snapshot."""
import argparse
from contextlib import redirect_stdout
import copy
import io
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest

import torch
import train_classifier as training


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_training(self, name, epochs=2, **kwargs):
        values = dict(run_dir=str(self.root / name), epochs=epochs, seed=42,
                      batch_size=64, lr=0.001, device="cpu", resume=False, evaluate_test=False)
        values.update(kwargs)
        with redirect_stdout(io.StringIO()):
            training.train(argparse.Namespace(**values))
        return torch.load(self.root / name / "checkpoint.pt", weights_only=True, map_location="cpu")

    def assert_nested_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for x, y in zip(left, right):
                self.assert_nested_equal(x, y)
        else:
            self.assertEqual(left, right)

    def test_cpu_resume_matches_uninterrupted(self):
        whole = self.run_training("whole", epochs=4, evaluate_test=True)
        first = self.run_training("split", epochs=2)
        resumed = self.run_training("split", epochs=4, resume=True, evaluate_test=True)
        self.assertEqual(first["last_epoch"], 2)
        self.assertEqual(resumed["last_epoch"], 4)
        self.assert_nested_equal(whole, resumed)
        self.assertGreater(whole["history"][-1]["val_accuracy"], 0.65)
        self.assertTrue((self.root / "split" / "test_metrics.json").is_file())

    def test_resume_in_a_new_python_process(self):
        whole = self.run_training("reference", epochs=3)
        script = str(Path(training.__file__).resolve())
        command = [sys.executable, script, "--run-dir", str(self.root / "process")]
        first = subprocess.run(command + ["--epochs", "1"], capture_output=True, text=True, timeout=60)
        self.assertEqual(first.returncode, 0, first.stderr)
        resumed = subprocess.run(command + ["--resume", "--epochs", "3"], capture_output=True, text=True, timeout=60)
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        self.assertIn("Resuming after epoch 1; next epoch 2", resumed.stdout)
        checkpoint = torch.load(self.root / "process" / "checkpoint.pt", map_location="cpu", weights_only=True)
        self.assert_nested_equal(whole, checkpoint)

    def test_fresh_run_does_not_overwrite(self):
        self.run_training("run")
        path = self.root / "run" / "checkpoint.pt"
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_training("run")
        self.assertEqual(path.read_bytes(), before)

    def test_noop_and_stale_config_rejected(self):
        self.run_training("run")
        for kwargs in ({"epochs": 2}, {"epochs": 4, "seed": 43}, {"epochs": 4, "lr": 0.01}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.run_training("run", resume=True, **kwargs)

    def test_malformed_and_incompatible_checkpoints_rejected(self):
        original = self.run_training("run")
        path = self.root / "run" / "checkpoint.pt"
        variants = []
        for key, value in (("schema", 999), ("last_epoch", -1), ("history", []),
                           ("model", {}), ("optimizer", {}), ("rng", {}),
                           ("versions", {}), ("dataset", {}), ("device", {})):
            broken = copy.deepcopy(original)
            broken[key] = value
            variants.append(broken)
        broken = copy.deepcopy(original)
        broken["optimizer"]["state"][0]["exp_avg"] = torch.zeros(1)
        variants.append(broken)
        for i, broken in enumerate(variants):
            torch.save(broken, path)
            with self.subTest(case=i), self.assertRaises(ValueError):
                self.run_training("run", epochs=3, resume=True)
        path.write_bytes(b"not a torch checkpoint")
        with self.assertRaises(ValueError):
            self.run_training("run", epochs=3, resume=True)

    def test_lock_prevents_two_writers(self):
        self.run_training("run")
        lock = self.root / "run" / ".training.lock"
        lock.write_text("another process")
        with self.assertRaises(ValueError):
            self.run_training("run", epochs=3, resume=True)
        self.assertTrue(lock.exists())

    def test_interrupted_atomic_write_preserves_previous_checkpoint(self):
        self.run_training("run")
        path = self.root / "run" / "checkpoint.pt"
        before = path.read_bytes()
        def fail(stream):
            stream.write(b"partial")
            raise OSError("simulated disk failure")
        with self.assertRaises(OSError):
            training.atomic_write(path, fail)
        self.assertEqual(path.read_bytes(), before)
        self.assertFalse(list(path.parent.glob(".writing-*")))


if __name__ == "__main__":
    unittest.main()
