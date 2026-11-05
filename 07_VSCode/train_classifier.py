#!/usr/bin/env python3
"""Tiny offline digits classifier with validated, epoch-boundary resumability.

Run from repository root. Checkpoints contain tensors and primitive values only;
never disable weights_only or load an untrusted checkpoint. No network data fetch.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import tempfile

import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import torch
from torch import nn

SCHEMA = 1


def versions():
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": np.__version__, "scikit_learn": sklearn.__version__}


def dataset(seed):
    digits = load_digits()
    x = np.asarray(digits.data / 16.0, dtype=np.float32)
    y = np.asarray(digits.target, dtype=np.int64)
    train, held = train_test_split(np.arange(len(y)), test_size=0.30,
                                  random_state=seed, stratify=y)
    val, test = train_test_split(held, test_size=0.50,
                                random_state=seed, stratify=y[held])
    digest = hashlib.sha256()
    for array in (x, y, train, val, test):
        digest.update(array.tobytes())
    splits = {name: (torch.from_numpy(x[idx]), torch.from_numpy(y[idx]))
              for name, idx in (("train", train), ("val", val), ("test", test))}
    return splits, {"name": "sklearn.load_digits", "sha256": digest.hexdigest(),
                    "sizes": {name: len(pair[1]) for name, pair in splits.items()}}


def model():
    return nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 10))


def atomic_write(path, writer):
    """Same-directory replace: interruption leaves either prior or complete new file."""
    fd, name = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        # Linux workspace durability; storage snapshot completion is separate.
        if hasattr(os, "O_DIRECTORY"):
            directory = os.open(path.parent, os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_json(path, value):
    atomic_write(path, lambda f: f.write((json.dumps(value, indent=2) + "\n").encode()))


@contextmanager
def run_lock(directory):
    lock = directory / ".training.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ValueError(f"Run is locked: {lock}. Check for an active trainer before removing a stale lock.") from exc
    try:
        with os.fdopen(fd, "w") as f:
            f.write(f"pid={os.getpid()} host={platform.node()}\n")
        yield
    finally:
        lock.unlink()


def load_checkpoint(path, config, identity, device):
    # CPU map avoids materializing arbitrary GPU allocations and keeps RNG tensors CPU.
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError("Checkpoint cannot be safely loaded; do not bypass weights_only.") from exc
    required = {"schema", "config", "dataset", "versions", "device", "last_epoch",
                "model", "optimizer", "rng", "history"}
    if not isinstance(ckpt, dict) or set(ckpt) != required or ckpt["schema"] != SCHEMA:
        raise ValueError("Unsupported or malformed checkpoint schema.")
    for key, expected in (("config", config), ("dataset", identity), ("versions", versions()),
                          ("device", device)):
        if ckpt[key] != expected:
            raise ValueError(f"Checkpoint {key} mismatch; restore the recorded environment/configuration or start a new run.")
    epoch = ckpt["last_epoch"]
    if type(epoch) is not int or epoch < 1 or not isinstance(ckpt["history"], list) or len(ckpt["history"]) != epoch:
        raise ValueError("Invalid checkpoint epoch/history.")
    for i, row in enumerate(ckpt["history"], 1):
        if not isinstance(row, dict) or set(row) != {"epoch", "train_loss", "val_loss", "val_accuracy"} or row["epoch"] != i:
            raise ValueError("Invalid metric history schema.")
        if any(type(row[k]) not in (float, int) or not math.isfinite(row[k]) for k in ("train_loss", "val_loss", "val_accuracy")):
            raise ValueError("Invalid metric values.")
    expected = model().state_dict()
    state = ckpt["model"]
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError("Invalid model state keys.")
    for k, reference in expected.items():
        if not isinstance(state[k], torch.Tensor) or state[k].shape != reference.shape or state[k].dtype != reference.dtype or not torch.isfinite(state[k]).all():
            raise ValueError(f"Invalid model tensor: {k}")
    opt = ckpt["optimizer"]
    if not isinstance(opt, dict) or set(opt) != {"state", "param_groups"} or not isinstance(opt["state"], dict) or not isinstance(opt["param_groups"], list):
        raise ValueError("Invalid optimizer schema.")
    reference_optimizer = torch.optim.Adam(model().parameters(), lr=config["lr"]).state_dict()
    if opt["param_groups"] != reference_optimizer["param_groups"]:
        raise ValueError("Optimizer parameter groups/configuration mismatch.")
    if set(opt["state"]) != set(range(len(expected))):
        raise ValueError("Invalid optimizer parameter IDs.")
    steps = epoch * math.ceil(identity["sizes"]["train"] / config["batch_size"])
    for index, reference in enumerate(expected.values()):
        item = opt["state"][index]
        if not isinstance(item, dict) or set(item) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("Invalid Adam state schema.")
        if not isinstance(item["step"], torch.Tensor) or item["step"].numel() != 1 or item["step"].item() != steps:
            raise ValueError("Optimizer step does not match checkpoint epoch.")
        for key in ("exp_avg", "exp_avg_sq"):
            value = item[key]
            if not isinstance(value, torch.Tensor) or value.shape != reference.shape or value.dtype != reference.dtype or not torch.isfinite(value).all():
                raise ValueError("Invalid optimizer tensor.")
    rng = ckpt["rng"]
    if not isinstance(rng, dict) or set(rng) != {"python", "torch", "cuda"}:
        raise ValueError("Invalid RNG schema.")
    for key in ("torch", "cuda"):
        value = rng[key]
        if key == "cuda" and device["type"] == "cpu" and value is None:
            continue
        if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1:
            raise ValueError("Invalid RNG tensor.")
    try:
        random.Random().setstate(rng["python"])
        torch.Generator().set_state(rng["torch"])
    except Exception as exc:
        raise ValueError("Invalid RNG state.") from exc
    return ckpt


@torch.no_grad()
def evaluate(net, split, device):
    net.eval()
    x, y = (t.to(device) for t in split)
    logits = net(x)
    return {"loss": nn.functional.cross_entropy(logits, y).item(),
            "accuracy": (logits.argmax(1) == y).float().mean().item()}


def train(args):
    if args.epochs < 1 or args.batch_size < 1 or not math.isfinite(args.lr) or args.lr <= 0 or not 0 <= args.seed < 2**32:
        raise ValueError("Use positive epochs/batch-size/lr and a seed in [0, 2**32).")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA/ROCm device unavailable; use CPU or an approved compatible image.")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
            raise ValueError("For GPU determinism set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting Python.")
        torch.backends.cudnn.benchmark = False
    device = {"type": args.device, "name": torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu",
              "cuda": torch.version.cuda, "hip": torch.version.hip}
    config = {"seed": args.seed, "batch_size": args.batch_size, "lr": args.lr,
              "architecture": "64-64-relu-dropout0.1-10", "optimizer": "Adam", "threads": 1}
    splits, identity = dataset(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    net = model().to(args.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    directory = Path(args.run_dir)
    if not args.resume:
        # A deliberate fresh run uses a NEW directory, never a destructive overwrite flag.
        directory.mkdir(parents=True, exist_ok=False)
    elif not directory.is_dir():
        raise ValueError("Resume directory does not exist.")
    with run_lock(directory):
        history = []
        last_epoch = 0
        if args.resume:
            ckpt = load_checkpoint(directory / "checkpoint.pt", config, identity, device)
            last_epoch, history = ckpt["last_epoch"], ckpt["history"]
            if args.epochs <= last_epoch:
                raise ValueError(f"Target epochs must exceed saved epoch {last_epoch}; --epochs is the total target, not extra epochs.")
            net.load_state_dict(ckpt["model"], strict=True)
            # Adam casts optimizer tensors to parameter devices when restoring state.
            optimizer.load_state_dict(ckpt["optimizer"])
            random.setstate(ckpt["rng"]["python"])
            torch.set_rng_state(ckpt["rng"]["torch"])
            if args.device == "cuda":
                torch.cuda.set_rng_state(ckpt["rng"]["cuda"], device=0)
            print(f"Resuming after epoch {last_epoch}; next epoch {last_epoch + 1}", flush=True)
        x, y = splits["train"]
        for epoch in range(last_epoch + 1, args.epochs + 1):
            net.train()
            order = torch.randperm(len(y))
            total_loss = 0.0
            for indices in order.split(args.batch_size):
                xb, yb = x[indices].to(args.device), y[indices].to(args.device)
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(net(xb), yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(indices)
            val = evaluate(net, splits["val"], args.device)
            row = {"epoch": epoch, "train_loss": total_loss / len(y),
                   "val_loss": val["loss"], "val_accuracy": val["accuracy"]}
            history.append(row)
            ckpt = {"schema": SCHEMA, "config": config, "dataset": identity, "versions": versions(),
                    "device": device, "last_epoch": epoch, "model": net.state_dict(),
                    "optimizer": optimizer.state_dict(), "history": history,
                    "rng": {"python": random.getstate(), "torch": torch.get_rng_state(),
                            "cuda": torch.cuda.get_rng_state(0) if args.device == "cuda" else None}}
            atomic_write(directory / "checkpoint.pt", lambda f: torch.save(ckpt, f))
            # Checkpoint is authoritative if interrupted between these two file writes.
            write_json(directory / "metrics.json", {k: ckpt[k] for k in
                       ("schema", "config", "dataset", "versions", "device", "last_epoch", "history")})
            print(json.dumps(row), flush=True)
        if args.evaluate_test:
            test = {"epoch": args.epochs, **evaluate(net, splits["test"], args.device)}
            write_json(directory / "test_metrics.json", test)
            print("Held-out test (do not tune on this):", json.dumps(test), flush=True)
        print(f"Saved {directory / 'checkpoint.pt'} at epoch {args.epochs}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="07_VSCode/lab_outputs/digits-run")
    parser.add_argument("--epochs", type=int, default=3, help="Total completed epochs desired")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true", help="Evaluate held-out test at final target only")
    args = parser.parse_args(argv)
    try:
        train(args)
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        parser.exit(2, f"Training stopped: {exc}\n")


if __name__ == "__main__":
    main()
