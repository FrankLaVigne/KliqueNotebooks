"""Offline, bounded retrieval and report helpers for Jupyter labs 05h and 05j."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

PUBLIC_DOCUMENTS = ("README.md", "05_Jupyter/README.md")


@dataclass(frozen=True)
class Chunk:
    path: str
    chunk_id: int
    text: str

    @property
    def citation(self):
        return f"{self.path}#chunk-{self.chunk_id:03d}"


def load_public_chunks(root, *, chunk_words=140, overlap=25, max_bytes=150_000):
    """Read only two reviewed public files, never crawl or follow symlinks.

    Chunk IDs are deterministic for the same file bytes and chunk settings,
    not stable identifiers across document edits.
    """
    if not 20 <= chunk_words <= 250 or not 0 <= overlap < chunk_words:
        raise ValueError("Use 20..250 words and an overlap smaller than the chunk.")
    if not isinstance(max_bytes, int) or not 1 <= max_bytes <= 500_000:
        raise ValueError("max_bytes must be between 1 and 500000")
    root = Path(root).resolve()
    chunks = []
    for relative in PUBLIC_DOCUMENTS:
        path = root / relative
        if any(parent.is_symlink() for parent in [path, *path.parents] if parent != root):
            raise ValueError(f"Symlink not allowed: {relative}")
        if not path.is_file():
            raise FileNotFoundError(f"Missing public source: {relative}. Use the complete repository checkout.")
        with path.open("rb") as handle:
            raw = handle.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise ValueError(f"Public source exceeds byte budget: {relative}")
        words = raw.decode("utf-8").split()
        for i, start in enumerate(range(0, len(words), chunk_words - overlap), 1):
            chunks.append(Chunk(relative, i, " ".join(words[start:start + chunk_words])))
    if not chunks:
        raise ValueError("The public sources are empty.")
    return chunks


class PublicIndex:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=12_000)
        self.matrix = self.vectorizer.fit_transform(c.text for c in self.chunks)

    def search(self, query, *, top_k=3, min_score=0.10):
        if not isinstance(query, str) or not query.strip() or len(query) > 500:
            raise ValueError("Use a nonempty query of at most 500 characters.")
        if not 1 <= top_k <= 5 or not math.isfinite(min_score) or not 0 <= min_score <= 1:
            raise ValueError("Use 1..5 results and a score threshold in [0, 1].")
        scores = (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [(self.chunks[int(i)], float(scores[i])) for i in order if scores[i] > 0 and scores[i] >= min_score]


def extractive_answer(hits):
    """Return source excerpts verbatim, not generated claims or executed instructions."""
    if not hits:
        return "I could not find relevant evidence in the two allowlisted public documents."
    return "Retrieved excerpts (not an LLM-generated answer):\n\n" + "\n\n".join(
        f"[{chunk.citation}] (similarity {score:.3f})\n{chunk.text}" for chunk, score in hits
    )


def bounded_context(hits, *, max_chars=5000):
    if not 1 <= max_chars <= 6000:
        raise ValueError("Context budget must be 1..6000 characters.")
    result = "\n\n".join(f"[{chunk.citation}]\n{chunk.text}" for chunk, _ in hits)
    return result[:max_chars]


def validate_metrics(metrics):
    """Allowlist numeric metrics only; reject NaN, secrets, arbitrary objects."""
    allowed = {"accuracy", "macro_f1", "fit_seconds", "predict_seconds", "train_samples", "test_samples"}
    if set(metrics) != allowed:
        raise ValueError("Metrics must contain exactly the documented numeric fields.")
    result = {}
    for key, value in metrics.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"Metric {key} must be numeric")
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"Metric {key} must be finite and nonnegative")
        if key in {"accuracy", "macro_f1"} and number > 1:
            raise ValueError(f"Metric {key} must be in [0, 1]")
        if key.endswith("samples"):
            if number < 1 or not number.is_integer():
                raise ValueError("Sample counts must be positive integers")
            result[key] = int(number)
        else:
            result[key] = number
    return result


def save_metrics(metrics, output_dir):
    """Overwrite just the two named outputs, allowing safe repeat execution."""
    result = validate_metrics(metrics)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("metrics.json", "metrics.csv"):
        if (output / name).is_symlink():
            raise ValueError("Output files must not be symlinks")
    (output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    keys = sorted(result)
    (output / "metrics.csv").write_text(
        ",".join(keys) + "\n" + ",".join(str(result[k]) for k in keys) + "\n", encoding="utf-8"
    )
    return result
