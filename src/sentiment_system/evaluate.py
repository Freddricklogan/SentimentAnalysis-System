"""Evaluation harness on the SST-2 development set (872 labelled sentences, GLUE distribution).
Computes accuracy, precision, recall and F1 for the positive class, the confusion counts and
throughput, for any Classifier."""

from __future__ import annotations

import csv
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .types import Classifier, Label, Prediction

DATA_SOURCE = (
    "SST-2 development set (872 sentences) from the GLUE distribution of the Stanford Sentiment "
    "Treebank (Socher et al., 2013); vendored in data/sst2_dev.tsv"
)


@dataclass(frozen=True)
class Example:
    text: str
    label: str  # "positive" | "negative"


@dataclass(frozen=True)
class Scores:
    name: str
    n: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    tn: int
    fn: int
    seconds: float
    per_second: float

    def as_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def load_sst2(path: Path, limit: int | None = None) -> list[Example]:
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        rows: list[Example] = []
        for row in reader:
            if row["label"] not in {"0", "1"}:
                msg = f"bad label {row['label']!r}"
                raise ValueError(msg)
            rows.append(
                Example(row["sentence"].strip(), "positive" if row["label"] == "1" else "negative")
            )
    if not rows:
        msg = "no rows"
        raise ValueError(msg)
    return rows[:limit] if limit else rows


def score(
    name: str, examples: Sequence[Example], preds: Sequence[Prediction], seconds: float
) -> Scores:
    if len(examples) != len(preds):
        msg = "examples and predictions differ in length"
        raise ValueError(msg)
    tp = sum(
        1
        for e, p in zip(examples, preds, strict=True)
        if e.label == "positive" and p.label == "positive"
    )
    fp = sum(
        1
        for e, p in zip(examples, preds, strict=True)
        if e.label == "negative" and p.label == "positive"
    )
    tn = sum(
        1
        for e, p in zip(examples, preds, strict=True)
        if e.label == "negative" and p.label == "negative"
    )
    fn = sum(
        1
        for e, p in zip(examples, preds, strict=True)
        if e.label == "positive" and p.label == "negative"
    )
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    n = len(examples)
    return Scores(
        name,
        n,
        (tp + tn) / n,
        precision,
        recall,
        f1,
        tp,
        fp,
        tn,
        fn,
        seconds,
        n / seconds if seconds > 0 else 0.0,
    )


def evaluate(name: str, classify: Classifier, examples: Sequence[Example]) -> Scores:
    t0 = time.perf_counter()
    preds = classify([e.text for e in examples])
    return score(name, examples, preds, time.perf_counter() - t0)


def disagreements(
    examples: Sequence[Example], a: Sequence[Prediction], b: Sequence[Prediction], limit: int = 8
) -> list[tuple[str, str, Label, Label]]:
    """Examples where the two classifiers disagree: (text, truth, a, b)."""
    out = [
        (e.text, e.label, pa.label, pb.label)
        for e, pa, pb in zip(examples, a, b, strict=True)
        if pa.label != pb.label
    ]
    return out[:limit]
