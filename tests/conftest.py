from collections.abc import Sequence

import pytest

from sentiment_system.types import Label, Prediction

POS = {"good", "great", "charming", "love", "excellent", "helpful", "easy", "fast", "affecting"}
NEG = {
    "bad",
    "bleak",
    "terrible",
    "hate",
    "awful",
    "rude",
    "late",
    "confusing",
    "flimsy",
    "desperate",
}


def fake_classify(texts: Sequence[str]) -> list[Prediction]:
    """Deterministic keyword double standing in for the transformer."""
    out: list[Prediction] = []
    for t in texts:
        words = set(t.lower().replace(".", " ").replace(",", " ").split())
        p, n = len(words & POS), len(words & NEG)
        label: Label = "positive" if p >= n else "negative"
        out.append(Prediction(label=label, score=min(1.0, 0.6 + 0.1 * abs(p - n))))
    return out


@pytest.fixture
def fake() -> object:
    return fake_classify
