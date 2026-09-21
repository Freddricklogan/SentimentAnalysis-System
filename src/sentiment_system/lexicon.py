"""VADER lexicon baseline: no learning, no download, a floor the transformer must beat."""

from __future__ import annotations

from collections.abc import Sequence

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .types import Label, Prediction

_analyzer = SentimentIntensityAnalyzer()


def classify(texts: Sequence[str]) -> list[Prediction]:
    """Positive when VADER's compound score is >= 0 (the convention used for binary SST-2)."""
    out: list[Prediction] = []
    for t in texts:
        compound = float(_analyzer.polarity_scores(t)["compound"])
        label: Label = "positive" if compound >= 0 else "negative"
        # Map |compound| in [0, 1] to a probability-like score in [0.5, 1].
        out.append(Prediction(label=label, score=0.5 + abs(compound) / 2))
    return out
