"""Aspect-based sentiment: find mentions of configured aspects, cut a window of words around each,
and score the window with whatever classifier the caller supplies. Deliberately simple."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from .types import Classifier, Prediction

DEFAULT_ASPECTS: dict[str, tuple[str, ...]] = {
    "price": ("price", "cost", "expensive", "cheap", "value", "pricing"),
    "quality": ("quality", "build", "durable", "sturdy", "flimsy"),
    "service": ("service", "support", "staff", "helpful", "rude"),
    "delivery": ("delivery", "shipping", "arrived", "late", "package"),
    "usability": ("easy", "difficult", "intuitive", "confusing", "interface", "setup"),
    "battery": ("battery", "charge", "charging"),
    "screen": ("screen", "display", "resolution"),
}
_WORD = re.compile(r"[A-Za-z']+")


@dataclass(frozen=True)
class AspectHit:
    aspect: str
    keyword: str
    window: str
    prediction: Prediction


_CONTRAST = re.compile(r"\s*(?:[;]|,?\s+(?:but|however|although|though|whereas|while)\s+)", re.I)


def split_sentences(text: str) -> list[str]:
    """Sentences, then clauses at contrast conjunctions, so "the screen is gorgeous but the battery
    dies" scores each half on its own rather than letting the second half colour the first."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    out: list[str] = []
    for p in parts:
        out.extend(c.strip() for c in _CONTRAST.split(p) if c and c.strip())
    return out


def windows(
    text: str, aspects: dict[str, tuple[str, ...]] | None = None, width: int = 6
) -> list[tuple[str, str, str]]:
    """(aspect, keyword, window) per keyword mention; the window is ±`width` words in-sentence."""
    table = aspects or DEFAULT_ASPECTS
    out: list[tuple[str, str, str]] = []
    for sentence in split_sentences(text):
        tokens = _WORD.findall(sentence)
        lowered = [t.lower() for t in tokens]
        for aspect, keys in table.items():
            for i, tok in enumerate(lowered):
                if tok in keys:
                    lo, hi = max(0, i - width), min(len(tokens), i + width + 1)
                    out.append((aspect, tok, " ".join(tokens[lo:hi])))
    return out


def analyse_aspects(
    text: str, classify: Classifier, aspects: dict[str, tuple[str, ...]] | None = None
) -> list[AspectHit]:
    found = windows(text, aspects)
    if not found:
        return []
    preds = classify([w for _, _, w in found])
    return [AspectHit(a, k, w, p) for (a, k, w), p in zip(found, preds, strict=True)]


def summarise(hits: Sequence[AspectHit]) -> dict[str, dict[str, float | int | str]]:
    """Per aspect: mentions, share positive, and the majority label."""
    out: dict[str, dict[str, float | int | str]] = {}
    for h in hits:
        entry = out.setdefault(h.aspect, {"mentions": 0, "positive": 0})
        entry["mentions"] = int(entry["mentions"]) + 1
        entry["positive"] = int(entry["positive"]) + (1 if h.prediction.label == "positive" else 0)
    for entry in out.values():
        m, p = int(entry["mentions"]), int(entry["positive"])
        entry["positive_share"] = p / m
        entry["label"] = "positive" if p * 2 >= m else "negative"
    return out
