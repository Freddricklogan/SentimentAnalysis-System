"""Shared types. A Classifier is any callable that scores a batch of texts; the transformer, the
lexicon baseline and the test double all satisfy it."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

from pydantic import BaseModel, Field

Label = Literal["positive", "negative"]


class Prediction(BaseModel):
    label: Label
    score: float = Field(ge=0.0, le=1.0, description="Probability of the predicted label")


Classifier = Callable[[Sequence[str]], list[Prediction]]
