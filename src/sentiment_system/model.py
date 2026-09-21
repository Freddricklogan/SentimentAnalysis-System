"""DistilBERT fine-tuned on SST-2, loaded lazily from the Hugging Face hub (cached after the first
download). The model id and revision are pinned so the report's numbers refer to one artefact."""

from __future__ import annotations

import os
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

from .types import Prediction

MODEL_ID = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
MODEL_REVISION = "714eb0f"  # commit on the hub; pinning makes the evaluation reproducible
MAX_TOKENS = 512
DEFAULT_BATCH = 32


@lru_cache(maxsize=1)
def _pipeline() -> Any:
    from transformers import pipeline

    return pipeline(
        "text-classification",
        model=MODEL_ID,
        revision=MODEL_REVISION,
        device=-1,
        truncation=True,
        max_length=MAX_TOKENS,
    )


def classify(texts: Sequence[str], batch_size: int = DEFAULT_BATCH) -> list[Prediction]:
    """Score texts in batches; the pipeline's POSITIVE/NEGATIVE labels are lower-cased."""
    if not texts:
        return []
    raw = _pipeline()(list(texts), batch_size=batch_size)
    return [Prediction(label=r["label"].lower(), score=float(r["score"])) for r in raw]


def model_info() -> dict[str, str]:
    return {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "device": "cpu",
        "cache": os.environ.get("HF_HOME", "default"),
    }
