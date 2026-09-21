"""Runs the real pinned model. Skipped unless SENTIMENT_INTEGRATION=1 (CI sets it; the model is a
268 MB download cached between runs)."""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("SENTIMENT_INTEGRATION") != "1", reason="set SENTIMENT_INTEGRATION=1"
)


def test_pinned_model_classifies_obvious_sentences() -> None:
    from sentiment_system import model

    preds = model.classify(
        ["it 's a charming and often affecting journey .", "unflinchingly bleak and desperate"]
    )
    assert [p.label for p in preds] == ["positive", "negative"]
    assert all(p.score > 0.9 for p in preds)
    assert model.classify([]) == []
    info = model.model_info()
    assert info["model"] == model.MODEL_ID and info["revision"] == model.MODEL_REVISION
