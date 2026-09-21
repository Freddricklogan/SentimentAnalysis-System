from pathlib import Path

import pytest

from sentiment_system import lexicon
from sentiment_system.aspects import analyse_aspects, split_sentences, summarise, windows
from sentiment_system.evaluate import (
    Example,
    disagreements,
    evaluate,
    load_sst2,
    score,
)
from sentiment_system.types import Prediction

from .conftest import fake_classify

DATA = Path(__file__).resolve().parents[1] / "data" / "sst2_dev.tsv"


def test_sst2_loads_872_balanced_rows() -> None:
    rows = load_sst2(DATA)
    assert len(rows) == 872
    assert sum(r.label == "positive" for r in rows) == 444
    assert rows[0].text.startswith("it 's a charming")
    assert len(load_sst2(DATA, limit=10)) == 10


def test_sst2_rejects_bad_labels(tmp_path: Path) -> None:
    p = tmp_path / "bad.tsv"
    p.write_text("sentence\tlabel\nx\t2\n")
    with pytest.raises(ValueError, match="bad label"):
        load_sst2(p)
    q = tmp_path / "empty.tsv"
    q.write_text("sentence\tlabel\n")
    with pytest.raises(ValueError, match="no rows"):
        load_sst2(q)


def test_score_arithmetic_and_length_check() -> None:
    ex = [
        Example("a", "positive"),
        Example("b", "positive"),
        Example("c", "negative"),
        Example("d", "negative"),
    ]
    preds = [
        Prediction(label="positive", score=0.9),
        Prediction(label="negative", score=0.9),
        Prediction(label="positive", score=0.6),
        Prediction(label="negative", score=0.7),
    ]
    s = score("t", ex, preds, seconds=2.0)
    assert (s.tp, s.fp, s.tn, s.fn) == (1, 1, 1, 1)
    assert s.accuracy == 0.5 and s.precision == 0.5 and s.recall == 0.5 and s.f1 == 0.5
    assert s.per_second == 2.0
    assert s.as_dict()["name"] == "t"
    with pytest.raises(ValueError, match="differ in length"):
        score("t", ex, preds[:2], 1.0)
    zero = score("z", ex[2:], preds[1:2] + preds[3:4], 1.0)
    assert zero.precision == 0.0 and zero.f1 == 0.0


def test_lexicon_baseline_scores_above_chance_on_sst2() -> None:
    rows = load_sst2(DATA)
    s = evaluate("vader", lexicon.classify, rows)
    assert s.n == 872
    assert s.accuracy > 0.6  # VADER on SST-2 dev is well above the 50.9% majority rate
    assert 0.5 <= lexicon.classify(["I love this"])[0].score <= 1.0
    assert lexicon.classify(["I hate this"])[0].label == "negative"


def test_disagreements_lists_only_conflicts() -> None:
    ex = [Example("x", "positive"), Example("y", "negative")]
    a = [Prediction(label="positive", score=0.9), Prediction(label="positive", score=0.6)]
    b = [Prediction(label="positive", score=0.8), Prediction(label="negative", score=0.7)]
    d = disagreements(ex, a, b)
    assert d == [("y", "negative", "positive", "negative")]


def test_aspect_windows_and_summary() -> None:
    text = "The price was great. The delivery was late. Support staff were helpful!"
    assert split_sentences(text) == [
        "The price was great.",
        "The delivery was late.",
        "Support staff were helpful!",
    ]
    w = windows(text)
    aspects = sorted({a for a, _, _ in w})
    assert aspects == ["delivery", "price", "service"]
    hits = analyse_aspects(text, fake_classify)
    by = {h.aspect: h.prediction.label for h in hits}
    assert (
        by["price"] == "positive" and by["delivery"] == "negative" and by["service"] == "positive"
    )
    summary = summarise(hits)
    assert summary["price"]["label"] == "positive" and summary["price"]["mentions"] == 1
    assert summarise([]) == {}
    assert analyse_aspects("Nothing relevant here.", fake_classify) == []
    custom = windows("The lens is sharp.", {"lens": ("lens",)}, width=1)
    assert custom == [("lens", "lens", "The lens is")]


def test_contrast_clauses_are_scored_separately() -> None:
    text = "The screen is gorgeous, but the battery dies by noon; support was rude however we coped"
    assert split_sentences(text) == [
        "The screen is gorgeous",
        "the battery dies by noon",
        "support was rude",
        "we coped",
    ]
    w = {a: win for a, _, win in windows(text)}
    assert "battery" not in w["screen"] and "screen" not in w["battery"]
