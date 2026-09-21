"""Evaluation report — the Pages artefact. Runs the harness on SST-2 dev for the transformer and the
lexicon baseline, and renders a static page with the Executive Shell plus report.json."""

from __future__ import annotations

import html
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from string import Template

from . import lexicon
from .evaluate import DATA_SOURCE, Example, Scores, disagreements, load_sst2, score
from .types import Classifier, Label, Prediction

PKG = Path(__file__).parent
SHELL_DIR = PKG / "shell"
TEMPLATES = PKG / "templates"


@dataclass(frozen=True)
class Report:
    examples: list[Example]
    transformer: Scores
    baseline: Scores
    conflicts: list[tuple[str, str, Label, Label]]
    model_name: str


def run(
    data: Path, classify: Classifier | None = None, limit: int | None = None, model_name: str = ""
) -> Report:
    examples = load_sst2(data, limit)
    if classify is None:
        from . import model

        classify = model.classify
        model_name = model_name or f"{model.MODEL_ID}@{model.MODEL_REVISION}"
    import time

    t0 = time.perf_counter()
    preds_t: list[Prediction] = classify([e.text for e in examples])
    t_secs = time.perf_counter() - t0
    t1 = time.perf_counter()
    preds_b = lexicon.classify([e.text for e in examples])
    b_secs = time.perf_counter() - t1
    return Report(
        examples=examples,
        transformer=score("transformer", examples, preds_t, t_secs),
        baseline=score("vader", examples, preds_b, b_secs),
        conflicts=disagreements(examples, preds_t, preds_b),
        model_name=model_name or "injected classifier",
    )


def _row(cells: list[str], head: bool = False) -> str:
    tag = "th" if head else "td"
    return "<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>"


def _scores_table(t: Scores, b: Scores) -> str:
    rows = [_row(["Metric", "DistilBERT (SST-2 fine-tune)", "VADER lexicon"], head=True)]
    for label, key in [
        ("Accuracy", "accuracy"),
        ("Precision (positive)", "precision"),
        ("Recall (positive)", "recall"),
        ("F1 (positive)", "f1"),
    ]:
        rows.append(_row([label, f"{getattr(t, key):.4f}", f"{getattr(b, key):.4f}"]))
    rows.append(
        _row(
            [
                "TP / FP / TN / FN",
                f"{t.tp} / {t.fp} / {t.tn} / {t.fn}",
                f"{b.tp} / {b.fp} / {b.tn} / {b.fn}",
            ]
        )
    )
    rows.append(_row(["Sentences per second (CPU)", f"{t.per_second:.0f}", f"{b.per_second:.0f}"]))
    return f"<table>{''.join(rows)}</table>"


def _conflicts_table(rows: list[tuple[str, str, Label, Label]]) -> str:
    out = [_row(["Sentence", "Truth", "DistilBERT", "VADER"], head=True)]
    out += [_row([html.escape(s), t, a, b]) for s, t, a, b in rows]
    return f"<table>{''.join(out)}</table>"


def kpi_json(r: Report) -> dict[str, object]:
    return {
        "n": r.transformer.n,
        "model": r.model_name,
        "source": DATA_SOURCE,
        "transformer": r.transformer.as_dict(),
        "baseline": r.baseline.as_dict(),
        "majority": max(
            sum(e.label == "positive" for e in r.examples),
            sum(e.label == "negative" for e in r.examples),
        )
        / len(r.examples),
    }


def render_html(r: Report, pages: str) -> str:
    tpl = Template((TEMPLATES / "page.html").read_text(encoding="utf-8"))
    return tpl.substitute(
        pages=html.escape(pages),
        source=html.escape(DATA_SOURCE),
        model=html.escape(r.model_name),
        n=str(r.transformer.n),
        t_f1=f"{r.transformer.f1:.4f}",
        b_f1=f"{r.baseline.f1:.4f}",
        t_acc=f"{r.transformer.accuracy * 100:.2f}%",
        b_acc=f"{r.baseline.accuracy * 100:.2f}%",
        scores_table=_scores_table(r.transformer, r.baseline),
        conflicts=_conflicts_table(r.conflicts),
        report_json=json.dumps(kpi_json(r)),
    )


def write_report(
    out: Path,
    data: Path,
    classify: Classifier | None = None,
    limit: int | None = None,
    model_name: str = "",
    pages: str = "https://freddricklogan.github.io/SentimentAnalysis-System/",
) -> Path:
    r = run(data, classify, limit, model_name)
    out.mkdir(parents=True, exist_ok=True)
    (out / "src").mkdir(exist_ok=True)
    shutil.copy(SHELL_DIR / "exec-shell.css", out / "src" / "exec-shell.css")
    shutil.copy(SHELL_DIR / "exec-shell.js", out / "src" / "exec-shell.js")
    shutil.copy(PKG / "report.js", out / "src" / "report.js")
    shutil.copy(TEMPLATES / "report.css", out / "src" / "report.css")
    (out / "index.html").write_text(render_html(r, pages), encoding="utf-8")
    (out / "report.json").write_text(json.dumps(kpi_json(r), indent=2), encoding="utf-8")
    return out / "index.html"
