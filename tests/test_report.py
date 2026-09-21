import json
from pathlib import Path

from sentiment_system.report import run, write_report

from .conftest import fake_classify

DATA = Path(__file__).resolve().parents[1] / "data" / "sst2_dev.tsv"


def test_write_report_with_injected_classifier(tmp_path: Path) -> None:
    out = write_report(
        tmp_path / "dist", DATA, classify=fake_classify, limit=100, model_name="fake"
    )
    assert out.exists()
    page = out.read_text(encoding="utf-8")
    assert "Content-Security-Policy" in page and "fake" in page and "onclick" not in page
    data = json.loads((tmp_path / "dist" / "report.json").read_text())
    assert data["n"] == 100 and data["model"] == "fake"
    assert 0 <= data["transformer"]["f1"] <= 1 and 0 <= data["baseline"]["f1"] <= 1
    assert 0.5 <= data["majority"] <= 1


def test_run_reports_conflicts() -> None:
    r = run(DATA, classify=fake_classify, limit=50, model_name="fake")
    assert r.transformer.n == 50 and r.baseline.n == 50
    assert len(r.conflicts) <= 8
    assert all(len(c) == 4 for c in r.conflicts)
