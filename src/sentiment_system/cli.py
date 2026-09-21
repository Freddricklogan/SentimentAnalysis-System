"""Command-line entry point: `sentiment-system report --out dist` and `sentiment-system serve`."""

from __future__ import annotations

from pathlib import Path

import typer

from .report import write_report

app = typer.Typer(add_completion=False, help="Sentiment analysis service.")
DEFAULT_DATA = Path("data") / "sst2_dev.tsv"


@app.callback()
def main() -> None:
    """Sentiment analysis service."""


@app.command()
def report(
    out: Path = typer.Option(Path("dist"), help="Output directory for the static report."),
    data: Path = typer.Option(DEFAULT_DATA, help="SST-2 dev TSV."),
    limit: int | None = typer.Option(None, help="Evaluate only the first N sentences."),
) -> None:
    """Evaluate the transformer and the lexicon baseline on SST-2 dev and write the report."""
    path = write_report(out, data, limit=limit)
    print(f"wrote {path}")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the API with uvicorn (loads the model on first request)."""
    import uvicorn

    from .api import create_app

    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    app()
