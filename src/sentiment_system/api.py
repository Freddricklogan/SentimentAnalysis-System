"""FastAPI service with Pydantic v2 request and response models. The classifier is a dependency so
tests inject a double and the real model loads only when the service starts."""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from . import __version__, lexicon
from .aspects import analyse_aspects, summarise
from .types import Classifier, Prediction

MAX_CHARS = 2000
MAX_BATCH = 64


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_CHARS)
    aspects: bool = True

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            msg = "text must not be blank"
            raise ValueError(msg)
        return v


class BatchRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=MAX_BATCH)

    @field_validator("texts")
    @classmethod
    def each_ok(cls, v: list[str]) -> list[str]:
        for i, t in enumerate(v):
            if not t.strip():
                msg = f"texts[{i}] must not be blank"
                raise ValueError(msg)
            if len(t) > MAX_CHARS:
                msg = f"texts[{i}] exceeds {MAX_CHARS} characters"
                raise ValueError(msg)
        return v


class AspectOut(BaseModel):
    aspect: str
    keyword: str
    window: str
    label: str
    score: float


class AnalyzeResponse(BaseModel):
    sentiment: Prediction
    lexicon: Prediction
    aspects: list[AspectOut]
    aspect_summary: dict[str, dict[str, float | int | str]]
    model: str


class BatchResponse(BaseModel):
    results: list[Prediction]
    count: int
    model: str


def default_classifier() -> Classifier:
    from . import model

    return model.classify


def create_app(classifier: Classifier | None = None, model_name: str | None = None) -> FastAPI:
    """Build the app. Pass a classifier to avoid loading the transformer (tests, offline demos)."""
    app = FastAPI(title="Sentiment Analysis System", version=__version__)
    chosen: Classifier | None = classifier
    name = model_name or "distilbert-base-uncased-finetuned-sst-2-english"

    def get_classifier() -> Classifier:
        nonlocal chosen
        if chosen is None:
            chosen = default_classifier()
        return chosen

    clf_dep = Depends(get_classifier)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/model")
    def model_route() -> dict[str, str | int]:
        return {"model": name, "max_chars": MAX_CHARS, "max_batch": MAX_BATCH}

    @app.post("/analyze", response_model=AnalyzeResponse)
    def analyze(req: AnalyzeRequest, clf: Classifier = clf_dep) -> AnalyzeResponse:
        sentiment = clf([req.text])[0]
        lex = lexicon.classify([req.text])[0]
        hits = analyse_aspects(req.text, clf) if req.aspects else []
        return AnalyzeResponse(
            sentiment=sentiment,
            lexicon=lex,
            aspects=[
                AspectOut(
                    aspect=h.aspect,
                    keyword=h.keyword,
                    window=h.window,
                    label=h.prediction.label,
                    score=h.prediction.score,
                )
                for h in hits
            ],
            aspect_summary=summarise(hits),
            model=name,
        )

    @app.post("/analyze/batch", response_model=BatchResponse)
    def analyze_batch(req: BatchRequest, clf: Classifier = clf_dep) -> BatchResponse:
        results = clf(req.texts)
        if len(results) != len(req.texts):
            raise HTTPException(
                status_code=500, detail="classifier returned the wrong number of results"
            )
        return BatchResponse(results=results, count=len(results), model=name)

    return app
