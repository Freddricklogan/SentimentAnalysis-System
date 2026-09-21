# AUDIT — SentimentAnalysis-System (pre-refactor)

Audit of the previous build: `sentiment_analyzer.py` (129 lines),
`app.py` (a 42-line Flask wrapper), a server template, a client-side
`index.html` with its own lexicon, and three images. The README
promised deep learning, NLTK and spaCy; the code imported `random`.

---

## A. Honesty of the copy

### A1 — "Deep learning models for NLP" that were eight words and a die
`sentiment_analyzer.py:20–21`: eight positive and eight negative words;
lines 30, 35 and 39: `confidence = 0.7 + random.random() * 0.25`. The
file's own header said "In a real implementation, this would use actual
ML models". The README did not. **Fix:** a fine-tuned DistilBERT
(`distilbert-base-uncased-finetuned-sst-2-english`, revision pinned)
behind the API, evaluated in CI on 872 labelled sentences with F1
reported, beside a VADER lexicon baseline so the gain is visible.

### A2 — "Emotion detection (joy, sadness, anger, fear, surprise)"
`detect_emotions` at line 84 matched keyword lists and, again, added
random noise. No emotion model ships now and the README no longer
claims one; the limits section says what it would take.

### A3 — "Aspect-based sentiment analysis"
`extract_aspects` found five nouns by substring and assigned a
sentiment by the same word counting. **Fix:** aspect windows — the
words around each configured keyword, cut at sentence and contrast
boundaries — scored by the same classifier as the whole text, with the
window returned so the reader can see what was scored. It is described
as simple, because it is.

### A4 — "Loading sentiment analysis models…" printed at import
Line 9 simulated a model load. **Fix:** the real model loads lazily on
first use, and `GET /model` reports which one.

## B. Method

### B1 — No evaluation
Nothing measured anything. **Fix:** `evaluate.py` computes accuracy,
precision, recall, F1, confusion counts and throughput for any
classifier on the vendored SST-2 development set; the report page and
`report.json` are produced by the CI run.

### B2 — No baseline
**Fix:** VADER is evaluated on the same sentences. On this run it
reaches F1 0.7035 against the transformer's 0.9137; the page shows the
sentences where they disagree.

### B3 — Contrast clauses coloured each other's aspects
Found while exercising the live API: "the screen is gorgeous but the
battery dies" scored the screen negative because the window crossed
"but". **Fix:** sentences are split at contrast conjunctions and
semicolons before windows are cut; a test pins it.

## C. Engineering

### C1 — Flask, untyped, no validation, no limits
`app.py` accepted any JSON and returned whatever the analyzer produced.
**Fix:** FastAPI with Pydantic v2 request and response models, blank
and length validation (2,000 characters, 64 texts per batch),
`/health` and `/model`, and a classifier dependency so tests inject a
double and the model is never downloaded to run the suite.

### C2 — No tests, no CI, no container, no pinned dependencies
**Fix:** 15 pytest tests (98 % statement coverage; the model module is
covered by an integration test that CI runs with the hub cache), ruff,
mypy strict, bandit, pip-audit, Trivy; a Dockerfile that bakes the
model in so the container starts offline; CPU-only torch through a
pinned index.

### C3 — Unsplash image in structured data; stock images committed
**Fix:** removed.
