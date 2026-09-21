# Case Study — SentimentAnalysis-System

**Repository:** [SentimentAnalysis-System](https://github.com/Freddricklogan/SentimentAnalysis-System) · **Evaluation report:** [freddricklogan.github.io/SentimentAnalysis-System](https://freddricklogan.github.io/SentimentAnalysis-System/) · **Author:** Freddrick Logan

---

## 1. Who has this problem

Teams that need sentiment scoring inside a product — support triage, review monitoring, course-feedback analysis — and the people who evaluate such a service before trusting it: an engineering lead asking how the model was measured, a procurement reviewer asking whether "AI" in the description means anything, and students learning how a machine-learning capability becomes a service with a contract.

## 2. The problem, as a scenario

A reviewer reads the README: deep learning, NLTK, spaCy, emotion detection, aspect-based analysis. She opens the code and finds sixteen words, `random.random()` added to every confidence, and a comment admitting that a real implementation would use a model. There is no evaluation, no baseline, no validation on the endpoint, no test. The earlier version of this repository was that: a claim about NLP with no NLP in it.

## 3. What it costs to leave it alone

Randomised confidence is worse than no confidence: it invites a downstream system to threshold on noise. A service with no measured accuracy cannot be compared with the free alternative, and the free alternative here — a lexicon — is not bad. An endpoint that accepts anything will eventually be sent a novel or an empty string. And a portfolio project that says "deep learning" over a word list is discovered in the first minute of a technical review.

## 4. The approach, and the alternative I rejected

I rejected training a model from scratch on the exercise's scale; a fine-tuned transformer that anyone can inspect and reproduce is the honest choice, and the engineering is in how it is served and measured. `model.py` loads DistilBERT fine-tuned on SST-2 at a pinned hub revision, lazily, on CPU. `lexicon.py` wraps VADER as the baseline every number is compared against. `aspects.py` cuts sentences at contrast conjunctions, finds configured aspect keywords, and scores a window of words around each with the same classifier, returning the window so the judgement is visible. `api.py` is FastAPI with Pydantic v2 models: blank and length validation, a batch limit, typed responses, and a classifier dependency so tests inject a double. `evaluate.py` and `report.py` score any classifier on the vendored SST-2 development set and render a page and JSON that CI publishes.

## 5. What the code does today

`sentiment-system serve` starts the API. `POST /analyze` takes one text and returns the transformer's label and score, VADER's label and score for comparison, every aspect window with its own label, and a per-aspect summary; `POST /analyze/batch` scores up to 64 texts; `/health` and `/model` report status and the pinned model. `sentiment-system report` evaluates both classifiers on 872 labelled sentences and writes a report with the Executive Shell — accuracy, precision, recall, F1, confusion counts, throughput, and the first eight sentences on which the two disagree — plus `report.json`. The Dockerfile installs CPU-only torch, downloads the model at build time and starts the service offline. The unit suite runs without the model; one integration test runs it when asked, which CI does with a cache.

## 6. Evidence

Fifteen tests at 98 % statement coverage cover the SST-2 loader and its rejections, the scoring arithmetic against hand counts and the length check, VADER above chance on the full set, disagreement listing, sentence and clause splitting, aspect windows and summaries with a custom aspect table, the API's health and model routes, the analyze and batch responses, every validation limit, a classifier that returns the wrong length, the report writer with an injected classifier, and the pinned model on two obvious sentences. On the 872-sentence development set with a 50.92 % majority class, DistilBERT reached accuracy 91.06 % and F1 0.9137 (413 true positives, 47 false positives, 381 true negatives, 31 false negatives) at 168 sentences per second on a laptop CPU; VADER reached 63.07 % and F1 0.7035. The live API returned a negative 0.9988 for a mixed review with four aspect windows and rejected a blank text with 422. The report rendered with zero console errors and no horizontal scroll at 1280 or 400 pixels. `AUDIT.md` records ten findings.

## 7. What it would take to run this in production

Host the container behind an authenticated gateway with request logging and rate limits; pin the model artefact in an internal registry rather than pulling from the hub; add a domain evaluation set — real support tickets or reviews, labelled — because SST-2 is movie sentences and the measured F1 does not transfer automatically; and monitor drift by re-scoring a labelled sample on a schedule. Emotion detection, if wanted, would be a second pinned model with its own harness, not a keyword list.

## 8. Limits and next steps

Binary sentiment only; no neutral class, which SST-2 does not have. Aspect windows are heuristic and English-only. The model is a general SST-2 fine-tune, so domain accuracy is unmeasured until a domain set exists. The service is not hosted; the published page is the evaluation. Next, in order: a neutral band from the score with a measured threshold, a domain evaluation set, and a small emotion model evaluated the same way.

## 9. Who should look at this

**Hiring manager:** evidence that I turn a machine-learning capability into a typed, tested, measured service and correct my own earlier overstatement.
**Consulting client:** a template for evaluating any text-classification vendor: baseline, labelled set, confusion counts, disagreements.
**Engineer:** read `src/sentiment_system/api.py` for the dependency-injected classifier and `evaluate.py` with `tests/test_core.py` for the harness.
