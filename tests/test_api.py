from fastapi.testclient import TestClient

from sentiment_system.api import MAX_BATCH, MAX_CHARS, create_app

from .conftest import fake_classify

client = TestClient(create_app(classifier=fake_classify, model_name="fake"))


def test_health_and_model() -> None:
    assert client.get("/health").json() == {"status": "ok", "version": "1.0.0"}
    m = client.get("/model").json()
    assert m == {"model": "fake", "max_chars": MAX_CHARS, "max_batch": MAX_BATCH}


def test_analyze_returns_sentiment_lexicon_and_aspects() -> None:
    r = client.post("/analyze", json={"text": "Great price. The delivery was late."})
    assert r.status_code == 200
    body = r.json()
    assert body["sentiment"]["label"] in {"positive", "negative"}
    assert 0 <= body["sentiment"]["score"] <= 1
    assert body["lexicon"]["label"] in {"positive", "negative"}
    assert {a["aspect"] for a in body["aspects"]} == {"price", "delivery"}
    assert body["aspect_summary"]["delivery"]["label"] == "negative"
    assert body["model"] == "fake"
    r2 = client.post("/analyze", json={"text": "Great price.", "aspects": False})
    assert r2.json()["aspects"] == [] and r2.json()["aspect_summary"] == {}


def test_analyze_validation() -> None:
    assert client.post("/analyze", json={"text": ""}).status_code == 422
    assert client.post("/analyze", json={"text": "   "}).status_code == 422
    assert client.post("/analyze", json={"text": "x" * (MAX_CHARS + 1)}).status_code == 422
    assert client.post("/analyze", json={}).status_code == 422


def test_batch_endpoint_and_limits() -> None:
    r = client.post("/analyze/batch", json={"texts": ["great", "terrible", "meh"]})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 3 and [x["label"] for x in body["results"]] == [
        "positive",
        "negative",
        "positive",
    ]
    assert client.post("/analyze/batch", json={"texts": []}).status_code == 422
    assert client.post("/analyze/batch", json={"texts": ["ok", " "]}).status_code == 422
    assert client.post("/analyze/batch", json={"texts": ["x" * (MAX_CHARS + 1)]}).status_code == 422
    assert client.post("/analyze/batch", json={"texts": ["a"] * (MAX_BATCH + 1)}).status_code == 422


def test_bad_classifier_length_is_a_500() -> None:
    def broken(texts):  # type: ignore[no-untyped-def]
        return []

    c = TestClient(create_app(classifier=broken), raise_server_exceptions=False)
    assert c.post("/analyze/batch", json={"texts": ["a"]}).status_code == 500
