import pytest
from fastapi.testclient import TestClient
from api import app, sentiment_engine
from models import SentimentModel
import nltk

client = TestClient(app)

@pytest.fixture(autouse=True)
def reset_engine():
    """Reset engine to known state before each test"""
    sentiment_engine.set_model("vader")  
    yield

@pytest.fixture
def engine():
    return SentimentModel()

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Sentiment API is live"}

def test_status():
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "active_model" in data

def test_404():
    response = client.get("/this-path-does-not-exist")
    assert response.status_code == 404

@pytest.mark.parametrize(
    "text, model, expected_sentiment",
    [
        ("I love this so much!", "vader", "positive"),
        ("This is the worst day ever.", "vader", "negative"),
        ("The weather is okay today.", "textblob", "positive"),   
        ("Absolutely terrible service.", "textblob", "negative"),
    ]
)
def test_predict_different_models_and_sentiments(text, model, expected_sentiment):
    payload = {"text": text, "model": model}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == expected_sentiment
    assert "confidence" in data
    assert data["model_used"] == model

def test_predict_default_model_when_no_model_given():
    payload = {"text": "Amazing product!"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["model_used"] == sentiment_engine.active_name


def test_predict_empty_text_rejected():
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 400
    assert "Text cannot be empty" in response.json()["detail"]


def test_predict_missing_text_field():
    response = client.post("/predict", json={"model": "vader"})
    assert response.status_code == 422  


def test_predict_invalid_model_fallback():
    prev_model = sentiment_engine.active_name
    payload = {"text": "test", "model": "random123"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["model_used"] == prev_model

@pytest.mark.parametrize(
    "text",
    [
        "I really enjoy this!",
        "This is complete garbage.",
        "It's 7:42 pm right now.",
    ]
)
def test_compare_endpoint(text):
    payload = {"text": text}
    response = client.post("/compare", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["input_text"] == text
    assert "results" in data
    assert len(data["results"]) >= 2   

    for model, res in data["results"].items():
        assert "sentiment" in res
        assert "confidence" in res
        assert res["sentiment"] in ["positive", "negative"]

def test_compare_when_active_is_vader():
    sentiment_engine.set_model("vader")
    response = client.post("/compare", json={"text": "Fantastic!"})
    data = response.json()
    assert "vader" in data["results"]
    assert "textblob" in data["results"]
    assert "roberta" not in data["results"]  

def test_model_init_default_vader():
    eng = SentimentModel()   
    assert eng.active_name in ["vader", "textblob", "roberta"]

    if eng.active_name == "vader":
        assert eng.model is not None  
    elif eng.active_name == "textblob":
        assert eng.model is None
    elif eng.active_name == "roberta":
        pass 


def test_set_model_invalid_keeps_previous(engine):
    prev = engine.active_name
    engine.set_model("invalid-model-name")
    assert engine.active_name == prev


@pytest.mark.parametrize("model_name", ["vader", "textblob", "roberta"])
def test_set_model_valid_changes_active(engine, model_name):
    engine.set_model(model_name)
    assert engine.active_name == model_name


def test_predict_detailed_vader_neutralish():
    engine = SentimentModel()
    engine.set_model("vader")
    res = engine.predict_detailed("Weather in Berlin today.")
    assert res["sentiment"] in ["positive", "negative"]


def test_textblob_zero_polarity_is_negative():
    engine = SentimentModel()
    engine.set_model("textblob")
    res = engine.predict_detailed("the")
    assert res["sentiment"] == "negative"   


def test_roberta_fallback_when_transformers_missing(engine, monkeypatch):
    def fake_import_error(name, *args, **kwargs):
        if name.startswith("transformers"):
            raise ImportError("Simulated: transformers not installed")
        return __import__(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import_error)

    engine.set_model("roberta")
    
    assert engine.model is None
    
    res = engine.predict_detailed("I like apples")
    
    assert res["sentiment"] == "positive"
    assert res["confidence"] == 50.0
    assert "note" in res
    assert "RoBERTa mock" in res["note"]


def test_roberta_mock_confidence_range():
    engine = SentimentModel()
    engine.set_model("roberta")
    res = engine.predict_detailed("test sentence")
    assert 0 <= res["confidence"] <= 100
    
def test_neutral_sentiment_coverage():
    """Forces the model to hit 'neutral' logic branches."""
    engine = SentimentModel()
    engine.set_model("vader")
    res = engine.predict_detailed("The table is made of wood.")
    assert "sentiment" in res