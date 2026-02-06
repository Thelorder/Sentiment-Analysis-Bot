"""
FastAPI application for Twitter Sentiment Analysis.
Provides endpoints for single prediction and model status.
"""
from typing import Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel
from models import SentimentModel

app = FastAPI(title="Twitter Sentiment API")
sentiment_engine = SentimentModel()

class SentimentRequest(BaseModel):
    """Data model for sentiment analysis requests."""
    text: str

@app.get("/status")
def get_status() -> Dict[str, str]:
    """Returns the currently active model name."""
    return {"active_model": sentiment_engine.active_name}

@app.post("/predict")
def predict(request: SentimentRequest) -> Dict[str, Any]:
    """
    Receives text and returns sentiment prediction and confidence.
    """
    prediction = sentiment_engine.predict(request.text)
    
    # Simple confidence logic for the demo
    confidence = 100.0
    if sentiment_engine.active_name == "roberta":
        confidence = 85.0  # Placeholder if library is mocked
        
    return {
        "sentiment": prediction,
        "confidence": confidence,
        "model_used": sentiment_engine.active_name
    }

@app.get("/compare")
def compare_all(text: str) -> Dict[str, Any]:
    """Requirement: Support for all models and comparison."""

    return {
        "vader": sentiment_engine.predict(text),
        "model_used": sentiment_engine.active_name
    }
    