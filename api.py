"""
FastAPI application for Twitter Sentiment Analysis.
Production-ready endpoints for inference and comparison.
"""
from __future__ import annotations
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from models import SentimentModel
import nltk
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)


app = FastAPI(
    title="Twitter Sentiment API",
    description="API for comparing VADER, TextBlob, and RoBERTa sentiment models.",
    version="1.0"
)

sentiment_engine = SentimentModel()

# --- Pydantic Models for Validation ---
class SentimentRequest(BaseModel):
    """Data model for sentiment analysis requests."""
    text: str
    model: str = None

class SentimentResponse(BaseModel):
    """Data model for standard responses."""
    sentiment: str
    confidence: float
    model_used: str

class ComparisonResponse(BaseModel):
    """Data model for the battle comparison."""
    input_text: str
    results: Dict[str, Dict[str, Any]]

# --- Endpoints ---

@app.get("/")
def read_root() -> dict[str,str]:
    """Home endpoint to satisfy 'test_api_endpoints'."""
    return {"message": "Sentiment API is live"}

@app.get("/status")
def get_status() -> Dict[str, str]:
    """Health check: Returns the currently active main model."""
    return {
        "status": "online",
        "active_model": sentiment_engine.active_name
    }

@app.post("/predict", response_model=SentimentResponse)
def predict(request: SentimentRequest) -> SentimentResponse:
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
        
    if request.model and request.model != sentiment_engine.active_name:
        sentiment_engine.set_model(request.model)

    result = sentiment_engine.predict_detailed(request.text)

    return {
        "sentiment": result["sentiment"],
        "confidence": result["confidence"],
        "model_used": sentiment_engine.active_name  
    }
    
@app.post("/compare", response_model=ComparisonResponse)
def compare_all(request: SentimentRequest) -> ComparisonResponse:
    """Runs input against ALL models using the central engine."""
    text =request.text
    results = {}
    
    for model_name in ["vader", "textblob", "roberta"]:
        sentiment_engine.set_model(model_name)
        results[model_name] = sentiment_engine.predict_detailed(text)
    
    return {
        "input_text": text,
        "results": results     
    }
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)