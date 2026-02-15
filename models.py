"""
Logic Module for Sentiment Analysis Models.
"""
import os
import re
from typing import Dict, Any
import yaml
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk 
from utils import clean_tweet

# Set environment variables
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class SentimentModel:
    """Manager class for various sentiment analysis pre-trained models."""

    def __init__(self, config_path: str = None) -> None:
        """Initializes the model based on the configuration file."""
        if config_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_dir, "config.yaml")

        self.config: Dict[str, Any] = self.load_config(path=config_path)
        self.active_name: str = self.config['api_config']['active_model']
        self.model: Any = None
        self.threshold: float = 0.0
        
        self.initialize_model(self.active_name)

    def load_config(self, path: str) -> Dict[str, Any]:
        """Loads the YAML configuration file."""
        try:
            with open(path, "r", encoding="utf-8") as file:
                return yaml.safe_load(file)
        except (FileNotFoundError, KeyError):
            return {
                "api_config": {"active_model": "vader"},
                "model_config": {
                    "vader": {"threshold": 0.05},
                    "textblob": {"polarity_threshold": 0.0},
                    "roberta": {"model_name": "cardiffnlp/twitter-roberta-base-sentiment"}
                }
            }

    def set_model(self, model_name: str) -> None:
        """
        Explicitly switches the active model and re-initializes the engine.
        CRITICAL: This ensures we aren't using RoBERTa when we want VADER.
        """
        if model_name not in ["vader", "textblob", "roberta"]:
            print(f"Warning: {model_name} is not valid. Keeping {self.active_name}")
            return
            
        self.active_name = model_name
        self.initialize_model(model_name)

    def initialize_model(self, model_name: str) -> None:
        """Sets up the specific model engine and thresholds."""
        if model_name == "vader":
            nltk.download('vader_lexicon', quiet=True)
            
            self.model = SentimentIntensityAnalyzer()
            self.threshold = self.config['model_config']['vader']['threshold']
        elif model_name == "textblob":
            self.model = None 
            self.threshold = self.config['model_config']['textblob']['polarity_threshold']
        elif model_name == "roberta":
            try:
                from transformers import pipeline
                model_id = self.config['model_config']['roberta']['model_name']
                self.model = pipeline("sentiment-analysis", model=model_id, framework="pt")
            except ImportError:
                print("Warning: Transformers not installed. RoBERTa unavailable.")
                self.model = None

    def predict(self, text: str) -> str:
        """Legacy wrapper for simple string return."""
        return self.predict_detailed(text)["sentiment"]

    def predict_detailed(self, text: str) -> Dict[str, Any]:
        """Returns label, confidence score, and raw output."""
        
        cleaned_text = clean_tweet(text)
        
        # --- VADER Logic ---
        if self.active_name == "vader":
            if not self.model: 
                self.model = SentimentIntensityAnalyzer()
            scores = self.model.polarity_scores(cleaned_text)
            #scores = self.model.polarity_scores(text)
            compound = scores['compound']
            label = "positive" if compound >= self.threshold else "negative"
            return {
                "sentiment": label, 
                "confidence": round(abs(compound) * 100, 2)
            }

        # --- TextBlob Logic ---
        if self.active_name == "textblob":
            blob = TextBlob(cleaned_text)
            #blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            label = "positive" if polarity > self.threshold else "negative"
            return {
                "sentiment": label, 
                "confidence": round(abs(polarity) * 100, 2)
            }

        # --- RoBERTa Logic ---
        if self.active_name == "roberta":
            if self.model is None:
                return {
                    "sentiment": "positive", 
                    "confidence": 50.0,
                    "note": "RoBERTa mock (Library missing in environment)"
                }
            
            if not self.model:
                return {"sentiment": "error", "confidence": 0.0}

            results = self.model(cleaned_text, top_k=None) 
            #results = self.model(text, top_k=None) 

            scores = {r['label']: r['score'] for r in results}

            neg_score = scores.get('LABEL_0', 0)
            neu_score = scores.get('LABEL_1', 0)
            pos_score = scores.get('LABEL_2', 0)

            if pos_score > neg_score:
                label = "positive"
                conf = pos_score
            else:
                label = "negative"
                conf = neg_score

            return {
                "sentiment": label, 
                "confidence": round(conf * 100, 2)
            }

