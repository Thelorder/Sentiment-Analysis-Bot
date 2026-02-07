"""
Logic Module for Sentiment Analysis Models.
"""
import os
from typing import Dict, Any
import yaml
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

# Set environment variables
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class SentimentModel:
    """Manager class for various sentiment analysis pre-trained models."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Initializes the model based on the configuration file."""
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
        except FileNotFoundError:
            return {"api_config": {"active_model": "vader"},
                    "model_config": {"vader": {"threshold": 0.05}}}

    def initialize_model(self, model_name: str) -> None:
        """Sets up the specific model engine and thresholds."""
        if model_name == "vader":
            self.model = SentimentIntensityAnalyzer()
            self.threshold = self.config['model_config']['vader']['threshold']
        elif model_name == "textblob":
            self.threshold = self.config['model_config']['textblob']['polarity_threshold']
        elif model_name == "roberta":
            try:
                from transformers import pipeline
                model_id = self.config['model_config']['roberta']['model_name']
                self.model = pipeline("sentiment-analysis", model=model_id, framework="pt")
            except ImportError:
                self.model = None

    def predict(self, text: str) -> str:
        """Predicts sentiment for a given string using the active model."""
        if self.active_name == "vader":
            score = self.model.polarity_scores(text)['compound']
            return "positive" if score >= self.threshold else "negative"

        if self.active_name == "textblob":
            score = TextBlob(text).sentiment.polarity
            return "positive" if score > self.threshold else "negative"

        if self.active_name == "roberta" and self.model:
            result = self.model(text)[0]
            return "positive" if result['label'] == "LABEL_2" else "negative"


        return "neutral"
