Twitter Sentiment Analysis Project

This project provides a sentiment analysis system using VADER, TextBlob, and RoBERTa models. It consists of a FastAPI backend and a Streamlit frontend.
Installation and Setup

pip install -r requirements.txt

Download necessary model data for local lexicon engines:
Bash

python -m nltk.downloader vader_lexicon
python -m textblob.download_corpora

    Note on RoBERTa: The RoBERTa model (twitter-roberta-base-sentiment) is not included in the repository due to its size. Upon first execution or when the model is initialized in models.py, the library will automatically download approximately 500MB of weights from the Hugging Face Hub.

Dataset Information

The evaluation module uses a file named twitterTraining.csv. This is based on the Sentiment140 dataset, a collection of 1.6 million tweets formatted for binary sentiment classification (positive and negative). Ensure this file is placed in the project root directory for the benchmarking features to function.
Running the Application

To use the system, you must run both the API and the App in separate terminal windows.
1. Start the API (Backend)

The backend manages the model logic and serves predictions.

uvicorn api:app --reload

2. Start the Streamlit App (Frontend)

The frontend provides the user interface for real-time analysis.
Bash

streamlit run app.py

Testing and Quality

To verify the installation and check code coverage:

pytest --cov=. test_sentiment.py
