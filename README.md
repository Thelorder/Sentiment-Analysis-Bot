# Twitter Sentiment Analysis Project

This project provides a real-time sentiment analysis system comparing three popular models:

- **VADER** (rule-based, lexicon + heuristics)
- **TextBlob** (simple pattern-based)
- **RoBERTa** (transformer-based, twitter-roberta-base-sentiment)

It includes:
- A **FastAPI** backend for model inference and model comparison ("AI Battle")
- A **Streamlit** frontend for interactive analysis
- Basic evaluation/benchmarking on a subset of the Sentiment140 dataset

## Features

- Single-model prediction endpoint (`/predict`)
- Multi-model comparison endpoint (`/compare`)
- Interactive web interface with real-time "AI Battle" mode
- Model benchmarking against labeled tweets

## Installation & Setup

1. Clone the repository

git clone <your-repo-url>
cd TwitterBot   # or whatever your folder is named

2. Install dependencies
pip install -r requirements.txt 

Note: For roberta you will alsow need 
pip install transformers torch

3. Download required NLTK data (only VADER needs it)
python -m nltk.downloader vader_lexicon

Important: RoBERTa Model Download
The RoBERTa model (cardiffnlp/twitter-roberta-base-sentiment) is not included in the repository (≈500–550 MB).

When you first set model = "roberta" (via the API or Streamlit UI), or when the default model in config.yaml is "roberta",
the transformers library will automatically download the model weights and tokenizer from Hugging Face Hub.
This happens only once — subsequent runs load from local cache (~/.cache/huggingface/hub/).

Expect to see a download progress in the terminal wduring the first run.

4. Dataset for Benchmarking
The evaluation script (evaluate.py) uses a file named twitterTraining.csv.
This should be a copy (or renamed version) of the Sentiment140 dataset:

1.6 million tweets
Binary sentiment: 0 = negative, 4 = positive
Columns expected: sentiment, id, date, query, user, text

How to get it:

Download from Kaggle:
https://www.kaggle.com/datasets/kazanova/sentiment140
(file: training.1600000.processed.noemoticon.csv ≈ 80–240 MB zipped)
Or from Hugging Face Datasets:
https://huggingface.co/datasets/stanfordnlp/sentiment140

Steps:

Download and unzip
Rename the file to twitterTraining.csv
(or change DATASET_PATH in evaluate.py)
Place it in the project root directory

5. Running the Application
You need two terminal windows (or use a process manager like concurrently).
5:1. Start the FastAPI backend

Bash
uvicorn api:app --reload --port 8000

5:2. Start the Streamlit frontend

Bash
streamlit run app.py

Open http://localhost:8501 in your browser.

6. Testing & Code Quality
Run the unit tests:

Bash 
pytest

Or

Bash

pytest --cov=api --cov=models --cov-branch --cov-report=term-missing
