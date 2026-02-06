import re

def clean_tweet(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'@[A-Za-z0-9_]+', '', text) # Remove @mentions
    text = re.sub(r'http\S+', '', text)        # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)    # Remove special characters/numbers
    return text.strip()