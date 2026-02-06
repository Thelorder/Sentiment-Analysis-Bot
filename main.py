import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import pandas as pd
from utils import clean_tweet
from models import SentimentModel
from sklearn.metrics import accuracy_score
from textblob import TextBlob



# 1. Load Data (Subset for speed)
DATASET_PATH = r"C:\Users\User\Desktop\Python Code\kaggle_data\datasets\kazanova\sentiment140\versions\2\twitterTraining.csv"
df = pd.read_csv(DATASET_PATH, encoding='latin-1', header=None, names=['sentiment', 'id', 'date', 'query', 'user', 'text'])
df['sentiment'] = df['sentiment'].map({0: 'negative', 4: 'positive'})

# 3. Test Model via Factory
model_wrapper = SentimentModel()
sample_df = df.sample(20).copy()
sample_df['clean_text'] = sample_df['text'].apply(clean_tweet)
sample_df['prediction'] = sample_df['clean_text'].apply(model_wrapper.predict)

accuracy = accuracy_score(sample_df['sentiment'], sample_df['prediction'])

print("\n" + "="*30)
print(f"MODEL ACCURACY: {accuracy * 100:.2f}%")
print("="*30)

# 5. Show detailed comparison
print("\nDetailed Results (Actual vs Predicted):")
for index, row in sample_df.iterrows():
    status = "✅" if row['sentiment'] == row['prediction'] else "❌"
    print(f"{status} Actual: {row['sentiment']} | Pred: {row['prediction']} | Text: {row['clean_text'][:70]}...")