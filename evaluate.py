import pandas as pd
import matplotlib.pyplot as plt
from models import SentimentModel
from typing import Dict
import os

def run_evaluator(sample_size: int = 100) -> Dict[str, float]:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_PATH = os.path.join(BASE_DIR, "twitterTraining.csv")

    df = pd.read_csv(DATASET_PATH, encoding='latin-1', 
                     header=None, 
                     names=['sentiment', 'id', 'date', 'query', 'user', 'text'])

    df = df.sample(sample_size)
    df['actual'] = df['sentiment'].map({4: 'positive', 0: 'negative'})
    
    results = {"vader": 0.0, "textblob": 0.0, "roberta": 0.0}
    engine = SentimentModel()

    for model_name in ["vader", "textblob", "roberta"]:
        print(f"Evaluating {model_name}...")
        engine.set_model(model_name) 
        
        correct_predictions = 0
        for _, row in df.iterrows():
            prediction = engine.predict(row['text'])
            if prediction == row['actual']:
                correct_predictions += 1
        
        accuracy = (correct_predictions / sample_size) * 100
        results[model_name] = accuracy

    return results

def plot_results(results: Dict[str,float]) -> None:
    names = [n.upper() for n in results.keys()]
    values = list(results.values())

    plt.figure(figsize=(10, 6))
    plt.bar(names, values, color=['skyblue', 'lightgreen', 'salmon'])
    plt.ylabel('Accuracy (%)')
    plt.title('Sentiment Model Comparison')
    plt.ylim(0, 100)
    plt.savefig('evaluation_chart.png')
    print("Chart saved as evaluation_chart.png")

if __name__ == "__main__":
    scores = run_evaluator(10)
    plot_results(scores)