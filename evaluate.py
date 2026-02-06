import pandas as pd
import matplotlib.pyplot as plt
from models import SentimentModel
from typing import Dict

def run_evaluator(sample_size: int = 100) -> Dict[str,float]:

    DATASET_PATH = r"C:\Users\User\Desktop\Python Code\kaggle_data\datasets\kazanova\sentiment140\versions\2\twitterTraining.csv"
    df = pd.read_csv(DATASET_PATH, encoding='latin-1', 
                     header=None, 
                     names=['sentiment', 'id', 'date', 'query', 'user', 'text'])

    df = df.sample(sample_size)
    df.columns = ['target', 'id', 'date', 'flag', 'user', 'text']

    df['actual'] = df['target'].map({4:'positive', 0: 'negative'})
    results = {"vader":0, "textblob":0,"roberta":0}
    engine = SentimentModel()

    for model_name in results.keys():
        print(f"Evaluateing...{model_name}")

        engine.active_name = model_name
        engine.initialize_model(model_name)

        correct = 0

        for _, row in df.iterrows():
            if engine.predict(row['text']) == row['actual']:
                correct += 1
        results[model_name] = (correct / sample_size) *100

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