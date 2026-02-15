"""
Streamlit frontend for the Twitter Sentiment Analysis project.
Includes real-time 'AI Battle' mode and historical model benchmarking.
"""
import streamlit as st
import requests
import matplotlib.pyplot as plt
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

try:
    from evaluate import run_evaluator
except ImportError:
    run_evaluator = None

st.set_page_config(page_title="Twitter AI Battle", layout="wide")

PREDICT_URL = "http://127.0.0.1:8000/predict"
COMAPRE_URL = "http://127.0.0.1:8000/compare" 

# Sidebar
st.sidebar.title("Settings")
mode = st.sidebar.selectbox(
    "Choose Analysis Mode:",
    ["RoBERTa (Transformer)",
     "VADER (Lexicon)",
     "TextBlob (Pattern)",
     "ALL (AI Battle)"]
)

# Benchmark Section in Sidebar
st.sidebar.divider()
st.sidebar.subheader("Model Benchmarking")
st.sidebar.write("Run accuracy tests on the Sentiment140 dataset.")

sample_size = st.sidebar.slider("Data", 100,1000,500)

if st.sidebar.button("📊 Run Global Benchmark"):
    if run_evaluator:
        with st.spinner("Calculating accuracy for all models..."):
            results = run_evaluator(sample_size)
            if results:
                fig, ax = plt.subplots()
                names = [n.upper() for n in results.keys()]
                values = list(results.values())
                ax.bar(names, values, color=['#4F8BF9', '#29B094', '#FF4B4B'])
                ax.set_ylabel("Accuracy (%)")
                ax.set_title("Dataset Accuracy Comparison")
                st.sidebar.pyplot(fig)
                st.sidebar.success("Benchmark completed successfully!")
            else:
                st.sidebar.error("Could not find dataset file.")
    else:
        st.sidebar.error("evaluate.py not found in directory.")

# Main UI
st.title("{o,o} Twitter Sentiment: The AI Battle")
st.markdown("""
Compare **VADER**, **TextBlob**, and **RoBERTa** in real-time.
""")

vader_analyzer = SentimentIntensityAnalyzer()

# UI Input
tweet_input: str = st.text_area(
    "Enter a tweet to test:",
    "I am feelling better and better!"
)

if st.button("Analyze"):
    if mode == "ALL (AI Battle)":
        if not tweet_input.strip():
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner('AI Models are competing...'):
                try:
                    
                    response = requests.post(
                        COMAPRE_URL,
                        json={"text":tweet_input},
                        timeout= 30
                    )

                    if response.status_code == 200:
                        data = response.json()["results"]
                        
                        # RoBERTa
                        r_label = data.get("roberta", {}).get("sentiment", "N/A").capitalize()
                        r_conf = f"{data.get('roberta', {}).get('confidence', 0)}%"
                        
                        # VADER
                        v_label = data.get("vader", {}).get("sentiment", "N/A").capitalize()
                        v_conf = f"{data.get('vader', {}).get('confidence', 0)}%"
                        
                        # TextBlob
                        t_label = data.get("textblob", {}).get("sentiment", "N/A").capitalize()
                        t_conf = f"{data.get('textblob', {}).get('confidence', 0)}%"
                        
                        st.subheader("Final Decision")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("VADER Sentiment", v_label)
                        c2.metric("TextBlob Sentiment", t_label)
                        c3.metric("RoBERTa Sentiment", r_label)

                        st.divider()

                        st.subheader("Model Certainty / Intensity")
                        c4, c5, c6 = st.columns(3)
                        c4.metric("VADER Score", v_conf, "Rule-based intensity")
                        c5.metric("TextBlob Score", t_conf, "Pattern-based polarity")
                        c6.metric("RoBERTa Score", r_conf, "Transformer Confidence")
                    else:
                        st.error(f"API Error: Status {response.status_code}")
                        st.stop()
                            
                except requests.exceptions.ConnectionError:
                    st.error("❌ **Backend Offline:** The RoBERTa API (api.py) is not running.")
                except requests.exceptions.Timeout:
                    st.error("⌛ **Timeout:** The API took too long to respond.")
                except Exception as e:
                    st.error(f"⚠️ An unexpected error occurred: {e}")
    else:
        mapping = {
            "RoBERTa (Transformer)": "roberta",
            "VADER (Lexicon)": "vader",
            "TextBlob (Pattern)": "textblob",
        }

        active_key = mapping[mode]

        with st.spinner(f'Querying API for {mode}...'):
            try:

                res_raw = requests.post(
                    PREDICT_URL, 
                    json={"text": tweet_input, "model": active_key},
                    timeout=30
                )
                res = res_raw.json()
                
                res_col1, res_col2 = st.columns(2)
                res_col1.metric(f"{mode} Result", res['sentiment'].capitalize())
                res_col2.metric("Confidence", f"{res['confidence']}%")
            except requests.exceptions.RequestException:
                st.error("Could not connect to the API. Ensure api.py is running.")
            except Exception as e:
                st.error(f"Processing error: {e}")
