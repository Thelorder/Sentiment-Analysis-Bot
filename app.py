# import streamlit as st
# import requests
# from textblob import TextBlob
# from nltk.sentiment.vader import SentimentIntensityAnalyzer

# # Page Config
# st.set_page_config(page_title="Twitter AI Battle", layout="wide")

# st.sidebar.title("Settings")
# mode = st.sidebar.selectbox(
#     "Choose Analysis Mode:",
#     ["RoBERTa (Transformer)", 
#      "VADER (Lexicon)", 
#      "TextBlob (Pattern)", 
#      "ALL (AI Battle)"]
# )

# st.title("🐦 Twitter Sentiment: The AI Battle")
# st.markdown("""
# Compare **VADER**, **TextBlob**, and **RoBERTa** in real-time.
# """)

# # Initialize Lexicon models locally for speed
# vader_analyzer = SentimentIntensityAnalyzer()

# # UI Input
# tweet_input: str = st.text_area(
#     "Enter a tweet to test:",
#     "I absolutely love how this project is coming together!"
# )

# if st.button("Analyze"):
#     if mode == "ALL (AI Battle)":
#         if not tweet_input.strip():
#             st.warning("Please enter some text to analyze.")
#         else:
#             with st.spinner('AI Models are competing...'):
#                 try:
#                     api_url = "http://127.0.0.1:8000/predict"
#                     response = requests.post(api_url, json={"text": tweet_input}, timeout=10)

#                     if response.status_code == 200:
#                         res = response.json()
#                         r_label: str = res['sentiment'].capitalize()
#                         r_conf: str = f"{res.get('confidence', 'N/A')}%"
#                     else:
#                         st.error(f"API Error: Status {response.status_code}")
#                         st.stop()

#                     # VADER
#                     v_scores = vader_analyzer.polarity_scores(tweet_input)
#                     v_label: str = "Positive" if v_scores['compound'] >= 0.05 else "Negative"
#                     v_conf: str = f"{abs(v_scores['compound']) * 100:.1f}%"

#                     # TextBlob
#                     t_pol = TextBlob(tweet_input).sentiment.polarity
#                     t_label: str = "Positive" if t_pol > 0 else "Negative"
#                     t_conf: str = f"{abs(t_pol) * 100:.1f}%"

#                     st.subheader("Final Decision")
#                     c1, c2, c3 = st.columns(3)
#                     c1.metric("VADER Sentiment", v_label)
#                     c2.metric("TextBlob Sentiment", t_label)
#                     c3.metric("RoBERTa Sentiment", r_label)

#                     st.divider()

#                     st.subheader("Model Certainty / Intensity")
#                     c4, c5, c6 = st.columns(3)
#                     c4.metric("VADER Score", v_conf, "Rule-based intensity")
#                     c5.metric("TextBlob Score", t_conf, "Pattern-based polarity")
#                     c6.metric("RoBERTa Score", r_conf, "Transformer Confidence")

#                 except requests.exceptions.ConnectionError:
#                     st.error("❌ **Backend Offline:** The RoBERTa API (api.py) is not running on port 8000.")
#                     st.info("Run: `python -m uvicorn api:app --reload` in your terminal.")
#                 except Exception as e:
#                     st.error(f"⚠️ An unexpected error occurred: {e}")
#     else:
#         mapping = {
#             "RoBERTa (Transformer)": "roberta",
#             "VADER (Lexicon)": "vader",
#             "TextBlob (Pattern)": "textblob"
#         }

#         active_key = mapping[mode]

#         with st.spinner('Querying API...'):
#             try:
#                 res = requests.post("http://127.0.0.1:8000/predict",
#                                     json={"text": tweet_input}).json()
#                 st.metric(f"{mode} Result", res['sentiment'].capitalize())
#                 col2.metric("Confidence", f"{res['confidence']}%")
#             except:
#                  st.error("Is your API running?")
                 
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

# Page Config
st.set_page_config(page_title="Twitter AI Battle", layout="wide")

# Constants should be UPPER_CASE to satisfy Pylint
API_URL = "http://127.0.0.1:8000/predict"

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

if st.sidebar.button("📊 Run Global Benchmark"):
    if run_evaluator:
        with st.spinner("Calculating accuracy for all models..."):
            results = run_evaluator(sample_size=100)
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
st.title("🐦 Twitter Sentiment: The AI Battle")
st.markdown("""
Compare **VADER**, **TextBlob**, and **RoBERTa** in real-time.
""")

# Initialize Lexicon models locally
vader_analyzer = SentimentIntensityAnalyzer()

# UI Input
tweet_input: str = st.text_area(
    "Enter a tweet to test:",
    "I absolutely love how this project is coming together!"
)

if st.button("Analyze"):
    if mode == "ALL (AI Battle)":
        if not tweet_input.strip():
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner('AI Models are competing...'):
                try:
                    # RoBERTa via API (Setting a timeout to satisfy Pylint)
                    response = requests.post(API_URL, json={"text": tweet_input}, timeout=15)

                    if response.status_code == 200:
                        res = response.json()
                        r_label: str = res['sentiment'].capitalize()
                        r_conf: str = f"{res.get('confidence', 'N/A')}%"
                    else:
                        st.error(f"API Error: Status {response.status_code}")
                        st.stop()

                    # VADER Local
                    v_scores = vader_analyzer.polarity_scores(tweet_input)
                    v_label: str = "Positive" if v_scores['compound'] >= 0.05 else "Negative"
                    v_conf: str = f"{abs(v_scores['compound']) * 100:.1f}%"

                    # TextBlob Local
                    t_pol = TextBlob(tweet_input).sentiment.polarity
                    t_label: str = "Positive" if t_pol > 0 else "Negative"
                    t_conf: str = f"{abs(t_pol) * 100:.1f}%"

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
            "TextBlob (Pattern)": "textblob"
        }

        active_key = mapping[mode]

        with st.spinner(f'Querying API for {mode}...'):
            try:
                # Use timeout here as well
                res_raw = requests.post(API_URL, json={"text": tweet_input}, timeout=15)
                res = res_raw.json()
                
                # Fixed 'col2 undefined' error by creating columns
                res_col1, res_col2 = st.columns(2)
                res_col1.metric(f"{mode} Result", res['sentiment'].capitalize())
                res_col2.metric("Confidence", f"{res['confidence']}%")
            except requests.exceptions.RequestException:
                st.error("Could not connect to the API. Ensure api.py is running.")
            except Exception as e:
                st.error(f"Processing error: {e}")
