"""
Data loading, lightweight offline detectors, and backend model wiring.

Everything that touches disk (CSVs) or src/ modules lives here so the tab
files and the main entry point don't need to know where data comes from.
"""
import re
import streamlit as st
import pandas as pd

# ─────────────────────────────────────────
# LOAD OFFLINE DATASET
# ─────────────────────────────────────────
@st.cache_data
def load():
    df         = pd.read_csv("data/final_dataset.csv")
    topic_df   = pd.read_csv("data/topic_results.csv")
    model_df   = pd.read_csv("data/model_comparison.csv")
    summary_df = pd.read_csv("data/topic_summary.csv")

    for col in ["vader_sentiment", "bert_sentiment", "roberta_sentiment"]:
        df[col] = df[col].astype(str).str.lower().str.strip()

    for fname, col, default in [
        ("spam_results.csv",         "spam_label",    "not_spam"),
        ("sarcasm_results.csv",      "sarcasm_label", "not_sarcastic"),
        ("fake_comment_results.csv", "fake_label",    "real"),
    ]:
        try:
            df[col] = pd.read_csv(f"data/{fname}")[col].values
        except Exception:
            df[col] = default

    return df, topic_df, model_df, summary_df


# ─────────────────────────────────────────
# LIGHTWEIGHT KEYWORD DETECTORS (used both offline as fallback and live)
# ─────────────────────────────────────────
EMOTION_KEYWORDS = {
    "joy":          ["love","amazing","great","excellent","wonderful","happy","awesome","fantastic","best"],
    "anger":        ["hate","worst","terrible","awful","bad","disgusting","stupid","useless","horrible"],
    "sadness":      ["sad","miss","disappointed","unfortunate","cry","poor","regret"],
    "fear":         ["scared","afraid","worried","nervous","anxious","panic"],
    "surprise":     ["wow","omg","unbelievable","shocked","unexpected","whoa"],
    "disgust":      ["gross","disgusting","nasty","yuck","eww"],
    "anticipation": ["excited","waiting","soon","hope","looking forward","can't wait"],
}

def detect_emotion(text):
    if not isinstance(text, str): return "neutral"
    t = text.lower()
    for emotion, kws in EMOTION_KEYWORDS.items():
        if any(k in t for k in kws):
            return emotion
    return "neutral"

def detect_spam(text):
    if not isinstance(text, str): return "not_spam"
    signals = ["subscribe","sub4sub","check my channel","visit my","free followers",
               "click here","giveaway","www.","http","bit.ly","follow me"]
    return "spam" if any(s in text.lower() for s in signals) else "not_spam"

def detect_sarcasm(text):
    if not isinstance(text, str): return "not_sarcastic"
    signals = ["oh great","yeah right","sure sure","totally","wow so helpful",
               "obviously","clearly the best","not","like that helps","so useful"]
    return "sarcastic" if any(s in text.lower() for s in signals) else "not_sarcastic"


# ─────────────────────────────────────────
# BACKEND MODEL WIRING (Live Analysis)
# ─────────────────────────────────────────
# Lightweight — no heavy ML deps, safe to import unconditionally
from src.fake_comment_detection import detect_fake_comments

try:
    from src.youtube_live_comments import fetch_comments, get_video_details
    from src.sentiment_roberta import predict_roberta, predict_roberta_batch
    from src.data_preprocessing import clean_comment
    MODELS_LOADED = True
except ImportError as e:
    MODELS_LOADED = False
    st.warning(f"Some modules not found: {e}. Running in demo mode.")

# Topic modeling has its own heavier deps (sentence-transformers, bertopic) —
# gated separately so a missing torch/bertopic install doesn't disable RoBERTa etc.
try:
    from src.topic_modeling import get_bertopic_topics, load_embedding_model
    TOPIC_MODEL_LOADED = True
except ImportError:
    TOPIC_MODEL_LOADED = False


def _clean_comment_fallback(text):
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = re.sub(r"http\S+|www\.\S+", "", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

if not MODELS_LOADED:
    clean_comment = _clean_comment_fallback


@st.cache_resource
def get_live_embedding_model():
    return load_embedding_model()