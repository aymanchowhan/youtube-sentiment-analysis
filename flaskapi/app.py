import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import re

app = Flask(__name__)
CORS(app)

analyzer = SentimentIntensityAnalyzer()

# Path to the dataset, resolved relative to this script's location
# (not the current working directory) so it works regardless of
# where you launch `python app.py` from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "data", "final_dataset.csv")


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return text.strip()


def get_sentiment(text):
    score = analyzer.polarity_scores(str(text))["compound"]
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"


@app.route("/")
def home():
    return jsonify({"message": "YouTube Sentiment Analysis API", "status": "running"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "comments" not in data:
        return jsonify({"error": "Send JSON with 'comments' key"}), 400

    comments = data["comments"]
    if isinstance(comments, str):
        comments = [comments]

    if not isinstance(comments, list):
        return jsonify({"error": "'comments' must be a string or a list of strings"}), 400

    results = []
    for comment in comments:
        comment_str = str(comment)
        sentiment = get_sentiment(comment_str)
        results.append({
            "comment":   comment_str,
            "sentiment": sentiment,
            "clean":     clean_text(comment_str)
        })

    pos = sum(1 for r in results if r["sentiment"] == "positive")
    neg = sum(1 for r in results if r["sentiment"] == "negative")
    neu = sum(1 for r in results if r["sentiment"] == "neutral")

    return jsonify({
        "total":   len(results),
        "summary": {"positive": pos, "negative": neg, "neutral": neu},
        "results": results
    })


@app.route("/stats", methods=["GET"])
def stats():
    if not os.path.exists(DATASET_PATH):
        return jsonify({"error": f"Dataset not found at {DATASET_PATH}"}), 500

    try:
        df = pd.read_csv(DATASET_PATH)
    except Exception as e:
        return jsonify({"error": f"Failed to read dataset: {str(e)}"}), 500

    required_cols = {"video", "vader_sentiment", "emotion"}
    missing = required_cols - set(df.columns)
    if missing:
        return jsonify({"error": f"Dataset is missing required columns: {sorted(missing)}"}), 500

    # Cast numpy int64 -> native int so jsonify doesn't choke on serialization
    sentiment_counts = {k: int(v) for k, v in df["vader_sentiment"].value_counts().items()}
    emotion_counts = {k: int(v) for k, v in df["emotion"].value_counts().items()}

    return jsonify({
        "total_comments":   int(len(df)),
        "total_videos":     int(df["video"].nunique()),
        "sentiment_counts": sentiment_counts,
        "emotion_counts":   emotion_counts
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)