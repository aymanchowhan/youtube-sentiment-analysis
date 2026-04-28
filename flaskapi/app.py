from flask import Flask, request, jsonify
from flask_cors import CORS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import re

app = Flask(__name__)
CORS(app)

analyzer = SentimentIntensityAnalyzer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return text.strip()

def get_sentiment(text):
    score = analyzer.polarity_scores(str(text))["compound"]
    if score >= 0.05:  return "positive"
    elif score <= -0.05: return "negative"
    else: return "neutral"

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

    results = []
    for comment in comments:
        sentiment = get_sentiment(comment)
        results.append({
            "comment":   comment,
            "sentiment": sentiment,
            "clean":     clean_text(comment)
        })

    pos = sum(1 for r in results if r["sentiment"]=="positive")
    neg = sum(1 for r in results if r["sentiment"]=="negative")
    neu = sum(1 for r in results if r["sentiment"]=="neutral")

    return jsonify({
        "total":    len(results),
        "summary":  {"positive": pos, "negative": neg, "neutral": neu},
        "results":  results
    })

@app.route("/stats", methods=["GET"])
def stats():
    df = pd.read_csv("data/final_dataset.csv")
    return jsonify({
        "total_comments":    len(df),
        "total_videos":      df["video"].nunique(),
        "sentiment_counts":  df["vader_sentiment"].value_counts().to_dict(),
        "emotion_counts":    df["emotion"].value_counts().to_dict()
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)