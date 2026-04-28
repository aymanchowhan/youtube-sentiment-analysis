from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

# ── Batch script — only runs when you call this file directly ──────
if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm

    df = pd.read_csv("data/cleaned_comments.csv")
    sentiments = []
    print("Running VADER sentiment analysis...")

    for comment in tqdm(df["clean_comment"]):
        score = analyzer.polarity_scores(str(comment))
        compound = score["compound"]
        if compound >= 0.05:
            sentiment = "positive"
        elif compound <= -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        sentiments.append(sentiment)

    df["sentiment_vader"] = sentiments
    df.to_csv("data/sentiment_vader.csv", index=False)
    print("VADER sentiment analysis completed!")


# ── Live prediction — used by app.py ──────────────────────────────
def predict_vader(text):
    if not isinstance(text, str) or text.strip() == "":
        return "neutral"
    score = analyzer.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"