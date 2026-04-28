from transformers import pipeline

print("Loading BERT model...")
sentiment_model = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    truncation=True
)

# ── Batch script — only runs when you call this file directly ──────
if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm

    df = pd.read_csv("data/cleaned_comments.csv")
    sentiments = []
    print("Running BERT sentiment analysis...")

    for comment in tqdm(df["clean_comment"]):
        try:
            text = str(comment)[:512]
            if text.strip() == "":
                sentiments.append("neutral")
                continue
            result = sentiment_model(text)[0]["label"]
            rating = int(result[0])
            if rating >= 4:
                sentiment = "positive"
            elif rating == 3:
                sentiment = "neutral"
            else:
                sentiment = "negative"
            sentiments.append(sentiment)
        except:
            sentiments.append("neutral")

    df["sentiment_bert"] = sentiments
    df.to_csv("data/sentiment_bert.csv", index=False)
    print("BERT sentiment analysis completed!")


# ── Live prediction — used by app.py ──────────────────────────────
def predict_bert(text):
    try:
        if not isinstance(text, str) or text.strip() == "":
            return "neutral"
        result = sentiment_model(text[:512])[0]["label"]
        rating = int(result[0])
        if rating >= 4:
            return "positive"
        elif rating == 3:
            return "neutral"
        else:
            return "negative"
    except:
        return "neutral"