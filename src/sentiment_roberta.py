from transformers import pipeline

print("Loading RoBERTa model...")
sentiment_model = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment",
    tokenizer="cardiffnlp/twitter-roberta-base-sentiment"
)

# ── Batch script — only runs when you call this file directly ──────
if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm

    df = pd.read_csv("data/cleaned_comments.csv")
    sentiments = []
    print("Running RoBERTa sentiment analysis...")

    for comment in tqdm(df["clean_comment"]):
        try:
            result = sentiment_model(str(comment)[:512])[0]
            label = result["label"]
            if label == "LABEL_0":
                sentiments.append("negative")
            elif label == "LABEL_1":
                sentiments.append("neutral")
            else:
                sentiments.append("positive")
        except:
            sentiments.append("neutral")

    df["sentiment"] = sentiments
    df.to_csv("data/sentiment_roberta.csv", index=False)
    print("RoBERTa sentiment analysis completed!")


# ── Live prediction — used by app.py ──────────────────────────────
def predict_roberta(text):
    try:
        if not isinstance(text, str) or text.strip() == "":
            return "neutral"
        result = sentiment_model(text[:512])[0]["label"]
        mapping = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}
        return mapping.get(result, "neutral")
    except:
        return "neutral"