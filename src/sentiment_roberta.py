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

    for comment in tqdm(df["comment"]):  # use ORIGINAL comment (RoBERTa is trained on raw tweet-like text)
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


def predict_roberta_batch(texts, batch_size=32):
    """
    Batched RoBERTa prediction for the Live Analysis dashboard.

    Calling predict_roberta() once per comment via .apply() means a
    separate tokenize + forward pass per comment — very slow for 100+
    live comments. The underlying HF pipeline can batch the forward pass
    itself, which is much faster (especially on GPU, but noticeably
    faster on CPU too). This does the same LABEL_0/1/2 -> negative/
    neutral/positive mapping as predict_roberta(), just for a whole list
    of comments in one (or a few) model calls.

    Parameters
    ----------
    texts : list[str]
        Original (uncleaned) comment text.
    batch_size : int
        How many comments the pipeline processes per forward pass.

    Returns
    -------
    list[str]  — one sentiment label per input text, same order.
    """
    mapping = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}

    safe_texts = [
        t[:512] if isinstance(t, str) and t.strip() != "" else ""
        for t in texts
    ]

    results = ["neutral"] * len(safe_texts)
    non_empty_idx = [i for i, t in enumerate(safe_texts) if t.strip() != ""]

    if non_empty_idx:
        try:
            outputs = sentiment_model(
                [safe_texts[i] for i in non_empty_idx],
                batch_size=batch_size,
                truncation=True
            )
            for idx, out in zip(non_empty_idx, outputs):
                results[idx] = mapping.get(out["label"], "neutral")
        except Exception:
            pass  # leave defaults as "neutral" if batching fails for any reason

    return results