import pandas as pd
from transformers import pipeline
from tqdm import tqdm

# Load dataset
df = pd.read_csv("data/cleaned_comments.csv")

print("Loading emotion detection model...")

emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    truncation=True
)

emotions = []

print("Detecting emotions...")

for comment in tqdm(df["comment"]):  # use ORIGINAL comment (emojis/punctuation carry emotional signal)

    try:
        text = str(comment)[:512]

        if text.strip() == "":
            emotions.append("neutral")
            continue

        result = emotion_model(text)[0]["label"]

        emotions.append(result)

    except:
        emotions.append("neutral")

df["emotion"] = emotions

df.to_csv("data/emotion_results.csv", index=False)

print("Emotion detection completed!")