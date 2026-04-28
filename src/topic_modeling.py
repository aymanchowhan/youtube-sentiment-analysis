import pandas as pd
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic

print("Loading dataset...")

df = pd.read_csv("data/final_dataset.csv")

# Remove empty comments
df = df.dropna(subset=["clean_comment"])

# Convert comments to string
comments = df["clean_comment"].astype(str).tolist()

print("Loading sentence transformer model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings...")

embeddings = model.encode(comments, show_progress_bar=True)

print("Running BERTopic...")

topic_model = BERTopic()

topics, probs = topic_model.fit_transform(comments, embeddings)

df["topic"] = topics

df.to_csv("data/topic_results.csv", index=False)

print("Topic modeling completed!")
topic_info = topic_model.get_topic_info()

topic_info.to_csv("data/topic_summary.csv", index=False)

print(topic_info.head())