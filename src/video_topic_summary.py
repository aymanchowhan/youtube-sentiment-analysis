import pandas as pd

print("Loading topic results...")
df = pd.read_csv("data/topic_results.csv")

print(f"Rows loaded: {len(df)}")
print(f"Videos: {df['video'].unique()}")

summary = df.groupby("video")["topic_keywords"].apply(
    lambda x: " | ".join(x)
).reset_index()

summary.columns = ["video", "all_topics"]
summary.to_csv("data/topic_summary.csv", index=False)

print("\nTopic Summary:")
print(summary.to_string(index=False))
print("\nSaved to data/topic_summary.csv")