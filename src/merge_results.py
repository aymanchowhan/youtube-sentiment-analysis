import pandas as pd

print("Loading datasets...")

vader = pd.read_csv("data/sentiment_vader.csv")
roberta = pd.read_csv("data/sentiment_roberta.csv")
bert = pd.read_csv("data/sentiment_bert.csv")
emotion = pd.read_csv("data/emotion_results.csv")
spam = pd.read_csv("data/spam_results.csv")
sarcasm = pd.read_csv("data/sarcasm_results.csv")

final_df = pd.DataFrame()

final_df["comment"] = vader["comment"]
final_df["clean_comment"] = vader["clean_comment"]
final_df["video"] = vader["video"]  # Add if it exists

final_df["vader_sentiment"] = vader.iloc[:, -1]
final_df["roberta_sentiment"] = roberta.iloc[:, -1]
final_df["bert_sentiment"] = bert.iloc[:, -1]

final_df["emotion"] = emotion.iloc[:, -1]
final_df["spam_label"] = spam.iloc[:, -1]
final_df["sarcasm_label"] = sarcasm.iloc[:, -1]

final_df.to_csv("data/final_dataset.csv", index=False)

print("Final dataset created successfully")
print(f"Shape: {final_df.shape}")
print(f"Columns: {final_df.columns.tolist()}")