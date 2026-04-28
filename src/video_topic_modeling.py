import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import warnings
warnings.filterwarnings("ignore")

print("Loading dataset...")
df = pd.read_csv("data/final_dataset.csv")

print(f"Total rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"Videos found: {df['video'].unique()}")

results = []
videos = df["video"].unique()

for video in videos:
    print(f"\nProcessing: {video}")
    comments = df[df["video"] == video]["clean_comment"].dropna()
    comments = comments[comments.str.strip() != ""]

    print(f"  Comments available: {len(comments)}")

    if len(comments) < 5:
        print(f"  Skipping — not enough comments")
        continue

    try:
        vectorizer = CountVectorizer(
            stop_words="english",
            max_features=300,
            min_df=2,
            max_df=0.95
        )
        X = vectorizer.fit_transform(comments)

        if X.shape[1] == 0:
            print(f"  Skipping — vocabulary empty after filtering")
            continue

        n_topics = min(3, X.shape[1])
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=10
        )
        lda.fit(X)

        words = vectorizer.get_feature_names_out()
        for topic_idx, topic in enumerate(lda.components_):
            top_indices = topic.argsort()[-7:][::-1]
            top_words = [words[i] for i in top_indices]
            results.append({
                "video": video,
                "topic_number": topic_idx + 1,
                "topic_keywords": ", ".join(top_words)
            })
            print(f"  Topic {topic_idx+1}: {', '.join(top_words)}")

    except Exception as e:
        print(f"  ERROR for {video}: {e}")
        continue

topic_df = pd.DataFrame(results)
topic_df.to_csv("data/topic_results.csv", index=False)
print(f"\nDone! {len(topic_df)} topic rows saved to data/topic_results.csv")
print(topic_df)