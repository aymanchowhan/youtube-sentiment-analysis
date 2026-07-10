import pandas as pd
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic


def load_embedding_model(model_name="all-MiniLM-L6-v2"):
    """Load the sentence-transformer embedding model. Cache this yourself
    (e.g. with st.cache_resource in the dashboard) — it's expensive to load."""
    return SentenceTransformer(model_name)


def run_bertopic(comments, embedding_model=None, min_topic_size=None,
                  calculate_probabilities=True, verbose=False):
    """
    Fit a fresh BERTopic model on a list of cleaned comments.

    Parameters
    ----------
    comments : list[str]
        Cleaned comment text (use clean_comment output, not raw text).
    embedding_model : SentenceTransformer, optional
        Pre-loaded embedding model. If None, loads the default model
        (slow — prefer passing a cached instance from the caller).
    min_topic_size : int, optional
        Passed through to BERTopic. Left as None by default so the
        offline full-dataset run behaves EXACTLY as before (BERTopic's
        own default of 10). Pass a smaller value (e.g. 3-5) for small
        live batches, otherwise everything gets bucketed as outliers.
    calculate_probabilities, verbose : passed through to BERTopic.

    Returns
    -------
    (topics, probs, topic_model, topic_info)
    """
    if embedding_model is None:
        embedding_model = load_embedding_model()

    embeddings = embedding_model.encode(comments, show_progress_bar=verbose)

    bertopic_kwargs = dict(language="english",
                            calculate_probabilities=calculate_probabilities,
                            verbose=verbose)
    if min_topic_size is not None:
        bertopic_kwargs["min_topic_size"] = min_topic_size

    topic_model = BERTopic(**bertopic_kwargs)
    topics, probs = topic_model.fit_transform(comments, embeddings)
    topic_info = topic_model.get_topic_info()

    return topics, probs, topic_model, topic_info


def _clean_topic_name(name):
    """Turn BERTopic's raw 'Name' ('0_camera_phone_battery') into a readable
    keyword string ('Camera, Phone, Battery')."""
    if not isinstance(name, str):
        return "N/A"
    parts = name.split("_")
    if parts and parts[0].lstrip("-").isdigit():
        parts = parts[1:]
    return ", ".join(p.capitalize() for p in parts if p) or "N/A"


def get_bertopic_topics(comments, embedding_model=None, min_topic_size=3):
    """
    Dashboard-friendly convenience wrapper.

    Fits BERTopic on `comments` and returns a small DataFrame — one row
    per input comment — with the topic number and a human-readable,
    already-cleaned keyword string (e.g. "Camera, Phone, Battery" instead
    of BERTopic's raw "0_camera_phone_battery"), ready to attach onto a
    live results dataframe:

        topics_df = get_bertopic_topics(rt_df["clean_comment"].tolist())
        rt_df["topic"]          = topics_df["topic_number"].values
        rt_df["topic_keywords"] = topics_df["topic_keywords"].values

    Uses a smaller min_topic_size (default 3) than the offline run since
    live batches are much smaller than the full training dataset.
    Topic -1 means "outlier / no clear topic" (BERTopic convention).
    """
    topics, probs, topic_model, topic_info = run_bertopic(
        comments, embedding_model=embedding_model, min_topic_size=min_topic_size
    )

    # Topic number -> cleaned, human-readable keyword string
    keyword_lookup = {row["Topic"]: _clean_topic_name(row["Name"]) for _, row in topic_info.iterrows()}

    return pd.DataFrame({
        "topic_number": topics,
        "topic_keywords": [keyword_lookup.get(t, "Outlier / No Topic") for t in topics],
    })


if __name__ == "__main__":
    print("Loading dataset...")
    df = pd.read_csv("data/final_dataset.csv")
    df = df.dropna(subset=["clean_comment"])

    print("Loading sentence transformer model...")
    embedding_model = load_embedding_model()

    videos = df["video"].unique()
    print(f"Videos found: {list(videos)}")

    all_results  = []   # per-comment topic assignments, one chunk per video
    all_summaries = []  # per-video topic tables (BERTopic get_topic_info()), stacked

    for video in videos:
        print(f"\nProcessing: {video}")
        video_df = df[df["video"] == video].copy()
        video_df = video_df[video_df["clean_comment"].astype(str).str.strip() != ""]
        comments = video_df["clean_comment"].astype(str).tolist()

        if len(comments) < 5:
            print(f"  Skipping — not enough comments ({len(comments)})")
            continue

        # Each video gets its OWN BERTopic model, fit only on that video's
        # comments — topic ids are meaningful only within a given video,
        # the same way the old per-video LDA script worked.
        topics, probs, topic_model, topic_info = run_bertopic(
            comments, embedding_model=embedding_model, min_topic_size=5, verbose=False
        )

        video_df["topic"] = topics
        video_df["topic_probability"] = [max(p) if p is not None else 0 for p in probs]
        all_results.append(video_df)

        topic_info = topic_info.copy()
        topic_info["video"] = video
        all_summaries.append(topic_info)

        n_topics = len(topic_info) - (1 if -1 in topic_info["Topic"].values else 0)
        print(f"  Found {n_topics} topics across {len(comments)} comments")

    result_df = pd.concat(all_results, ignore_index=True)
    result_df.to_csv("data/topic_results.csv", index=False)

    summary_df = pd.concat(all_summaries, ignore_index=True)
    summary_df.to_csv("data/topic_summary.csv", index=False)

    print("\nTopic modeling completed!")
    print(f"Videos processed: {len(all_results)}")
    print(f"Total comments: {len(result_df)}")
    print(summary_df.head(10))