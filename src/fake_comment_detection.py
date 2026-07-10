import pandas as pd
import re

# --- Emoji pattern (module-level so it's compiled once) ---
emoji_pattern = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols
    "\U0001F680-\U0001F6FF"  # transport
    "\U0001F1E0-\U0001F1FF"  # flags
    "]+", flags=re.UNICODE
)


def has_excessive_emojis(text):
    return len(emoji_pattern.findall(str(text))) >= 5  # 5+ emojis = likely spam/fake


def detect_fake_comments(df, comment_col="comment", video_col="video"):
    """
    Rule-based fake/bot comment detector.

    Runs the same 5 rules as the original script (duplicate, too_short,
    bot_flood, all_caps, excessive_emoji) against ANY dataframe of comments —
    the full offline dataset OR a small live batch from the dashboard.

    IMPORTANT: always pass the ORIGINAL (uncleaned) comment column.
    Casing, punctuation, and emojis are required signals for these rules
    and are stripped by clean_comment().

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `comment_col`. `video_col` is optional — if absent,
        the bot_flood rule is skipped gracefully (e.g. for a live batch
        that only contains comments from a single video).
    comment_col : str
        Column holding the original, uncleaned comment text.
    video_col : str
        Column identifying which video a comment belongs to (used for the
        bot_flood rule only).

    Returns
    -------
    pd.DataFrame
        Copy of `df` with two new columns: 'fake_label' ('fake'/'real')
        and 'fake_reason' (comma-separated list of triggered rules).
    """
    df = df.copy()

    # --- Rule 1: Exact duplicate comments (5+ times) ---
    duplicate_counts = df[comment_col].value_counts()
    duplicate_mask = df[comment_col].isin(duplicate_counts[duplicate_counts >= 5].index)

    # --- Rule 2: Very short comments (1 word max) ---
    short_mask = df[comment_col].astype(str).str.split().str.len() <= 1

    # --- Rule 3: Same comment repeated by same video (bot flooding) ---
    if video_col in df.columns:
        video_comment_counts = df.groupby(
            [video_col, comment_col]
        )[comment_col].transform("count")
        flood_mask = video_comment_counts >= 5
    else:
        flood_mask = pd.Series([False] * len(df), index=df.index)

    # --- Rule 4: All-caps comments (aggressive bots) ---
    caps_mask = df[comment_col].apply(
        lambda x: str(x).isupper() and len(str(x)) > 20
    )

    # --- Rule 5: Repeated emojis ---
    excessive_emoji_mask = df[comment_col].apply(has_excessive_emojis)

    def classify_fake(idx):
        reasons = []
        if duplicate_mask[idx]:
            reasons.append("duplicate")
        if short_mask[idx]:
            reasons.append("too_short")
        if flood_mask[idx]:
            reasons.append("bot_flood")
        if caps_mask[idx]:
            reasons.append("all_caps")
        if excessive_emoji_mask[idx]:
            reasons.append("excessive_emoji")

        if reasons:
            return "fake", ", ".join(reasons)
        return "real", "none"

    results = [classify_fake(i) for i in df.index]
    df["fake_label"] = [r[0] for r in results]
    df["fake_reason"] = [r[1] for r in results]
    return df


if __name__ == "__main__":
    print("Loading dataset...")
    df = pd.read_csv("data/final_dataset.csv")  # runs AFTER merge_results.py

    print("Detecting fake comments...")
    df = detect_fake_comments(df, comment_col="comment", video_col="video")

    df.to_csv("data/fake_comment_results.csv", index=False)

    # Summary
    fake_count = (df["fake_label"] == "fake").sum()
    total = len(df)
    print(f"\nFake Comment Detection Complete!")
    print(f"Total comments : {total}")
    print(f"Fake detected  : {fake_count} ({fake_count/total*100:.1f}%)")
    print(f"Real comments  : {total - fake_count}")
    print(f"\nFake reasons breakdown:")
    print(df[df["fake_label"] == "fake"]["fake_reason"].value_counts())
    print(f"\nSaved to data/fake_comment_results.csv")