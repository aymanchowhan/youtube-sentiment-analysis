import pandas as pd

print("Loading dataset...")
df = pd.read_csv("data/spam_results.csv")  # builds on spam results

flags = []

# --- Rule 1: Exact duplicate comments ---
duplicate_mask = df.duplicated(subset=["comment"], keep=False)

# --- Rule 2: Very short comments (likely bot: "nice", "great", "lol") ---
short_mask = df["clean_comment"].str.split().str.len() <= 2

# --- Rule 3: Same comment repeated by same video (bot flooding) ---
video_comment_counts = df.groupby(
    ["video", "comment"]
)["comment"].transform("count")
flood_mask = video_comment_counts >= 3

# --- Rule 4: All-caps comments (aggressive bots) ---
caps_mask = df["comment"].apply(
    lambda x: str(x).isupper() and len(str(x)) > 10
)

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
    
    if reasons:
        return "fake", ", ".join(reasons)
    return "real", "none"

print("Detecting fake comments...")
results = [classify_fake(i) for i in df.index]
df["fake_label"] = [r[0] for r in results]
df["fake_reason"] = [r[1] for r in results]

df.to_csv("data/fake_comment_results.csv", index=False)

# Summary
fake_count = (df["fake_label"] == "fake").sum()
total = len(df)
print(f"\nFake Comment Detection Complete!")
print(f"Total comments : {total}")
print(f"Fake detected  : {fake_count} ({fake_count/total*100:.1f}%)")
print(f"Real comments  : {total - fake_count}")
print(f"\nFake reasons breakdown:")
print(df[df["fake_label"]=="fake"]["fake_reason"].value_counts())
print(f"\nSaved to data/fake_comment_results.csv")