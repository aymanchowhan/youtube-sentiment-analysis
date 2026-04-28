import pandas as pd
import re

print("Loading dataset...")
df = pd.read_csv("data/final_dataset.csv")

# Spam keyword patterns — covers most YouTube spam
SPAM_PATTERNS = [
    r"subscribe.*channel",
    r"check.*my.*channel",
    r"visit.*my.*page",
    r"follow.*me",
    r"free.*giveaway",
    r"giveaway",
    r"click.*link",
    r"link.*bio",
    r"earn.*money",
    r"make.*money",
    r"work.*from.*home",
    r"sub.*for.*sub",
    r"s4s",
    r"http[s]?://",
    r"www\.",
    r"\.com",
    r"telegram",
    r"whatsapp.*group",
    r"dm.*me",
    r"first.*\d+.*subscribers",
    r"promoting",
]

def is_spam(text):
    text = str(text).lower()
    for pattern in SPAM_PATTERNS:
        if re.search(pattern, text):
            return "spam"
    return "not_spam"

print("Detecting spam...")
df["spam_label"] = df["comment"].apply(is_spam)

# Save full results
df.to_csv("data/spam_results.csv", index=False)

# Summary
spam_count = (df["spam_label"] == "spam").sum()
total = len(df)
print(f"\nSpam Detection Complete!")
print(f"Total comments : {total}")
print(f"Spam detected  : {spam_count} ({spam_count/total*100:.1f}%)")
print(f"Clean comments : {total - spam_count}")
print(f"\nSaved to data/spam_results.csv")