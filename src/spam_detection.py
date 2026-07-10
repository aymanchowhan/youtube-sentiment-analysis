import pandas as pd
import re

print("Loading dataset...")
df = pd.read_csv("data/cleaned_comments.csv")  # run BEFORE merge (avoids circular dependency on final_dataset.csv)

# More comprehensive spam patterns
SPAM_PATTERNS = [
    # Self-promotion
    r"subscribe",
    r"check.*channel",
    r"visit.*page",
    r"follow\s*(me|us)",
    r"like.*comment",
    
    # Monetization spam
    r"free.*money",
    r"earn\s*(money|cash|dollars)",
    r"make\s*money",
    r"work.*from.*home",
    r"giveaway",
    r"free\s*(gift|prize)",
    
    # Links and contact
    r"http[s]?://",
    r"www\.",
    r"\.com",
    r"\.io",
    r"telegram",
    r"whatsapp",
    r"discord",
    r"dm\s*me",
    
    # Sub-for-sub engagement
    r"s4s",
    r"sub.*for.*sub",
    r"like.*for.*like",
    
    # Bot indicators
    r"first\s*\d+",
    r"click\s*link",
    r"link\s*in\s*bio",

    # Emoji-based spam indicators (only detectable on original, uncleaned text)
    r"🔥{3,}",        # multiple fire emojis
    r"💯{2,}",        # multiple 100 emojis
    r"!!!!+",          # excessive exclamation
]

def is_spam(text):
    text = str(text).lower().strip()
    
    # Skip very short comments (less likely spam)
    if len(text) < 5:
        return "not_spam"
    
    # Check patterns
    for pattern in SPAM_PATTERNS:
        if re.search(pattern, text):
            return "spam"
    
    return "not_spam"

print("Detecting spam...")
df["spam_label"] = df["comment"].apply(is_spam)

df.to_csv("data/spam_results.csv", index=False)

spam_count = (df["spam_label"] == "spam").sum()
total = len(df)
print(f"\nSpam Detection Complete!")
print(f"Total comments : {total}")
print(f"Spam detected  : {spam_count} ({spam_count/total*100:.1f}%)")
print(f"Clean comments : {total - spam_count}")
print(f"\nExample spam comments:")
print(df[df["spam_label"]=="spam"]["comment"].head(5).to_string())
print(f"\nSaved to data/spam_results.csv")