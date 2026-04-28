import pandas as pd
import re

print("Loading dataset...")
df = pd.read_csv("data/fake_comment_results.csv")  # builds on previous

# Sarcasm detection using linguistic patterns
# This is rule-based — appropriate for a Master's project
# (deep learning sarcasm models need labelled training data you don't have)

SARCASM_PATTERNS = [
    # Contradiction patterns
    r"(great|amazing|awesome|love|best|perfect|excellent).{0,30}(but|except|however|yet|still|though)",
    r"(but|except|however|yet|still|though).{0,30}(terrible|worst|bad|horrible|awful|useless|broken|fail)",
    
    # Obvious sarcasm phrases
    r"oh great",
    r"oh sure",
    r"yeah right",
    r"totally (works|fine|great|amazing|helpful)",
    r"wow (so|such|very|really).{0,20}(helpful|useful|great|amazing)",
    r"thanks (for nothing|a lot for nothing)",
    r"just what (i|we) needed",
    
    # Contradiction: positive word + negative outcome
    r"(love|loved|loving).{0,20}(not|never|doesn.t|doesn't|don.t)",
    r"works (perfectly|great|fine).{0,15}(not|never|nope|lol|haha|😂|🤣)",
    
    # Classic sarcasm markers
    r"(sure|obviously|clearly|definitely).{0,20}(not|never)",
    r"what a (great|wonderful|amazing|fantastic).{0,20}(waste|joke|disaster|mess|fail)",
    r"so (helpful|useful|clear|simple|easy).{0,10}(not|😂|🤣|lol|lmao)",
]

def detect_sarcasm(text):
    text = str(text).lower()
    for pattern in SARCASM_PATTERNS:
        if re.search(pattern, text):
            return "sarcastic"
    return "not_sarcastic"

print("Detecting sarcasm...")
df["sarcasm_label"] = df["comment"].apply(detect_sarcasm)

df.to_csv("data/sarcasm_results.csv", index=False)

# Summary
sarcastic_count = (df["sarcasm_label"] == "sarcastic").sum()
total = len(df)
print(f"\nSarcasm Detection Complete!")
print(f"Total comments  : {total}")
print(f"Sarcastic       : {sarcastic_count} ({sarcastic_count/total*100:.1f}%)")
print(f"Not sarcastic   : {total - sarcastic_count}")
print(f"\nExample sarcastic comments found:")
print(df[df["sarcasm_label"]=="sarcastic"]["comment"].head(5).to_string())
print(f"\nSaved to data/sarcasm_results.csv")