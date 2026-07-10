import pandas as pd
import re

print("Loading dataset...")
df = pd.read_csv("data/cleaned_comments.csv")  # run BEFORE merge (avoids circular dependency on final_dataset.csv)

# Sarcasm patterns (text-based + emoji-based for flexibility)
SARCASM_PATTERNS = [
    # Obvious sarcasm markers
    r"oh\s*(sure|great|yeah)",
    r"yeah\s*right",
    r"sure\s*(jan|buddy|thing)",
    
    # Contradiction patterns (positive word + negative)
    r"(love|great|amazing|best|perfect).{0,40}(not|never|except|but|however)",
    r"(not|never|except|but|however).{0,40}(terrible|worst|bad|awful|waste|useless)",
    
    # Sarcasm phrases
    r"thanks\s*(for\s*)?(nothing|a lot|a bunch)",
    r"just\s*what\s*(i|we|you)\s*need(ed)?",
    r"so\s*(helpful|useful|clear|easy|simple)",
    r"what\s*a\s*(great|wonderful|brilliant|fantastic)\s*(waste|joke|fail)",
    
    # Opposite sentiments
    r"(best|greatest|worst|terrible).*ever.*not",
    r"works\s*(perfectly|great|fine).{0,20}(not|nope)",
    
    # Text-based sarcasm markers (replacing emojis)
    r"lol.*not",
    r"haha.*actually",
    r"sure.*right",
    r"yeah.*totally",
    
    # EMOJI PATTERNS (for when emojis are present)
    r"love.*😒",  # love + eye roll
    r"great.*🙄",  # great + eye roll
    r"(best|perfect|amazing).*🤮",  # positive + vomit
    r"(terrible|awful|hate|worst).*😂",  # negative + laughing
    r"(terrible|awful|hate|worst).*😆",  # negative + laughing
    r"thanks.*😤",  # thanks + angry
    r"so\s*helpful.*😤",  # so helpful + angry
    r"(love|great).*😡",  # positive sentiment + angry face
    r"great.*🤡",     # great + clown (mockery)
    r"amazing.*💀",   # amazing + skull (dead/done)
]

def detect_sarcasm(text):
    text = str(text).lower()
    
    # Skip very short comments
    if len(text) < 10:
        return "not_sarcastic"
    
    for pattern in SARCASM_PATTERNS:
        if re.search(pattern, text):
            return "sarcastic"
    
    return "not_sarcastic"

print("Detecting sarcasm...")
df["sarcasm_label"] = df["comment"].apply(detect_sarcasm)

df.to_csv("data/sarcasm_results.csv", index=False)

sarcastic_count = (df["sarcasm_label"] == "sarcastic").sum()
total = len(df)
print(f"\nSarcasm Detection Complete!")
print(f"Total comments  : {total}")
print(f"Sarcastic       : {sarcastic_count} ({sarcastic_count/total*100:.1f}%)")
print(f"Not sarcastic   : {total - sarcastic_count}")
print(f"\nExample sarcastic comments:")
print(df[df["sarcasm_label"]=="sarcastic"]["comment"].head(5).to_string())
print(f"\nSaved to data/sarcasm_results.csv")