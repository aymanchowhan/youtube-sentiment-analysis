import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load dataset
df = pd.read_csv("data/comments.csv")

# Stopwords
stop_words = set(stopwords.words("english"))

# Cleaning function
def clean_comment(text):

    text = str(text)

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove special characters but keep emojis
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # Join words
    cleaned_text = " ".join(tokens)

    return cleaned_text


# Apply cleaning
df["clean_comment"] = df["comment"].apply(clean_comment)

# Save cleaned data
df.to_csv("data/cleaned_comments.csv", index=False)

print("✅ Text preprocessing completed")