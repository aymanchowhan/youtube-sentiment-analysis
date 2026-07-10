import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Stopwords
stop_words = set(stopwords.words("english"))

# Reusable cleaning function
def clean_comment(text):

    text = str(text)

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    return " ".join(tokens)


# Only run this when executing this file directly
if __name__ == "__main__":

    df = pd.read_csv("data/comments.csv")

    df["clean_comment"] = df["comment"].apply(clean_comment)

    df.to_csv("data/cleaned_comments.csv", index=False)

    print("✅ Text preprocessing completed")