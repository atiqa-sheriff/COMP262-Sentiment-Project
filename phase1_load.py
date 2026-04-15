import pandas as pd
import os

print("Current directory:", os.getcwd())

df = pd.read_json("Software_5.json", lines=True)

print("Shape (rows, cols):", df.shape)
print("\nColumns:\n", df.columns)
print("\nFirst 3 rows:\n", df.head(3))

print("\n--- BASIC STATS ---")
print("Total reviews:", len(df))
print("Unique users (reviewerID):", df["reviewerID"].nunique())
print("Unique products (asin):", df["asin"].nunique())

print("\n--- MISSING VALUES (top) ---")
print(df.isna().sum().sort_values(ascending=False).head(10))

print("\n--- DUPLICATES ---")
dup_subset = ["reviewerID", "asin", "reviewTime", "overall", "reviewText", "summary"]
print("Duplicate rows (subset):", df.duplicated(subset=dup_subset).sum())

print("\n--- RATINGS DISTRIBUTION (overall) ---")
print(df["overall"].value_counts().sort_index())

import matplotlib.pyplot as plt
rating_counts = df["overall"].value_counts().sort_index()

plt.figure(figsize=(6,4))
rating_counts.plot(kind="bar")
plt.title("Ratings Distribution (Software_5)")
plt.xlabel("Rating (overall)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("ratings_distribution.png", dpi=200)
plt.show()

df["review_length"] = df["reviewText"].astype(str).apply(lambda x: len(x.split()))

print("\n--- REVIEW LENGTH STATS ---")
print("Average length:", df["review_length"].mean())
print("Min length:", df["review_length"].min())
print("Max length:", df["review_length"].max())

plt.figure(figsize=(6,4))
plt.hist(df["review_length"], bins=50)
plt.title("Review Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("review_length_distribution.png", dpi=200)
plt.show()

def label_sentiment(rating):
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"

df["sentiment_label"] = df["overall"].apply(label_sentiment)

print("\n--- SENTIMENT LABEL DISTRIBUTION ---")
print(df["sentiment_label"].value_counts())

df["summary"] = df["summary"].fillna("")
df["reviewText"] = df["reviewText"].fillna("")

df["text"] = df["summary"] + " " + df["reviewText"]

print("\nExample combined text:\n")
print(df["text"].iloc[0][:300]) 

import re


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)  # remove extra whitespace
    return text.strip()

df["clean_text"] = df["text"].apply(preprocess_text)

print("\nExample cleaned text:\n")
print(df["clean_text"].iloc[0][:300])

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def vader_predict(text):
    score = analyzer.polarity_scores(text)["compound"]
    
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df["vader_prediction"] = df["clean_text"].apply(vader_predict)

print("\n--- VADER PREDICTION DISTRIBUTION ---")
print(df["vader_prediction"].value_counts())


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
y_true = df["sentiment_label"]
y_pred = df["vader_prediction"]

print("\n--- VADER ACCURACY ---")
print("Accuracy:", accuracy_score(y_true, y_pred))

print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_true, y_pred))

print("\n--- CONFUSION MATRIX ---")
print(confusion_matrix(y_true, y_pred))

from textblob import TextBlob

def textblob_predict(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

df["textblob_prediction"] = df["clean_text"].apply(textblob_predict)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("TEXTBLOB FULL DATA ACCURACY:")
print(accuracy_score(df["sentiment_label"], df["textblob_prediction"]))

sample_df = df.sample(n=1000, random_state=42)

y_true_sample = sample_df["sentiment_label"]

y_vader_sample = sample_df["vader_prediction"]
y_textblob_sample = sample_df["textblob_prediction"]

print("\nVADER SAMPLE ACCURACY:")
print(accuracy_score(y_true_sample, y_vader_sample))

print("\nVADER SAMPLE CONFUSION MATRIX:")
print(confusion_matrix(y_true_sample, y_vader_sample))

print("\nTEXTBLOB SAMPLE ACCURACY:")
print(accuracy_score(y_true_sample, y_textblob_sample))

print("\nTEXTBLOB SAMPLE CONFUSION MATRIX:")
print(confusion_matrix(y_true_sample, y_textblob_sample))

print("\nVADER CLASSIFICATION REPORT:")
print(classification_report(y_true_sample, y_vader_sample))

print("\nTEXTBLOB CLASSIFICATION REPORT:")
print(classification_report(y_true_sample, y_textblob_sample))