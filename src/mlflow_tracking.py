import mlflow
import mlflow.sklearn
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

# Fix for Windows username with spaces
mlflow.set_tracking_uri(
    "file:///C:/Users/Ayman Chowhan/OneDrive/Desktop/youtube-sentiment-analysis/mlruns"
)

print("Starting MLflow experiment tracking...")

# ── Set experiment name ──────────────────────────────
mlflow.set_experiment("YouTube-Sentiment-Analysis")

# ── Load data ────────────────────────────────────────
df       = pd.read_csv("data/final_dataset.csv")
model_df = pd.read_csv("data/model_comparison.csv")
topic_df = pd.read_csv("data/topic_results.csv")

print(f"Dataset loaded: {len(df)} rows")
print(f"Model comparison columns: {model_df.columns.tolist()}")

# ── Helper: log one model run ─────────────────────────
def log_model_run(model_name, accuracy, f1, precision, recall):
    with mlflow.start_run(run_name=model_name):

        # Parameters
        mlflow.log_param("model_name",          model_name)
        mlflow.log_param("dataset_size",         len(df))
        mlflow.log_param("num_videos",           df["video"].nunique())
        mlflow.log_param("num_classes",          3)
        mlflow.log_param("classes",              "positive,negative,neutral")
        mlflow.log_param("evaluation_baseline",  "VADER")

        # Metrics
        mlflow.log_metric("accuracy",   accuracy)
        mlflow.log_metric("f1_score",   f1)
        mlflow.log_metric("precision",  precision)
        mlflow.log_metric("recall",     recall)

        # Dataset metrics
        mlflow.log_metric("total_comments",
            len(df))
        mlflow.log_metric("positive_comments",
            int((df["vader_sentiment"] == "positive").sum()))
        mlflow.log_metric("negative_comments",
            int((df["vader_sentiment"] == "negative").sum()))
        mlflow.log_metric("neutral_comments",
            int((df["vader_sentiment"] == "neutral").sum()))
        mlflow.log_metric("spam_comments",
            int((df["spam_label"] == "spam").sum())
            if "spam_label" in df.columns else 0)
        mlflow.log_metric("topics_extracted",
            len(topic_df))

        print(f"  Logged run: {model_name} | "
              f"Accuracy: {accuracy:.4f} | F1: {f1:.4f}")

# ── Log VADER baseline ────────────────────────────────
print("\nLogging VADER (baseline)...")
log_model_run(
    model_name = "VADER-Baseline",
    accuracy   = 1.0,
    f1         = 1.0,
    precision  = 1.0,
    recall     = 1.0
)

# ── Log BERT and RoBERTa from model_comparison.csv ───
for _, row in model_df.iterrows():
    model_name = str(row["Model"])

    accuracy = float(row["Accuracy"])

    # F1 — handle different possible column names
    f1 = 0.0
    for col in ["F1 Score (weighted)", "F1 Score", "F1_Score",
                "f1_score", "F1"]:
        if col in row.index:
            f1 = float(row[col])
            break

    # Precision
    precision = 0.0
    for col in ["Precision", "precision"]:
        if col in row.index:
            precision = float(row[col])
            break

    # Recall
    recall = 0.0
    for col in ["Recall", "recall"]:
        if col in row.index:
            recall = float(row[col])
            break

    print(f"\nLogging {model_name}...")
    log_model_run(model_name, accuracy, f1, precision, recall)

# ── Full pipeline summary run ─────────────────────────
print("\nLogging full pipeline summary run...")
with mlflow.start_run(run_name="Full-Pipeline-Summary"):

    mlflow.log_param("pipeline_steps",
        "collection,preprocessing,vader,bert,roberta,"
        "emotion,spam,fake,sarcasm,lda_topics,flask_api")
    mlflow.log_param("best_model",      "RoBERTa")
    mlflow.log_param("framework",       "HuggingFace Transformers")
    mlflow.log_param("dashboard",       "Streamlit")
    mlflow.log_param("api",             "Flask REST API")
    mlflow.log_param("deployment",      "Docker + AWS EC2")
    mlflow.log_param("experiment_tracking", "MLflow")

    mlflow.log_metric("best_accuracy",   0.7320)
    mlflow.log_metric("best_f1",         0.7155)
    mlflow.log_metric("bert_accuracy",   0.3973)
    mlflow.log_metric("bert_f1",         0.3630)
    mlflow.log_metric("total_comments",  len(df))
    mlflow.log_metric("total_videos",    df["video"].nunique())
    mlflow.log_metric("topics_per_video", 3)
    mlflow.log_metric("emotion_classes", 7)

    # Save CSVs as artifacts
    mlflow.log_artifact("data/model_comparison.csv")
    mlflow.log_artifact("data/topic_summary.csv")
    mlflow.log_artifact("data/final_dataset.csv")

print("\n" + "="*50)
print("All runs logged successfully!")
print("="*50)
print("\nNow run:")
print('mlflow ui --backend-store-uri "file:///C:/Users/Ayman Chowhan/OneDrive/Desktop/final yt project/mlruns"')
print("\nThen open: http://localhost:5000")