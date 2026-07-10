import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

print("Loading final dataset...")

df = pd.read_csv("data/final_dataset.csv")

# Use VADER as the baseline (ground truth)
ground_truth = df["vader_sentiment"]

# Models to evaluate
models = {
    "BERT": df["bert_sentiment"],
    "RoBERTa": df["roberta_sentiment"],
}

print("\n=== MODEL COMPARISON ===\n")

comparison = []

for model_name, predictions in models.items():

    # Remove missing values
    mask = ground_truth.notna() & predictions.notna()

    y_true = ground_truth[mask]
    y_pred = predictions[mask]

    # Metrics
    acc = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    comparison.append(
        {
            "Model": model_name,
            "Accuracy": round(acc, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1 Score (weighted)": round(f1, 4),
        }
    )

    print(f"--- {model_name} ---")
    print(classification_report(y_true, y_pred, zero_division=0))

# Save comparison table
result_df = pd.DataFrame(comparison)

result_df.to_csv(
    "data/model_comparison.csv",
    index=False,
)

print("\n=== SUMMARY TABLE ===")
print(result_df.to_string(index=False))

print("\nSaved to data/model_comparison.csv")