"""
train_model.py
--------------
Training script for Insights Inside sentiment analysis model.

Split strategy : 70 % train / 15 % validation / 15 % test
                 (stratified on Sentiment to preserve class ratios)

Outputs saved to:
  model/tfidf_vectorizer.pkl
  model/sentiment_model.pkl
  evaluation/metrics_summary.json
  evaluation/classification_report.txt
  evaluation/confusion_matrix.png
  evaluation/metrics_chart.png
"""

import os
import re
import json
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Dataset-SA.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
EVAL_DIR  = os.path.join(BASE_DIR, "evaluation")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EVAL_DIR,  exist_ok=True)

LABELS = ["positive", "negative", "neutral"]


# ── 1. Text cleaning ───────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── 2. Load & preprocess data ─────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    print(f"[1/7] Loading dataset from {path} ...")
    df = pd.read_csv(path)
    print(f"      Raw shape: {df.shape}")

    # Keep only valid sentiments
    df = df[df["Sentiment"].isin(LABELS)].copy()

    # Combine Review + Summary into a single feature
    df["text"] = (
        df["Review"].fillna("") + " " + df["Summary"].fillna("")
    ).apply(clean_text)

    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    print(f"      After cleaning: {df.shape}")
    print(f"      Class distribution:\n{df['Sentiment'].value_counts()}\n")
    return df


# ── 3. Split  70 / 15 / 15 ────────────────────────────────────────────────────
def split_data(df: pd.DataFrame):
    print("[2/7] Splitting data  70 % train / 15 % val / 15 % test (stratified) ...")

    X = df["text"].values
    y = df["Sentiment"].values

    # First cut: 70 % train, 30 % temp  (stratified)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # Second cut: temp → 50/50 → 15 % val, 15 % test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"      Train : {len(X_train):>7,}")
    print(f"      Val   : {len(X_val):>7,}")
    print(f"      Test  : {len(X_test):>7,}\n")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ── 4. Vectorise ──────────────────────────────────────────────────────────────
def build_vectorizer(X_train, X_val, X_test):
    print("[3/7] Fitting TF-IDF vectorizer (bigrams, max_features=60,000) ...")
    vectorizer = TfidfVectorizer(
        max_features=60_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        strip_accents="unicode",
        analyzer="word",
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec   = vectorizer.transform(X_val)
    X_test_vec  = vectorizer.transform(X_test)
    print(f"      Vocabulary size: {len(vectorizer.vocabulary_):,}\n")
    return vectorizer, X_train_vec, X_val_vec, X_test_vec


# ── 5. Train ──────────────────────────────────────────────────────────────────
def train(X_train_vec, y_train):
    print("[4/7] Training Logistic Regression (C=1.0, class_weight='balanced') ...")
    t0 = time.time()
    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_vec, y_train)
    print(f"      Done in {time.time()-t0:.1f}s\n")
    return model


# ── 6. Evaluate ───────────────────────────────────────────────────────────────
def evaluate(model, vectorizer,
             X_train_vec, y_train,
             X_val_vec,   y_val,
             X_test_vec,  y_test):

    print("[5/7] Evaluating on val & test sets ...")

    val_preds  = model.predict(X_val_vec)
    test_preds = model.predict(X_test_vec)

    val_acc   = accuracy_score(y_val,  val_preds)
    test_acc  = accuracy_score(y_test, test_preds)

    print(f"      Val  accuracy : {val_acc*100:.2f}%")
    print(f"      Test accuracy : {test_acc*100:.2f}%")

    # Per-class metrics on test set
    report_str  = classification_report(y_test, test_preds, target_names=LABELS)
    report_dict = classification_report(y_test, test_preds, target_names=LABELS, output_dict=True)
    print(f"\n{report_str}")

    # 5-fold cross-val on training data (fast, uses existing vectorised matrix)
    print("      Running 5-fold CV on training split ...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_vec, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"      CV scores : {cv_scores.round(4)}")
    print(f"      CV mean   : {cv_scores.mean():.4f}  ±{cv_scores.std():.4f}\n")

    # Metrics summary dict
    metrics = {
        "split_strategy": "70/15/15 (stratified)",
        "train_samples":  int(X_train_vec.shape[0]),
        "val_samples":    int(X_val_vec.shape[0]),
        "test_samples":   int(X_test_vec.shape[0]),
        "val_accuracy":   round(val_acc,  4),
        "test_accuracy":  round(test_acc, 4),
        "cv_mean":        round(cv_scores.mean(), 4),
        "cv_std":         round(cv_scores.std(),  4),
        "cv_scores":      cv_scores.round(4).tolist(),
        "per_class": {
            lbl: {
                "precision": round(report_dict[lbl]["precision"], 4),
                "recall":    round(report_dict[lbl]["recall"],    4),
                "f1":        round(report_dict[lbl]["f1-score"],  4),
                "support":   int(report_dict[lbl]["support"]),
            }
            for lbl in LABELS
        },
        "macro_f1":    round(report_dict["macro avg"]["f1-score"],    4),
        "weighted_f1": round(report_dict["weighted avg"]["f1-score"], 4),
    }

    return metrics, report_str, test_preds


# ── 7. Save artefacts ──────────────────────────────────────────────────────────
def save_artifacts(model, vectorizer, metrics, report_str, y_test, test_preds):
    print("[6/7] Saving model artefacts ...")

    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), compress=3)
    joblib.dump(model,      os.path.join(MODEL_DIR, "sentiment_model.pkl"),  compress=3)
    print("      model/tfidf_vectorizer.pkl  ✓")
    print("      model/sentiment_model.pkl   ✓")

    with open(os.path.join(EVAL_DIR, "metrics_summary.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("      evaluation/metrics_summary.json  ✓")

    with open(os.path.join(EVAL_DIR, "classification_report.txt"), "w") as f:
        f.write("Classification Report — Test Set\n")
        f.write("=" * 50 + "\n\n")
        f.write(report_str)
        f.write("\n\nAdditional Metrics\n")
        f.write("=" * 50 + "\n")
        f.write(f"Val  Accuracy : {metrics['val_accuracy']*100:.2f}%\n")
        f.write(f"Test Accuracy : {metrics['test_accuracy']*100:.2f}%\n")
        f.write(f"CV Mean       : {metrics['cv_mean']*100:.2f}% ± {metrics['cv_std']*100:.2f}%\n")
    print("      evaluation/classification_report.txt  ✓")

    _plot_confusion_matrix(y_test, test_preds)
    _plot_metrics_chart(metrics)
    print("      evaluation/confusion_matrix.png  ✓")
    print("      evaluation/metrics_chart.png     ✓\n")


def _plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABELS, yticklabels=LABELS,
        linewidths=0.5, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual",    fontsize=12)
    ax.set_title("Confusion Matrix — Test Set", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(EVAL_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)


def _plot_metrics_chart(metrics: dict):
    labels     = LABELS
    precisions = [metrics["per_class"][l]["precision"] for l in labels]
    recalls    = [metrics["per_class"][l]["recall"]    for l in labels]
    f1s        = [metrics["per_class"][l]["f1"]        for l in labels]

    x    = np.arange(len(labels))
    w    = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))

    bars_p = ax.bar(x - w, precisions, w, label="Precision", color="#6c63ff", alpha=0.85)
    bars_r = ax.bar(x,     recalls,    w, label="Recall",    color="#48cfad", alpha=0.85)
    bars_f = ax.bar(x + w, f1s,        w, label="F1-Score",  color="#ff6b6b", alpha=0.85)

    for bars in (bars_p, bars_r, bars_f):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.005,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([l.capitalize() for l in labels], fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        f"Per-Class Metrics — Test Accuracy {metrics['test_accuracy']*100:.2f}%",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_facecolor("#f9f9fc")
    fig.patch.set_facecolor("#ffffff")
    plt.tight_layout()
    fig.savefig(os.path.join(EVAL_DIR, "metrics_chart.png"), dpi=150)
    plt.close(fig)


# ── 8. Summary ────────────────────────────────────────────────────────────────
def print_summary(metrics: dict):
    print("[7/7] ── FINAL SUMMARY ─────────────────────────────────")
    print(f"  Split          : {metrics['split_strategy']}")
    print(f"  Train samples  : {metrics['train_samples']:,}")
    print(f"  Val   samples  : {metrics['val_samples']:,}")
    print(f"  Test  samples  : {metrics['test_samples']:,}")
    print(f"  Val  Accuracy  : {metrics['val_accuracy']*100:.2f}%")
    print(f"  Test Accuracy  : {metrics['test_accuracy']*100:.2f}%")
    print(f"  CV Mean        : {metrics['cv_mean']*100:.2f}% ± {metrics['cv_std']*100:.2f}%")
    print(f"  Macro F1       : {metrics['macro_f1']*100:.2f}%")
    print(f"  Weighted F1    : {metrics['weighted_f1']*100:.2f}%")
    print("  Per-class F1:")
    for lbl in LABELS:
        f1 = metrics["per_class"][lbl]["f1"]
        print(f"    {lbl:10s}: {f1*100:.2f}%")
    print("────────────────────────────────────────────────────────")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("   Insights Inside — Sentiment Model Training")
    print("="*55 + "\n")

    df = load_data(DATA_PATH)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    vectorizer, X_train_vec, X_val_vec, X_test_vec = build_vectorizer(
        X_train, X_val, X_test
    )

    model = train(X_train_vec, y_train)

    metrics, report_str, test_preds = evaluate(
        model, vectorizer,
        X_train_vec, y_train,
        X_val_vec,   y_val,
        X_test_vec,  y_test,
    )

    save_artifacts(model, vectorizer, metrics, report_str, y_test, test_preds)
    print_summary(metrics)

    print("\nTraining complete. All files saved.\n")
