"""
sentiment_utils.py
==================
Model loading & prediction helpers for Insights Inside.
Import this in app.py — do not run directly.
"""

import os
import re
import json
import joblib

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
VECTORIZER_PATH = os.path.join(BASE_DIR, "model", "tfidf_vectorizer.pkl")
MODEL_PATH      = os.path.join(BASE_DIR, "model", "sentiment_model.pkl")
METRICS_PATH    = os.path.join(BASE_DIR, "evaluation", "metrics_summary.json")

# ── UI helpers ─────────────────────────────────────────────────────────────────
LABEL_COLOR = {
    "positive": "#48cfad",
    "negative": "#ff6b6b",
    "neutral":  "#f7b731",
}
LABEL_EMOJI = {
    "positive": "😊",
    "negative": "😞",
    "neutral":  "😐",
}
LABELS = ["positive", "negative", "neutral"]


# ── Text cleaning ──────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Lowercase, strip non-alphanumeric characters, collapse whitespace."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_features(review: str, summary: str = "") -> str:
    """Combine review + summary into a single cleaned feature string."""
    return clean_text(f"{review} {summary}")


# ── Lazy-loaded singletons ─────────────────────────────────────────────────────
_vectorizer = None
_model      = None


def load_artifacts():
    """Load and cache vectorizer + model (called once on first prediction)."""
    global _vectorizer, _model
    if _vectorizer is None:
        _vectorizer = joblib.load(VECTORIZER_PATH)
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        # Backward compatibility for cross-version sklearn pickle loading.
        if not hasattr(_model, "multi_class"):
            _model.multi_class = "auto"
    return _vectorizer, _model


def load_metrics() -> dict:
    """Return metrics_summary.json as a dict, empty dict if not found."""
    if not os.path.exists(METRICS_PATH):
        return {}
    with open(METRICS_PATH) as f:
        return json.load(f)


# ── Prediction ─────────────────────────────────────────────────────────────────
def predict_one(review: str, summary: str = "") -> dict:
    """
    Predict sentiment for a single review.

    Returns
    -------
    dict:
        sentiment     – "positive" | "negative" | "neutral"
        confidence    – float, 0–100
        probabilities – {label: float_pct}
    """
    vectorizer, model = load_artifacts()
    text  = build_features(review, summary)
    vec   = vectorizer.transform([text])
    pred  = model.predict(vec)[0]
    probs = model.predict_proba(vec)[0]

    prob_dict = {
        cls: round(float(p) * 100, 2)
        for cls, p in zip(model.classes_, probs)
    }
    return {
        "sentiment":     pred,
        "confidence":    round(float(max(probs)) * 100, 2),
        "probabilities": prob_dict,
    }


def predict_batch(reviews: list, summaries: list = None) -> list:
    """
    Predict sentiment for a list of reviews in one vectorizer call.

    Parameters
    ----------
    reviews   : list[str]
    summaries : list[str] | None  (same length; defaults to empty strings)

    Returns
    -------
    list[dict]  – same format as predict_one per element
    """
    if summaries is None:
        summaries = [""] * len(reviews)

    vectorizer, model = load_artifacts()
    texts = [build_features(r, s) for r, s in zip(reviews, summaries)]
    vecs  = vectorizer.transform(texts)
    preds = model.predict(vecs)
    probs = model.predict_proba(vecs)

    results = []
    for pred, prob_row in zip(preds, probs):
        prob_dict = {
            cls: round(float(p) * 100, 2)
            for cls, p in zip(model.classes_, prob_row)
        }
        results.append({
            "sentiment":     pred,
            "confidence":    round(float(max(prob_row)) * 100, 2),
            "probabilities": prob_dict,
        })
    return results


def batch_summary(results: list) -> dict:
    """Count each sentiment class from a list of predict_one/batch results."""
    counts = {"positive": 0, "negative": 0, "neutral": 0}
    for r in results:
        counts[r["sentiment"]] = counts.get(r["sentiment"], 0) + 1
    return counts
