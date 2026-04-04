"""
app.py  —  Insights Inside · Product Sentiment Analysis
Streamlit application with full analytics dashboard for batch analysis.
"""

import io
import os
import re
import json
import joblib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import streamlit as st

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Insights Inside — Sentiment Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE      = os.path.dirname(os.path.abspath(__file__))
VEC_PATH  = os.path.join(BASE, "model", "tfidf_vectorizer.pkl")
MDL_PATH  = os.path.join(BASE, "model", "sentiment_model.pkl")
EVAL_PATH = os.path.join(BASE, "evaluation", "metrics_summary.json")
SAMPLE_DATA_PATH = os.path.join(BASE, "sample_data")

LABELS = ["positive", "negative", "neutral"]
C      = {"positive": "#48cfad", "negative": "#ff6b6b", "neutral": "#f7b731"}
EMO    = {"positive": "😊", "negative": "😞", "neutral": "😐"}

# dark palette for matplotlib
PLT_BG   = "#1a1a2e"
PLT_FG   = "#e0e0e0"
PLT_GRID = "#2a2a4a"

# ══════════════════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:#0f0f1a;color:#e0e0e0;}
#MainMenu,footer{visibility:hidden;}
.block-container{padding-top:0!important;max-width:1100px;}

/* ── hero ── */
.hero{
  background:linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);
  border-radius:0 0 28px 28px;
  padding:52px 32px 38px;text-align:center;
  margin-bottom:26px;border:1px solid #2a2a4a;
}
.hero h1{
  font-size:2.6rem;font-weight:800;margin-bottom:10px;
  background:linear-gradient(90deg,#6c63ff,#48cfad,#ff6b6b);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
}
.hero p{color:#aaa;font-size:1.05rem;max-width:600px;margin:0 auto;}

/* ── stat cards ── */
.stat-row{display:flex;gap:12px;margin-bottom:24px;flex-wrap:wrap;}
.stat-card{
  flex:1;min-width:110px;
  background:#1a1a2e;border:1px solid #2a2a4a;
  border-radius:14px;padding:16px 14px;text-align:center;
}
.stat-card .val{font-size:1.45rem;font-weight:800;}
.stat-card .lbl{font-size:0.7rem;color:#888;text-transform:uppercase;
                letter-spacing:1px;margin-top:4px;}

/* ── section card ── */
.card{
  background:#1a1a2e;border:1px solid #2a2a4a;
  border-radius:16px;padding:26px;margin-bottom:20px;
}
.card-title{font-size:1.05rem;font-weight:700;color:#ccc;margin-bottom:14px;
            border-bottom:1px solid #2a2a4a;padding-bottom:10px;}

/* ── result badge ── */
.result-wrap{text-align:center;padding:18px 0 10px;}
.result-badge{
  display:inline-block;padding:10px 30px;
  border-radius:50px;font-size:1.3rem;font-weight:700;margin-bottom:6px;
}

/* ── prob bars ── */
.prob-row{display:flex;align-items:center;gap:10px;margin:7px 0;}
.prob-label{width:80px;font-size:0.84rem;color:#ccc;text-align:right;}
.prob-bar-wrap{flex:1;background:#2a2a4a;border-radius:6px;height:11px;overflow:hidden;}
.prob-bar{height:11px;border-radius:6px;transition:width .4s;}
.prob-pct{width:50px;font-size:0.82rem;color:#aaa;}

/* ── metric pill ── */
.metric-pill{
  display:inline-block;padding:4px 14px;border-radius:20px;
  font-size:0.82rem;font-weight:600;margin:3px;
}

/* ── preview table rows ── */
.prev-row{
  display:flex;align-items:center;gap:10px;
  padding:9px 14px;border-radius:10px;
  background:#12122a;margin-bottom:6px;border:1px solid #2a2a4a;
  font-size:0.84rem;
}
.prev-idx{color:#555;min-width:28px;font-size:0.78rem;}
.prev-sent{font-weight:700;min-width:90px;}
.prev-conf{color:#888;min-width:60px;}
.prev-text{color:#bbb;flex:1;overflow:hidden;
           white-space:nowrap;text-overflow:ellipsis;}

/* ── streamlit overrides ── */
.stTextArea textarea,.stTextInput input{
  background:#12122a!important;color:#e0e0e0!important;
  border:1px solid #3a3a5a!important;border-radius:10px!important;
}
.stButton>button{
  background:linear-gradient(135deg,#6c63ff,#48cfad)!important;
  color:#fff!important;border:none!important;
  border-radius:10px!important;font-weight:600!important;padding:10px 26px!important;
}
.stTabs [data-baseweb="tab-list"]{background:#1a1a2e;border-radius:12px;padding:4px;}
.stTabs [data-baseweb="tab"]{color:#888!important;}
.stTabs [aria-selected="true"]{background:#6c63ff!important;border-radius:8px;color:#fff!important;}
div[data-testid="stMetric"]{background:#1a1a2e;border:1px solid #2a2a4a;
  border-radius:12px;padding:14px;}
div[data-testid="stMetricLabel"]{color:#888!important;font-size:0.75rem!important;}
div[data-testid="stMetricValue"]{color:#e0e0e0!important;font-size:1.4rem!important;font-weight:700!important;}

/* ── sidebar polish ── */
section[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#0f0f1a 0%,#151528 100%);
    border-right:1px solid #2a2a4a;
}
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3{
    color:#d8d8f7;
}
section[data-testid="stSidebar"] .stExpander{
    border:1px solid #2a2a4a;border-radius:12px;background:#17172b;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    vec = joblib.load(VEC_PATH)
    mdl = joblib.load(MDL_PATH)
    # Backward compatibility: older pickled sklearn models may miss attrs
    # expected by newer sklearn versions during predict_proba.
    if not hasattr(mdl, "multi_class"):
        mdl.multi_class = "auto"
    return vec, mdl

@st.cache_data
def load_metrics():
    if os.path.exists(EVAL_PATH):
        with open(EVAL_PATH) as f:
            return json.load(f)
    return {}

vectorizer, model = load_model()
M = load_metrics()


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def clean(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def predict_one(review, summary=""):
    t    = clean(f"{review} {summary}")
    vec  = vectorizer.transform([t])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    cls  = model.classes_
    return pred, {c: round(float(p)*100, 2) for c, p in zip(cls, prob)}

def predict_batch(reviews, summaries=None):
    if summaries is None:
        summaries = [""] * len(reviews)
    texts = [clean(f"{r} {s}") for r, s in zip(reviews, summaries)]
    vecs  = vectorizer.transform(texts)
    preds = model.predict(vecs)
    probs = model.predict_proba(vecs)
    cls   = model.classes_
    out   = []
    for pred, prob in zip(preds, probs):
        pd_   = {c: round(float(p)*100, 2) for c, p in zip(cls, prob)}
        conf  = round(float(max(prob))*100, 2)
        out.append({"sentiment": pred, "confidence": conf, "probabilities": pd_})
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  MATPLOTLIB THEME HELPER
# ══════════════════════════════════════════════════════════════════════════════
def dark_fig(w=7, h=4.5):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=PLT_BG)
    ax.set_facecolor(PLT_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(PLT_GRID)
    ax.tick_params(colors=PLT_FG)
    ax.xaxis.label.set_color(PLT_FG)
    ax.yaxis.label.set_color(PLT_FG)
    ax.title.set_color(PLT_FG)
    return fig, ax

def fig_to_buf(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return buf


# ══════════════════════════════════════════════════════════════════════════════
#  CHART BUILDERS
# ══════════════════════════════════════════════════════════════════════════════
def chart_pie(counts: dict):
    vals   = [counts.get(l, 0) for l in LABELS]
    colors = [C[l] for l in LABELS]
    labels = [f"{l.capitalize()}\n{counts.get(l,0):,}" for l in LABELS]

    fig, ax = plt.subplots(figsize=(5.5, 4.5), facecolor=PLT_BG)
    wedges, texts, autotexts = ax.pie(
        vals, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=140,
        wedgeprops=dict(edgecolor="#0f0f1a", linewidth=2),
        textprops=dict(color=PLT_FG, fontsize=10),
    )
    for at in autotexts:
        at.set_fontsize(9); at.set_color("#0f0f1a"); at.set_fontweight("bold")
    ax.set_title("Sentiment Distribution", color=PLT_FG, fontsize=13, fontweight="bold", pad=12)
    fig.patch.set_facecolor(PLT_BG)
    return fig_to_buf(fig)


def chart_bar_counts(counts: dict):
    fig, ax = dark_fig(5.5, 4)
    vals  = [counts.get(l, 0) for l in LABELS]
    bars  = ax.bar(LABELS, vals, color=[C[l] for l in LABELS],
                   edgecolor="#0f0f1a", linewidth=1.5, width=0.55)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02,
                f"{v:,}", ha="center", va="bottom", color=PLT_FG, fontsize=10, fontweight="bold")
    ax.set_xticklabels([l.capitalize() for l in LABELS], fontsize=11, color=PLT_FG)
    ax.set_ylabel("Count", color=PLT_FG, fontsize=10)
    ax.set_title("Reviews per Sentiment", color=PLT_FG, fontsize=13, fontweight="bold")
    ax.yaxis.grid(True, color=PLT_GRID, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    return fig_to_buf(fig)


def chart_confidence_hist(df_res: pd.DataFrame):
    fig, ax = dark_fig(6.5, 4)
    for lbl in LABELS:
        sub = df_res[df_res["Sentiment"] == lbl]["Confidence_%"]
        if len(sub):
            ax.hist(sub, bins=20, alpha=0.70, color=C[lbl],
                    label=lbl.capitalize(), edgecolor="#0f0f1a", linewidth=0.5)
    ax.set_xlabel("Confidence %", color=PLT_FG, fontsize=10)
    ax.set_ylabel("Count",        color=PLT_FG, fontsize=10)
    ax.set_title("Confidence Score Distribution by Sentiment",
                 color=PLT_FG, fontsize=12, fontweight="bold")
    ax.legend(facecolor=PLT_BG, edgecolor=PLT_GRID,
              labelcolor=PLT_FG, fontsize=9)
    ax.yaxis.grid(True, color=PLT_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    return fig_to_buf(fig)


def chart_avg_confidence(df_res: pd.DataFrame):
    fig, ax = dark_fig(5.5, 4)
    avgs  = [df_res[df_res["Sentiment"] == l]["Confidence_%"].mean() for l in LABELS]
    bars  = ax.bar(LABELS, avgs, color=[C[l] for l in LABELS],
                   edgecolor="#0f0f1a", linewidth=1.5, width=0.55)
    for bar, v in zip(bars, avgs):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{v:.1f}%", ha="center", va="bottom",
                    color=PLT_FG, fontsize=10, fontweight="bold")
    ax.set_xticklabels([l.capitalize() for l in LABELS], fontsize=11, color=PLT_FG)
    ax.set_ylabel("Avg Confidence %", color=PLT_FG, fontsize=10)
    ax.set_ylim(0, 110)
    ax.set_title("Average Confidence per Sentiment",
                 color=PLT_FG, fontsize=12, fontweight="bold")
    ax.yaxis.grid(True, color=PLT_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    return fig_to_buf(fig)


def chart_stacked_probs(df_res: pd.DataFrame, max_rows=40):
    # sample for readability
    sample = df_res.head(max_rows).reset_index(drop=True)
    pos_p  = sample.get("Prob_Positive_%", pd.Series([0]*len(sample))).fillna(0)
    neg_p  = sample.get("Prob_Negative_%", pd.Series([0]*len(sample))).fillna(0)
    neu_p  = sample.get("Prob_Neutral_%",  pd.Series([0]*len(sample))).fillna(0)
    idx    = np.arange(len(sample))

    fig, ax = dark_fig(max(7, len(sample)*0.35), 4)
    ax.bar(idx, pos_p, color=C["positive"], label="Positive", alpha=0.85)
    ax.bar(idx, neg_p, bottom=pos_p, color=C["negative"], label="Negative", alpha=0.85)
    ax.bar(idx, neu_p, bottom=pos_p+neg_p, color=C["neutral"], label="Neutral", alpha=0.85)
    ax.set_xlabel(f"Review index (first {len(sample)})", color=PLT_FG, fontsize=9)
    ax.set_ylabel("Probability %", color=PLT_FG, fontsize=9)
    ax.set_title("Stacked Probability per Review",
                 color=PLT_FG, fontsize=12, fontweight="bold")
    ax.legend(facecolor=PLT_BG, edgecolor=PLT_GRID, labelcolor=PLT_FG, fontsize=9)
    ax.yaxis.grid(True, color=PLT_GRID, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    return fig_to_buf(fig)


def chart_model_metrics():
    """Bar chart of model-level precision / recall / F1 per class."""
    pc   = M.get("per_class", {})
    keys = [k for k in LABELS if k in pc]
    prec = [pc[k]["precision"]*100 for k in keys]
    rec  = [pc[k]["recall"]*100    for k in keys]
    f1   = [pc[k]["f1"]*100        for k in keys]

    x, w = np.arange(len(keys)), 0.24
    fig, ax = dark_fig(7.5, 4.5)
    b1 = ax.bar(x - w, prec, w, color="#6c63ff", alpha=0.85, label="Precision")
    b2 = ax.bar(x,     rec,  w, color="#48cfad", alpha=0.85, label="Recall")
    b3 = ax.bar(x + w, f1,   w, color="#ff6b6b", alpha=0.85, label="F1-Score")
    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+0.6,
                    f"{h:.1f}", ha="center", va="bottom",
                    color=PLT_FG, fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([k.capitalize() for k in keys], fontsize=11, color=PLT_FG)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Score %", color=PLT_FG, fontsize=10)
    ax.set_title("Model Metrics — Precision / Recall / F1 per Class",
                 color=PLT_FG, fontsize=12, fontweight="bold")
    ax.legend(facecolor=PLT_BG, edgecolor=PLT_GRID, labelcolor=PLT_FG, fontsize=9)
    ax.yaxis.grid(True, color=PLT_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    return fig_to_buf(fig)


def chart_cv_scores():
    scores = M.get("cv_scores", [])
    if not scores:
        return None
    fig, ax = dark_fig(6, 3.8)
    folds = [f"Fold {i+1}" for i in range(len(scores))]
    bars  = ax.bar(folds, [s*100 for s in scores],
                   color="#6c63ff", alpha=0.85, edgecolor="#0f0f1a", width=0.5)
    mn = M.get("cv_mean", 0)*100
    ax.axhline(mn, color="#48cfad", linewidth=2, linestyle="--",
               label=f"Mean {mn:.2f}%")
    for bar, s in zip(bars, scores):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                f"{s*100:.2f}%", ha="center", va="bottom",
                color=PLT_FG, fontsize=9)
    ax.set_ylim(min(s*100 for s in scores) - 1, max(s*100 for s in scores) + 2)
    ax.set_ylabel("Accuracy %", color=PLT_FG, fontsize=10)
    ax.set_title("5-Fold Cross-Validation Accuracy",
                 color=PLT_FG, fontsize=12, fontweight="bold")
    ax.legend(facecolor=PLT_BG, edgecolor=PLT_GRID, labelcolor=PLT_FG, fontsize=9)
    ax.yaxis.grid(True, color=PLT_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    return fig_to_buf(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  EXCEL EXPORT
# ══════════════════════════════════════════════════════════════════════════════
def build_excel(df_res: pd.DataFrame, counts: dict) -> bytes:
    buf = io.BytesIO()
    rows = [
        ("Total Reviews", len(df_res)),
        ("Positive", counts.get("positive", 0)),
        ("Negative", counts.get("negative", 0)),
        ("Neutral", counts.get("neutral", 0)),
        ("Positive %", f"{counts.get('positive',0)/max(len(df_res),1)*100:.1f}%"),
        ("Negative %", f"{counts.get('negative',0)/max(len(df_res),1)*100:.1f}%"),
        ("Neutral %", f"{counts.get('neutral',0)/max(len(df_res),1)*100:.1f}%"),
        ("Avg Confidence (All)", f"{df_res['Confidence_%'].mean():.2f}%"),
        ("Avg Confidence (Positive)", f"{df_res[df_res['Sentiment']=='positive']['Confidence_%'].mean():.2f}%" if counts.get('positive') else "N/A"),
        ("Avg Confidence (Negative)", f"{df_res[df_res['Sentiment']=='negative']['Confidence_%'].mean():.2f}%" if counts.get('negative') else "N/A"),
        ("Avg Confidence (Neutral)", f"{df_res[df_res['Sentiment']=='neutral']['Confidence_%'].mean():.2f}%" if counts.get('neutral') else "N/A"),
        ("Model Test Accuracy", f"{M.get('test_accuracy',0)*100:.2f}%"),
        ("Model CV Mean", f"{M.get('cv_mean',0)*100:.2f}%"),
        ("Model Macro F1", f"{M.get('macro_f1',0)*100:.2f}%"),
        ("Model Weighted F1", f"{M.get('weighted_f1',0)*100:.2f}%"),
    ]

    try:
        import xlsxwriter  # noqa: F401

        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            wb = writer.book

            # ── formats
            hdr_fmt = wb.add_format({"bold": True, "bg_color": "#1a1a2e",
                                      "font_color": "#e0e0e0", "border": 1,
                                      "align": "center"})

            # ── Sheet 1: Results
            df_res.to_excel(writer, sheet_name="Results", index=False)
            ws = writer.sheets["Results"]
            for col_num, col_name in enumerate(df_res.columns):
                ws.write(0, col_num, col_name, hdr_fmt)
                ws.set_column(col_num, col_num, max(14, len(col_name)+4))
            ws.set_column(0, 0, 55)   # Review column wider

            # ── Sheet 2: Summary
            ws2 = wb.add_worksheet("Summary")
            ws2.write(0, 0, "Metric", hdr_fmt)
            ws2.write(0, 1, "Value", hdr_fmt)
            for i, (k, v) in enumerate(rows, 1):
                ws2.write(i, 0, k)
                ws2.write(i, 1, str(v))
            ws2.set_column(0, 0, 28)
            ws2.set_column(1, 1, 18)
    except ModuleNotFoundError:
        # Fallback when xlsxwriter is unavailable in the runtime environment.
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df_res.to_excel(writer, sheet_name="Results", index=False)
            pd.DataFrame(rows, columns=["Metric", "Value"]).to_excel(
                writer,
                sheet_name="Summary",
                index=False,
            )

    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <h1>Insights Inside</h1>
  <p>Professional sentiment intelligence workspace for product review analysis.</p>
</div>
""", unsafe_allow_html=True)

acc = M.get("test_accuracy", 0)
cvm = M.get("cv_mean", 0)
mf1 = M.get("macro_f1", 0)
wf1 = M.get("weighted_f1", 0)
tot = M.get("train_samples", 0) + M.get("val_samples", 0) + M.get("test_samples", 0)

st.markdown(f"""
<div class="stat-row">
  <div class="stat-card"><div class="val" style="color:#6c63ff;">{acc*100:.2f}%</div><div class="lbl">Test Accuracy</div></div>
  <div class="stat-card"><div class="val" style="color:#48cfad;">{cvm*100:.2f}%</div><div class="lbl">CV Accuracy</div></div>
  <div class="stat-card"><div class="val" style="color:#f7b731;">{mf1*100:.2f}%</div><div class="lbl">Macro F1</div></div>
  <div class="stat-card"><div class="val" style="color:#ff6b6b;">{wf1*100:.2f}%</div><div class="lbl">Weighted F1</div></div>
  <div class="stat-card"><div class="val" style="color:#aaa;">{tot:,}</div><div class="lbl">Reviews Trained</div></div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — MODEL / CONTROLS
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## Model Control Center")
    st.caption("All model diagnostics and analysis controls live here.")

    v_ok = os.path.exists(VEC_PATH)
    m_ok = os.path.exists(MDL_PATH)
    e_ok = os.path.exists(EVAL_PATH)
    st.success("Model artifacts loaded" if (v_ok and m_ok) else "Model artifact missing")
    st.caption(f"Vectorizer: {'OK' if v_ok else 'Missing'} | Model: {'OK' if m_ok else 'Missing'} | Metrics: {'OK' if e_ok else 'Missing'}")

    c1, c2 = st.columns(2)
    c1.metric("Test Acc", f"{acc*100:.2f}%")
    c2.metric("Macro F1", f"{mf1*100:.2f}%")

    with st.expander("Per-Class Metrics", expanded=True):
        for lbl in LABELS:
            info = M.get("per_class", {}).get(lbl, {})
            st.markdown(
                f"<span style='color:{C[lbl]};font-weight:700'>{EMO[lbl]} {lbl.capitalize()}</span>"
                f"<br><small>Precision {info.get('precision', 0)*100:.1f}%"
                f" | Recall {info.get('recall', 0)*100:.1f}%"
                f" | F1 {info.get('f1', 0)*100:.1f}%</small>",
                unsafe_allow_html=True,
            )

    with st.expander("Analysis Filters", expanded=True):
        min_conf = st.slider("Minimum confidence", 0, 100, 0, 5)
        sent_filter = st.multiselect("Sentiments", LABELS, default=LABELS)
        preview_rows = st.slider("Preview rows", 5, 30, 12)

    with st.expander("Training Configuration"):
        st.markdown(f"""
        | Setting | Value |
        |---|---|
        | Split strategy | 70 / 15 / 15 (stratified) |
        | Train samples  | {M.get('train_samples', 0):,} |
        | Val samples    | {M.get('val_samples', 0):,} |
        | Test samples   | {M.get('test_samples', 0):,} |
        | Algorithm      | Logistic Regression |
        | Features       | TF-IDF bigrams (60k vocab) |
        """)


# cache latest run so the dashboard stays visible after widget interactions
if "df_full" not in st.session_state:
    st.session_state["df_full"] = None


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN WORKSPACE
# ══════════════════════════════════════════════════════════════════════════════
t_workspace, t_single, t_about = st.tabs(["Batch Workspace", "Single Review", "About"])

with t_workspace:
    st.markdown('<div class="card"><div class="card-title">1) Upload or Paste Reviews</div>', unsafe_allow_html=True)

    input_mode = st.radio(
        "Input source",
        ["Upload CSV", "Upload Excel", "Sample data folder", "Paste text lines"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if input_mode == "Sample data folder":
        st.info(
            "How to use sample data: place one or more .csv/.xlsx files in the project's "
            "sample_data folder, then select the file below and run analysis."
        )

    reviews_raw, summaries_raw = [], []
    src_name = ""

    if input_mode == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
        if uploaded:
            df_up = pd.read_csv(uploaded)
            src_name = uploaded.name
            cols = list(df_up.columns)
            rev_guess = next((c for c in cols if c.lower() == "review"), cols[0])
            sum_guess = next((c for c in cols if c.lower() == "summary"), "(none)")
            col1, col2 = st.columns(2)
            review_col = col1.selectbox("Review column", cols, index=cols.index(rev_guess))
            summary_opts = ["(none)"] + cols
            summary_col = col2.selectbox(
                "Summary column",
                summary_opts,
                index=summary_opts.index(sum_guess) if sum_guess in summary_opts else 0,
            )
            reviews_raw = df_up[review_col].fillna("").astype(str).tolist()
            if summary_col != "(none)":
                summaries_raw = df_up[summary_col].fillna("").astype(str).tolist()

    elif input_mode == "Upload Excel":
        uploaded = st.file_uploader("Upload Excel", type=["xlsx"], label_visibility="collapsed")
        if uploaded:
            xls = pd.ExcelFile(uploaded)
            sheet = st.selectbox("Sheet", xls.sheet_names)
            df_up = pd.read_excel(xls, sheet_name=sheet)
            src_name = f"{uploaded.name} · {sheet}"
            cols = list(df_up.columns)
            if cols:
                rev_guess = next((c for c in cols if c.lower() == "review"), cols[0])
                sum_guess = next((c for c in cols if c.lower() == "summary"), "(none)")
                col1, col2 = st.columns(2)
                review_col = col1.selectbox("Review column", cols, index=cols.index(rev_guess), key="xl_review_col")
                summary_opts = ["(none)"] + cols
                summary_col = col2.selectbox(
                    "Summary column",
                    summary_opts,
                    index=summary_opts.index(sum_guess) if sum_guess in summary_opts else 0,
                    key="xl_summary_col",
                )
                reviews_raw = df_up[review_col].fillna("").astype(str).tolist()
                if summary_col != "(none)":
                    summaries_raw = df_up[summary_col].fillna("").astype(str).tolist()

    elif input_mode == "Sample data folder":
        if not os.path.exists(SAMPLE_DATA_PATH):
            st.info("Create a sample_data folder in the project root to use this mode.")
        else:
            sample_files = [
                f for f in os.listdir(SAMPLE_DATA_PATH)
                if f.lower().endswith((".csv", ".xlsx")) and os.path.isfile(os.path.join(SAMPLE_DATA_PATH, f))
            ]
            if not sample_files:
                st.info("No CSV/XLSX files found in sample_data yet.")
            else:
                sample_files.sort()
                selected_file = st.selectbox("Choose sample file", sample_files)
                sample_path = os.path.join(SAMPLE_DATA_PATH, selected_file)
                src_name = f"sample_data/{selected_file}"

                if selected_file.lower().endswith(".csv"):
                    df_up = pd.read_csv(sample_path)
                    cols = list(df_up.columns)
                    if cols:
                        rev_guess = next((c for c in cols if c.lower() == "review"), cols[0])
                        sum_guess = next((c for c in cols if c.lower() == "summary"), "(none)")
                        col1, col2 = st.columns(2)
                        review_col = col1.selectbox("Review column", cols, index=cols.index(rev_guess), key="sample_csv_review_col")
                        summary_opts = ["(none)"] + cols
                        summary_col = col2.selectbox(
                            "Summary column",
                            summary_opts,
                            index=summary_opts.index(sum_guess) if sum_guess in summary_opts else 0,
                            key="sample_csv_summary_col",
                        )
                        reviews_raw = df_up[review_col].fillna("").astype(str).tolist()
                        if summary_col != "(none)":
                            summaries_raw = df_up[summary_col].fillna("").astype(str).tolist()
                else:
                    xls = pd.ExcelFile(sample_path)
                    sheet = st.selectbox("Sheet", xls.sheet_names, key="sample_xlsx_sheet")
                    df_up = pd.read_excel(xls, sheet_name=sheet)
                    cols = list(df_up.columns)
                    if cols:
                        rev_guess = next((c for c in cols if c.lower() == "review"), cols[0])
                        sum_guess = next((c for c in cols if c.lower() == "summary"), "(none)")
                        col1, col2 = st.columns(2)
                        review_col = col1.selectbox("Review column", cols, index=cols.index(rev_guess), key="sample_xlsx_review_col")
                        summary_opts = ["(none)"] + cols
                        summary_col = col2.selectbox(
                            "Summary column",
                            summary_opts,
                            index=summary_opts.index(sum_guess) if sum_guess in summary_opts else 0,
                            key="sample_xlsx_summary_col",
                        )
                        reviews_raw = df_up[review_col].fillna("").astype(str).tolist()
                        if summary_col != "(none)":
                            summaries_raw = df_up[summary_col].fillna("").astype(str).tolist()

    else:
        pasted = st.text_area(
            "One review per line",
            placeholder="Great value for money.\nTerrible quality and slow support.\nIt is okay, nothing special.",
            height=170,
            label_visibility="collapsed",
        )
        src_name = "Pasted text"
        if pasted.strip():
            reviews_raw = [r.strip() for r in pasted.splitlines() if r.strip()]

    reviews_raw = [r for r in reviews_raw if str(r).strip()]
    if len(summaries_raw) != len(reviews_raw):
        summaries_raw = [""] * len(reviews_raw)

    if reviews_raw:
        st.success(f"Loaded {len(reviews_raw):,} reviews from {src_name}.")

    col_run, col_clear = st.columns([1, 1])
    run_batch = col_run.button("Run Batch Analysis", use_container_width=True, disabled=(len(reviews_raw) == 0))
    clear_last = col_clear.button("Clear Last Result", use_container_width=True)

    if clear_last:
        st.session_state["df_full"] = None
        st.info("Previous analysis cleared.")

    st.markdown("</div>", unsafe_allow_html=True)

    if run_batch and reviews_raw:
        with st.spinner(f"Analysing {len(reviews_raw):,} reviews..."):
            results = predict_batch(reviews_raw, summaries_raw or None)

        rows = []
        for i, (rev, summ, res) in enumerate(zip(reviews_raw, summaries_raw, results), 1):
            pb = res["probabilities"]
            rows.append({
                "#": i,
                "Review": rev,
                "Summary": summ,
                "Sentiment": res["sentiment"],
                "Confidence_%": res["confidence"],
                "Prob_Positive_%": pb.get("positive", 0),
                "Prob_Negative_%": pb.get("negative", 0),
                "Prob_Neutral_%": pb.get("neutral", 0),
            })
        st.session_state["df_full"] = pd.DataFrame(rows)

    df_full = st.session_state.get("df_full")

    if df_full is not None and not df_full.empty:
        filt = (df_full["Confidence_%"] >= min_conf) & (df_full["Sentiment"].isin(sent_filter))
        df_res = df_full.loc[filt].copy()
        total_all = len(df_full)

        if df_res.empty:
            st.warning("No rows match current sidebar filters. Reduce the confidence threshold or include more sentiments.")
        else:
            counts = {l: int((df_res["Sentiment"] == l).sum()) for l in LABELS}
            total = len(df_res)

            st.markdown("### 2) Results Dashboard")
            k1, k2, k3, k4, k5, k6 = st.columns(6)
            k1.metric("Visible", f"{total:,}", f"of {total_all:,}")
            k2.metric("Positive", f"{counts['positive']:,}", f"{counts['positive']/total*100:.1f}%")
            k3.metric("Negative", f"{counts['negative']:,}", f"{counts['negative']/total*100:.1f}%")
            k4.metric("Neutral", f"{counts['neutral']:,}", f"{counts['neutral']/total*100:.1f}%")
            k5.metric("Avg Confidence", f"{df_res['Confidence_%'].mean():.1f}%")
            k6.metric("High Conf (>90%)", f"{(df_res['Confidence_%'] > 90).sum():,}")

            st.markdown("#### Distribution")
            ca, cb = st.columns(2)
            ca.image(chart_pie(counts), use_container_width=True)
            cb.image(chart_bar_counts(counts), use_container_width=True)

            st.markdown("#### Confidence Analysis")
            cc, cd = st.columns(2)
            cc.image(chart_confidence_hist(df_res), use_container_width=True)
            cd.image(chart_avg_confidence(df_res), use_container_width=True)

            st.markdown("#### Stacked Probability (first 40 rows)")
            st.image(chart_stacked_probs(df_res), use_container_width=True)

            st.markdown("#### Preview")
            st.dataframe(
                df_res[["#", "Review", "Sentiment", "Confidence_%", "Prob_Positive_%", "Prob_Negative_%", "Prob_Neutral_%"]].head(preview_rows),
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("#### Export")
            d1, d2 = st.columns(2)
            d1.download_button(
                "Download Filtered CSV",
                data=df_res.to_csv(index=False).encode(),
                file_name="sentiment_results_filtered.csv",
                mime="text/csv",
                use_container_width=True,
            )
            d2.download_button(
                "Download Filtered Excel",
                data=build_excel(df_res, counts),
                file_name="sentiment_results_filtered.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

            with st.expander("Model Visual Diagnostics"):
                va, vb = st.columns(2)
                va.image(chart_model_metrics(), use_container_width=True, caption="Precision / Recall / F1")
                cv_buf = chart_cv_scores()
                if cv_buf:
                    vb.image(cv_buf, use_container_width=True, caption="Cross-validation accuracy")


with t_single:
    st.markdown('<div class="card"><div class="card-title">Single Review Prediction</div>', unsafe_allow_html=True)
    review_txt = st.text_area(
        "Review text",
        height=130,
        label_visibility="collapsed",
        placeholder="Absolutely love this product. Great quality and fast delivery.",
    )
    summ_txt = st.text_input("Summary (optional)", label_visibility="collapsed", placeholder="Great product")
    go = st.button("Analyse Sentiment", use_container_width=False)

    if go:
        if not review_txt.strip():
            st.warning("Please enter a review.")
        else:
            pred, probs = predict_one(review_txt, summ_txt)
            color = C[pred]
            st.markdown(
                f"<div class='result-wrap'><div class='result-badge' style='background:{color}22;color:{color};border:2px solid {color};'>"
                f"{EMO[pred]} {pred.upper()}</div></div>",
                unsafe_allow_html=True,
            )
            for lbl in ["positive", "neutral", "negative"]:
                p = probs.get(lbl, 0)
                st.markdown(
                    f"<div class='prob-row'><div class='prob-label'>{lbl}</div>"
                    f"<div class='prob-bar-wrap'><div class='prob-bar' style='width:{p}%;background:{C[lbl]};'></div></div>"
                    f"<div class='prob-pct'>{p:.1f}%</div></div>",
                    unsafe_allow_html=True,
                )
    st.markdown("</div>", unsafe_allow_html=True)


with t_about:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    <div class="card-title">About Insights Inside</div>
    <p style="color:#aaa;line-height:1.8;">
      Insights Inside is a sentiment analytics workspace designed for operations, product,
      and CX teams to evaluate customer feedback at scale with explainable class probabilities.
    </p>
    """, unsafe_allow_html=True)
    st.markdown("""
    - Supports CSV and Excel uploads
    - Batch scoring with confidence and per-class probabilities
    - Filtered analysis and exports for reporting workflows
    - Sidebar-first model diagnostics for transparent decisions
    """)
    st.markdown("</div>", unsafe_allow_html=True)