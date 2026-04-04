# Insights Inside

Professional Streamlit dashboard for product review sentiment analysis.

Insights Inside classifies customer reviews into:
- Positive
- Negative
- Neutral

It supports single-review prediction, batch analysis, model diagnostics, confidence-based filtering, and downloadable CSV/Excel reports.

## Features

- Clean, professional UI/UX with sidebar-driven controls
- Batch workspace for:
  - CSV upload
  - Excel upload
  - Sample-data-folder loading
  - Pasted text lines
- Automatic review/summary column mapping
- Rich analytics:
  - sentiment distribution
  - confidence analysis
  - per-class probability charts
- Filter results by confidence threshold and sentiment labels
- Export filtered results to CSV and Excel
- Single-review quick prediction mode
- Model metrics and diagnostics visible in the sidebar and dashboard

## Tech Stack

- Python 3.10+
- Streamlit
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- joblib
- openpyxl
- xlsxwriter

## Project Structure

```text
insights_inside/
├─ app.py
├─ sentiment_utils.py
├─ requirements.txt
├─ SAMPLE_DATA_GUIDE.md
├─ sample_data/
├─ model/
│  ├─ tfidf_vectorizer.pkl
│  └─ sentiment_model.pkl
└─ evaluation/
   ├─ metrics_summary.json
   └─ classification_report.txt
```

## Setup

1. Create and activate a virtual environment.

Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
pip install -r requirements.txt
```

3. Start the app.

```powershell
streamlit run app.py
```

4. Open the local URL shown in terminal (usually http://localhost:8501).

## How To Use

### 1) Batch Analysis (recommended)

1. Open Batch Workspace.
2. Choose Input source:
   - Upload CSV
   - Upload Excel
   - Sample data folder
   - Paste text lines
3. Select mapping for:
   - Review column (required)
   - Summary column (optional)
4. Click Run Batch Analysis.
5. Use sidebar filters (confidence and sentiment).
6. Download filtered results as CSV/Excel.

### 2) Single Review

1. Open Single Review tab.
2. Enter review text (and optional summary).
3. Click Analyse Sentiment.

## Sample Data Workflow

Use this when you want quick testing without re-uploading files:

1. Place test files inside sample_data/.
2. In app -> Batch Workspace -> Input source, choose Sample data folder.
3. Pick a file from the dropdown and run analysis.

See detailed steps in SAMPLE_DATA_GUIDE.md.

## Expected Input Format

Recommended columns:
- Review (required)
- Summary (optional)

If your file uses different names, map them in the UI.

Example CSV:

```csv
Review,Summary
Absolutely love this phone.,Great value
Battery drains too fast.,Not happy
It works as expected.,Okay product
```

## Troubleshooting

### 1) ModuleNotFoundError: xlsxwriter or openpyxl

Install missing packages:

```powershell
pip install xlsxwriter openpyxl
```

### 2) LogisticRegression missing attribute: multi_class

This app includes a compatibility guard while loading legacy pickled models.

If you still face model issues, align scikit-learn version between training and runtime.

### 3) Version mismatch warnings for scikit-learn

If your model was trained in a different scikit-learn version, prediction can still run but may be unstable.

Recommended fix:
- Use the same scikit-learn version used during training
- Re-export model artifacts after version alignment

## Notes

- Keep model artifacts in model/.
- Keep evaluation outputs in evaluation/.
- For team demos, add one or more files to sample_data/ and use Sample data folder mode.

## License

Use the appropriate license for your project/repository.
