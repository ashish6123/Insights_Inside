# Sample Data Guide

Use this flow when you want to demo the app quickly without uploading files manually each time.

## 1) Add sample files

Put your test files inside:

sample_data/

Supported formats:
- .csv
- .xlsx

Recommended columns:
- Review (required)
- Summary (optional)

If column names are different, the app lets you map the review/summary columns in the UI.

## 2) Run using sample files

1. Start the app:
   streamlit run app.py
2. Open Batch Workspace.
3. In Input source, choose Sample data folder.
4. Select the file from the dropdown.
5. Choose review/summary columns.
6. Click Run Batch Analysis.

## 3) Share with others

To let another person test quickly:
1. Ask them to place their CSV/XLSX in sample_data/.
2. Tell them to use the Sample data folder option in Batch Workspace.
3. They can export filtered results from the Export section.

## Example CSV

Review,Summary
Absolutely love this phone.,Great value
Battery drains too fast.,Not happy
It works as expected.,Okay product
