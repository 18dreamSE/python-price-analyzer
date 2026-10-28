# 💸 Expense Analyzer

A command-line personal expense analyzer built with Python (pandas + matplotlib).
It reads a CSV of transactions, auto-categorizes them, computes statistics, flags anomalous spending, and generates charts.

## Features

- Loads and cleans data (drops malformed rows, negative amounts)
- Auto-categorization based on keywords in the transaction description
- Report: totals, daily average, breakdown by category and month, month-over-month change
- Top 5 largest transactions
- Anomaly detection (z-score within each category)
- Charts: spending share by category, monthly trend, daily spending with 7-day rolling average
- `--demo` mode — runs out of the box with no data of your own

## Setup & Usage

```bash
pip install -r requirements.txt

python analyzer.py --demo                # use generated demo data
python analyzer.py --file my_data.csv    # use your own data
python analyzer.py --file my_data.csv --out results
```

## CSV Format

```csv
date,amount,description,category
2026-10-01,1400.00,Rent payment,
2026-10-02,54.30,Walmart,
```

The `category` column is optional — if left blank, the category is detected automatically from the description.

## Output

Saved to the `output/` folder: `report.txt`, `categories.png`, `months.png`, `daily.png`.

## Ideas for Extension

- [ ] Category budgets with overspending alerts
- [ ] Spending forecast for next month (linear regression, scikit-learn)
- [ ] ML-based category classifier instead of keyword matching
- [ ] Bank statement import
- [ ] Web interface with Streamlit
- [ ] Unit tests with pytest

## Stack

Python 3.10+, pandas, numpy, matplotlib
