"""Expense Analyzer.

Reads a CSV of expenses, auto-detects categories, computes statistics,
flags anomalous spending, and generates charts.

Usage:
    python analyzer.py --demo               # generate demo data and analyze it
    python analyzer.py --file my_data.csv   # analyze your own file

CSV format: date,amount,description[,category]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to file, no GUI window needed — works on servers too
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------- Settings ----------

CATEGORY_RULES = {
    "Groceries": ["walmart", "kroger", "trader joe", "whole foods", "safeway", "costco"],
    "Transport": ["uber", "lyft", "gas station", "metro", "subway fare", "parking"],
    "Dining Out": ["coffee", "starbucks", "cafe", "restaurant", "pizza", "mcdonald"],
    "Subscriptions": ["netflix", "spotify", "youtube", "subscription", "prime"],
    "Health": ["pharmacy", "clinic", "doctor", "cvs", "walgreens"],
    "Entertainment": ["cinema", "steam", "theater", "concert", "movie"],
    "Housing": ["rent", "utilities", "internet bill", "electricity"],
}
DEFAULT_CATEGORY = "Other"
ANOMALY_Z_THRESHOLD = 2.5  # how far above normal counts as an "anomaly"


# ---------- Data ----------

def categorize(description: str) -> str:
    """Guess a category from keywords in the description."""
    text = str(description).lower()
    for category, keywords in CATEGORY_RULES.items():
        if any(word in text for word in keywords):
            return category
    return DEFAULT_CATEGORY


def generate_demo(path: Path, days: int = 180, seed: int = 42) -> None:
    """Create a realistic demo CSV of expenses for the last `days` days."""
    rng = np.random.default_rng(seed)
    shops = {
        "Walmart": (30, 150),
        "Kroger": (20, 120),
        "Metro fare": (5, 15),
        "Uber": (10, 90),
        "Starbucks": (5, 15),
        "Local Cafe": (10, 60),
        "CVS Pharmacy": (10, 150),
        "Cinema": (12, 40),
        "Netflix subscription": (15.49, 15.49),
        "Steam": (10, 80),
    }
    names = list(shops)
    rows = []
    start = pd.Timestamp.today().normalize() - pd.Timedelta(days=days)

    for day in pd.date_range(start, periods=days):
        for _ in range(rng.integers(0, 4)):  # 0-3 purchases per day
            name = names[rng.integers(len(names))]
            low, high = shops[name]
            amount = round(float(rng.uniform(low, high)), 2)
            rows.append((day, amount, name))
        if day.day == 1:
            rows.append((day, 1400.0, "Rent payment"))
        if day.day == 15:
            rows.append((day, round(float(rng.uniform(80, 180)), 2), "Utilities and internet"))

    # a couple of deliberately "weird" expenses so the anomaly detector has something to find
    rows.append((start + pd.Timedelta(days=60), 600.0, "Uber"))
    rows.append((start + pd.Timedelta(days=120), 450.0, "Steam"))

    df = pd.DataFrame(rows, columns=["date", "amount", "description"])
    df.sort_values("date").to_csv(path, index=False)
    print(f"Demo data saved to: {path} ({len(df)} records)")


def load_expenses(path: Path) -> pd.DataFrame:
    """Load and clean the data."""
    df = pd.read_csv(path)

    required = {"date", "amount", "description"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["date", "amount"])  # drop malformed rows
    df = df[df["amount"] > 0].copy()

    if "category" not in df.columns:
        df["category"] = df["description"].apply(categorize)
    else:
        df["category"] = df["category"].fillna(df["description"].apply(categorize))

    df["month"] = df["date"].dt.to_period("M")
    return df.sort_values("date").reset_index(drop=True)


# ---------- Analysis ----------

def find_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Find expenses that are unusually high relative to their own category (z-score)."""
    stats = df.groupby("category")["amount"].agg(["mean", "std", "count"])
    df = df.join(stats, on="category")
    df["z"] = (df["amount"] - df["mean"]) / df["std"]
    mask = (df["count"] >= 5) & (df["z"] > ANOMALY_Z_THRESHOLD)
    return df.loc[mask, ["date", "category", "description", "amount", "z"]]


def build_report(df: pd.DataFrame) -> str:
    """Assemble the text report."""
    total = df["amount"].sum()
    days = (df["date"].max() - df["date"].min()).days + 1

    by_category = df.groupby("category")["amount"].sum().sort_values(ascending=False)
    by_month = df.groupby("month")["amount"].sum()
    mom_change = by_month.pct_change() * 100
    top5 = df.nlargest(5, "amount")
    anomalies = find_anomalies(df)

    lines = [
        "=" * 50,
        "EXPENSE REPORT",
        "=" * 50,
        f"Period: {df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d} ({days} days)",
        f"Total spent: ${total:,.2f}",
        f"Average per day: ${total / days:,.2f}",
        f"Number of transactions: {len(df)}",
        "",
        "--- By Category ---",
    ]
    for category, value in by_category.items():
        lines.append(f"{category:<20} ${value:>10,.2f}  ({value / total:5.1%})")

    lines += ["", "--- By Month ---"]
    for month, value in by_month.items():
        change = mom_change[month]
        sign = "" if np.isnan(change) else f"  ({change:+.1f}% vs previous month)"
        lines.append(f"{month}  ${value:>10,.2f}{sign}")

    lines += ["", "--- Top 5 Largest Expenses ---"]
    for _, row in top5.iterrows():
        lines.append(f"{row['date']:%Y-%m-%d}  ${row['amount']:>9,.2f}  {row['description']}")

    lines += ["", "--- Suspiciously Large Expenses ---"]
    if anomalies.empty:
        lines.append("None found")
    else:
        for _, row in anomalies.iterrows():
            lines.append(
                f"{row['date']:%Y-%m-%d}  ${row['amount']:>9,.2f}  "
                f"{row['description']} [{row['category']}]  (z={row['z']:.1f})"
            )
    return "\n".join(lines)


# ---------- Visualization ----------

def make_charts(df: pd.DataFrame, out_dir: Path) -> None:
    """Save charts as PNG files."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Spending share by category
    by_category = df.groupby("category")["amount"].sum().sort_values(ascending=False)
    plt.figure(figsize=(7, 7))
    plt.pie(by_category, labels=by_category.index, autopct="%1.0f%%", startangle=90)
    plt.title("Spending by Category")
    plt.savefig(out_dir / "categories.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Monthly trend
    by_month = df.groupby("month")["amount"].sum()
    plt.figure(figsize=(9, 5))
    plt.bar(by_month.index.astype(str), by_month.values, color="#4C72B0")
    plt.title("Spending by Month")
    plt.ylabel("$")
    plt.xticks(rotation=45)
    plt.savefig(out_dir / "months.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Daily spending + rolling average
    daily = df.groupby("date")["amount"].sum().asfreq("D", fill_value=0)
    plt.figure(figsize=(10, 5))
    plt.plot(daily.index, daily.values, alpha=0.4, label="daily")
    plt.plot(daily.index, daily.rolling(7).mean(), linewidth=2, label="7-day average")
    plt.title("Daily Spending")
    plt.ylabel("$")
    plt.legend()
    plt.savefig(out_dir / "daily.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------- CLI ----------

def main() -> None:
    parser = argparse.ArgumentParser(description="Personal expense analyzer")
    parser.add_argument("--file", type=Path, help="path to a CSV file of expenses")
    parser.add_argument("--demo", action="store_true", help="generate demo data")
    parser.add_argument("--out", type=Path, default=Path("output"), help="output folder for results")
    args = parser.parse_args()

    if args.demo:
        args.file = Path("demo_expenses.csv")
        generate_demo(args.file)
    if args.file is None:
        parser.error("provide --file path_to_csv or use --demo")

    df = load_expenses(args.file)
    report = build_report(df)
    print(report)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.txt").write_text(report, encoding="utf-8")
    make_charts(df, args.out)
    print(f"\nReport and charts saved to: {args.out}/")


if __name__ == "__main__":
    main()
