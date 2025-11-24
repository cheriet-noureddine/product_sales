class Product:
    """Represents a product with monthly sales."""
    def __init__(self, name, sales):
        self.name = name
        self.sales = sales

    def total_value(self):
        return sum(self.sales)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------ Data Cleaning ------------------
def clean_lines(lines):
    return [line.strip() for line in lines if line.strip()]

def convert_to_int_list(arr):
    cleaned = []
    for value in arr:
        try:
            cleaned.append(int(value))
        except ValueError:
            continue
    return cleaned

# ------------------ ETL Pipeline ------------------
def extract_data():
    return pd.date_range("2025-01-01", periods=12, freq="MS")

def transform_data(months):
    df = pd.DataFrame({
        "Month": months.strftime("%B"),
        "A": np.random.randint(50, 101, 12),
        "B": np.random.randint(30, 81, 12),
        "C": np.random.randint(20, 61, 12),
        "D": np.random.randint(10, 51, 12),
    })
    df.to_csv("data/initial.csv", index=False)

    df["Total"] = df[["A", "B", "C", "D"]].sum(axis=1)
    df["Average"] = df[["A", "B", "C", "D"]].mean(axis=1)
    df["Growth"] = df["Total"].pct_change().fillna(0) * 100
    df["Quarter"] = ((months.month - 1) // 3 + 1).map(lambda x: f"Q{x}")
    df["Top_Product"] = df[["A", "B", "C", "D"]].idxmax(axis=1)
    df["Low_Product"] = df[["A", "B", "C", "D"]].idxmin(axis=1)

    df.to_csv("data/final.csv", index=False)
    return df

# ------------------ Analysis ------------------
def quarterly_summary(df):
    avg = df.groupby("Quarter")[["A", "B", "C", "D"]].mean()
    total = df.groupby("Quarter")["Total"].sum()
    summary = pd.concat([avg, total], axis=1)
    summary.to_csv("data/output.csv")
    return summary

# ------------------ Visualizations ------------------
def make_visuals(df):
    plt.figure(figsize=(10,6))
    for col in ["A", "B", "C", "D"]:
        plt.plot(df["Month"], df[col], marker="o", linewidth=2)
    plt.title("Monthly Product Trends")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("viz_line.png")
    plt.close()

    df.set_index("Month")[["A", "B", "C", "D"]].plot(kind="bar", stacked=True, figsize=(10,6))
    plt.title("Sales Composition by Month")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("viz_stacked.png")
    plt.close()

    plt.figure(figsize=(10,6))
    sns.heatmap(df.set_index("Month")[["A", "B", "C", "D"]], annot=True, cmap="viridis", fmt="d")
    plt.title("Sales Heatmap")
    plt.tight_layout()
    plt.savefig("viz_heatmap.png")
    plt.close()

    plt.figure(figsize=(8,6))
    sns.boxplot(data=df[["A", "B", "C", "D"]])
    plt.title("Sales Distribution")
    plt.tight_layout()
    plt.savefig("viz_boxplot.png")
    plt.close()

# ------------------ Insights ------------------
def insights(df, summary):
    result = {
        "Best Month": df.loc[df["Total"].idxmax(), "Month"],
        "Top Product (Annual)": df[["A", "B", "C", "D"]].sum().idxmax(),
        "Best Quarter": summary["Total"].idxmax(),
    }
    pd.DataFrame(result.items(), columns=["Metric", "Value"]).to_csv("data/insights.csv", index=False)
