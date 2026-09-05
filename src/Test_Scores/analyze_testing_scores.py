from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Public portfolio version
#
# This script expects locally supplied aggregate ACT and SAT trend tables. The
# source files themselves are not redistributed in this repository because
# redistribution rights and source terms should be checked independently.
#
# Expected ACT columns include: Year, Math, English, Reading, Science
# Expected SAT columns include: Year, Average Total Score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "data" / "Test_Scores" / "local_inputs"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "Test_Scores" / "visualizations"

ACT_CSV_PATH = INPUT_DIR / "act_averages_test_takers.csv"
SAT_CSV_PATH = INPUT_DIR / "sat_totals_participation.csv"


def load_data():
    """Load locally supplied ACT and SAT aggregate tables."""
    try:
        act_df = pd.read_csv(ACT_CSV_PATH)
        sat_df = pd.read_csv(SAT_CSV_PATH)
        return act_df, sat_df
    except FileNotFoundError as exc:
        print(
            "Required local source files were not found. "
            "See data/README.md for the expected schemas and source-data boundary."
        )
        print(exc)
        return None, None


def plot_act_math_trend(df):
    df = df.sort_values("Year")
    plt.figure(figsize=(12, 7))
    plt.plot(df["Year"], df["Math"], marker="o", linestyle="-", linewidth=2.5, label="Math Score")
    plt.axvline(x=2014, linestyle="--", linewidth=2, label="2014 reference point")
    plt.title("National Average ACT Math Scores")
    plt.xlabel("Year")
    plt.ylabel("Average Math Score")
    plt.grid(True, linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "act_math_scores_trend.png")
    plt.close()


def plot_all_act_subjects(df):
    df = df.sort_values("Year")
    plt.figure(figsize=(12, 7))
    for subject in ("Math", "English", "Reading", "Science"):
        if subject in df.columns:
            plt.plot(df["Year"], df[subject], marker="o", linestyle="-", label=subject)
    plt.title("National Average ACT Scores by Subject")
    plt.xlabel("Year")
    plt.ylabel("Average Score")
    plt.grid(True, linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "act_all_subjects_trend.png")
    plt.close()


def plot_math_scores_by_decade(df):
    working = df.copy()
    working["Decade"] = (working["Year"] // 10) * 10
    decade_avg = working.groupby("Decade", as_index=False)["Math"].mean()
    decade_avg["Decade"] = decade_avg["Decade"].astype(str) + "s"

    plt.figure(figsize=(10, 6))
    plt.bar(decade_avg["Decade"], decade_avg["Math"])
    plt.title("Average ACT Math Score by Decade")
    plt.xlabel("Decade")
    plt.ylabel("Average Math Score")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "act_math_scores_by_decade.png")
    plt.close()


def plot_sat_total_trend(df):
    df = df.sort_values("Year")
    plt.figure(figsize=(12, 7))
    plt.plot(df["Year"], df["Average Total Score"], marker="o", linestyle="-", linewidth=2.5)
    plt.title("National Average SAT Total Scores")
    plt.xlabel("Year")
    plt.ylabel("Average Total Score")
    plt.grid(True, linewidth=0.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sat_total_scores_trend.png")
    plt.close()


def exploratory_math_projection(df):
    """
    Produce a simple linear extrapolation from a selected recent window.

    This is an exploratory visualization only. It is not presented as a causal
    forecast and should not be used to infer the effect of any policy or
    technology without stronger identification and controls.
    """
    trend_data = df[df["Year"] >= 2014].dropna(subset=["Year", "Math"]).copy()
    if len(trend_data) < 2:
        return

    X = trend_data["Year"].to_numpy().reshape(-1, 1)
    y = trend_data["Math"].to_numpy()
    model = LinearRegression().fit(X, y)

    future_years = np.arange(int(trend_data["Year"].max()) + 1, int(trend_data["Year"].max()) + 6).reshape(-1, 1)
    predicted_scores = model.predict(future_years)

    plt.figure(figsize=(12, 7))
    plt.plot(df["Year"], df["Math"], marker="o", linestyle="-", label="Historical average")
    plt.plot(future_years, predicted_scores, linestyle="--", label="Exploratory linear extension")
    plt.title("Exploratory ACT Math Trend Extension")
    plt.xlabel("Year")
    plt.ylabel("Average Math Score")
    plt.grid(True, linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "act_math_exploratory_projection.png")
    plt.close()


if __name__ == "__main__":
    act_data, sat_data = load_data()
    if act_data is not None and sat_data is not None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        plot_act_math_trend(act_data)
        plot_all_act_subjects(act_data)
        plot_math_scores_by_decade(act_data)
        plot_sat_total_trend(sat_data)
        exploratory_math_projection(act_data)
