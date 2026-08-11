"""
charts.py
---------
Static seaborn chart functions used ONLY by pipeline.py (CLI) to save PNGs
to output/. The interactive Streamlit app (main.py) does NOT use this
module -- it uses st_charts.py instead, which renders natively-scaling
Altair/pandas charts (st.line_chart/bar_chart/altair_chart) rather than
fixed-size matplotlib images. This module still needs a raster backend
(seaborn is matplotlib under the hood) because it's writing PNG files to
disk, which has no scaling concern in the first place.

Every function accepts pre-computed Series/dicts and saves a PNG, closing
the figure after saving to avoid memory leaks when generating many charts
in one pipeline run.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _safe_filename(name):
    return "".join(c if c.isalnum() or c == "-" else "_" for c in name)


def _save_and_close(fig, path):
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[CHART] Saved: {path}")


def plot_timeseries_comparison(series_dict, title, y_label, filename_stem, normalize=False):
    """
    Overlay any number of named Series on one seaborn line chart.

    Args:
        series_dict (dict): {label: pd.Series or None}
        normalize (bool): rebase every series to 100 at its first valid value
    """
    ensure_output_dir()
    plottable = {k: s for k, s in series_dict.items() if s is not None and not s.empty}
    if not plottable:
        print(f"[SKIP] {title}: no plottable series")
        return

    frames = []
    for label, s in plottable.items():
        # strip tz before stacking into one shared "Date" column -- mixing
        # tz-aware (stock/index data) and tz-naive (FRED data) dates in one
        # column makes it an "object" dtype matplotlib/seaborn can't convert
        if s.index.tz is not None:
            s = pd.Series(s.values, index=s.index.tz_localize(None))
        y = s
        if normalize:
            valid = s.dropna()
            if valid.empty or valid.iloc[0] == 0:
                continue
            y = (s / valid.iloc[0]) * 100
        frames.append(pd.DataFrame({"Date": s.index, "Value": y.values, "Series": label}))
    long_df = pd.concat(frames, ignore_index=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=long_df, x="Date", y="Value", hue="Series", ax=ax, linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Indexed to 100 at start" if normalize else y_label)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()

    _save_and_close(fig, os.path.join(OUTPUT_DIR, f"{_safe_filename(filename_stem)}.png"))


def plot_macro_series(label, series, y_label=None, title=None):
    """Standalone time-series chart for one FRED macro series."""
    ensure_output_dir()
    if series is None or series.empty:
        print(f"[SKIP] macro series '{label}': no data available")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(x=series.index, y=series.values, ax=ax, color="teal", linewidth=1.5)
    ax.set_title(title or label)
    ax.set_xlabel("Date")
    ax.set_ylabel(y_label or "Value")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()

    _save_and_close(fig, os.path.join(OUTPUT_DIR, f"macro_{_safe_filename(label)}.png"))


def plot_regression(label, result):
    """Combined scatter+fit (seaborn.regplot) / residual chart for one simple linear regression."""
    ensure_output_dir()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    sns.regplot(x=result["x_clean"], y=result["y_clean"], ax=ax1,
                scatter_kws={"alpha": 0.6, "s": 30}, line_kws={"color": "red"})
    ax1.set_xlabel(result["x_name"])
    ax1.set_ylabel(result["y_name"])
    ax1.set_title("Linear Regression")

    eq = f"y = {result['slope']:.4g}x + {result['intercept']:.4g}"
    r2 = f"R² = {result['r_squared']:.3f}   n = {result['n']}"
    ax1.text(0.05, 0.95, f"{eq}\n{r2}", transform=ax1.transAxes,
              bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8), verticalalignment="top")

    residuals = result["y_clean"] - result["y_pred"]
    sns.scatterplot(x=result["y_pred"], y=residuals, ax=ax2, alpha=0.6, color="green")
    ax2.axhline(y=0, color="red", linestyle="--")
    ax2.set_xlabel("Predicted Values")
    ax2.set_ylabel("Residuals")
    ax2.set_title("Residual Plot")

    fig.suptitle(label)
    plt.tight_layout()

    _save_and_close(fig, os.path.join(OUTPUT_DIR, f"regression_{_safe_filename(label)}.png"))


def plot_multiple_regression(label, result):
    """Actual-vs-predicted scatter AND residuals-vs-predicted scatter for a multiple regression fit."""
    ensure_output_dir()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    y_actual = result["y_actual"].values
    y_pred = np.asarray(result["y_pred"])
    residuals = y_actual - y_pred

    sns.scatterplot(x=y_actual, y=y_pred, ax=ax1, alpha=0.6, s=30)
    lo, hi = min(y_actual.min(), y_pred.min()), max(y_actual.max(), y_pred.max())
    ax1.plot([lo, hi], [lo, hi], "r--", linewidth=2, label="y = x")
    ax1.set_xlabel("Actual")
    ax1.set_ylabel("Predicted")
    ax1.set_title("Actual vs Predicted")
    ax1.legend(loc="upper left", fontsize=8)

    sns.scatterplot(x=y_pred, y=residuals, ax=ax2, alpha=0.6, color="green")
    ax2.axhline(y=0, color="red", linestyle="--")
    ax2.set_xlabel("Predicted Values")
    ax2.set_ylabel("Residuals")
    ax2.set_title("Residual Plot")

    r2 = f"R² = {result['r_squared']:.3f}   Adj. R² = {result['adj_r_squared']:.3f}   n = {result['n']}"
    fig.suptitle(f"{label}\n{r2}")
    plt.tight_layout()

    _save_and_close(fig, os.path.join(OUTPUT_DIR, f"multiregression_{_safe_filename(label)}.png"))


def plot_market_share_bar(labels, values, title, filename_stem):
    """Horizontal seaborn bar chart of market-cap shares (percent)."""
    ensure_output_dir()
    pairs = [(l, v) for l, v in zip(labels, values) if v is not None]
    if not pairs:
        print(f"[SKIP] {title}: no market-cap share data")
        return
    pairs.sort(key=lambda p: p[1])
    bar_labels, bar_values = zip(*pairs)

    fig, ax = plt.subplots(figsize=(9, max(3, 0.4 * len(bar_labels))))
    sns.barplot(x=list(bar_values), y=list(bar_labels), ax=ax, color="steelblue")
    ax.set_xlabel("Share of Market Cap (%)")
    ax.set_title(title)
    for i, v in enumerate(bar_values):
        ax.text(v, i, f" {v:.1f}%", va="center", fontsize=8)
    plt.tight_layout()

    _save_and_close(fig, os.path.join(OUTPUT_DIR, f"{_safe_filename(filename_stem)}.png"))


def plot_market_share_pie(labels, values, title, filename_stem):
    """Pie chart of market-cap shares (percent), using a seaborn color palette."""
    ensure_output_dir()
    pairs = [(l, v) for l, v in zip(labels, values) if v is not None]
    if not pairs:
        print(f"[SKIP] {title}: no market-cap share data")
        return
    pie_labels, pie_values = zip(*pairs)

    fig, ax = plt.subplots(figsize=(7, 7))
    colors = sns.color_palette("pastel", n_colors=len(pie_labels))
    ax.pie(pie_values, labels=pie_labels, autopct="%1.1f%%", colors=colors, startangle=90)
    ax.set_title(title)
    plt.tight_layout()

    _save_and_close(fig, os.path.join(OUTPUT_DIR, f"{_safe_filename(filename_stem)}_pie.png"))
