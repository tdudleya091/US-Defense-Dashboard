"""
charts.py
---------
Matplotlib chart functions for the US defense dashboard, per user spec
(matplotlib, not plotly). Used both by pipeline.py (saves static PNGs to
output/) and by main.py (returns the Figure for st.pyplot() instead of
saving, via the return_fig=True path on each function).

Every function accepts pre-computed Series/dicts and either saves a PNG or
returns the Figure, closing the figure in the save path to avoid memory
leaks when generating many charts in one pipeline run.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _safe_filename(name):
    return "".join(c if c.isalnum() or c == "-" else "_" for c in name)


def _finish(fig, path, return_fig):
    """Either return the live Figure (Streamlit) or save+close it (pipeline.py)."""
    if return_fig:
        return fig
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[CHART] Saved: {path}")
    return None


def plot_timeseries_comparison(series_dict, title, y_label, filename_stem, normalize=False, return_fig=False):
    """
    Overlay any number of named Series on one time-series chart -- the
    generic "comparison chart in time series with menus for each" required
    by the checklist. Each key in series_dict becomes one line.

    Args:
        series_dict (dict): {label: pd.Series or None}
        title (str), y_label (str)
        filename_stem (str): output filename (without extension) when saved
        normalize (bool): rebase every series to 100 at its first valid value
        return_fig (bool): True to return the Figure (Streamlit use) instead
                            of saving a PNG (pipeline.py use)
    """
    ensure_output_dir()
    plottable = {k: s for k, s in series_dict.items() if s is not None and not s.empty}
    if not plottable:
        print(f"[SKIP] {title}: no plottable series")
        return None

    fig, ax = plt.subplots(figsize=(12, 6))
    for label, s in plottable.items():
        y = s.values
        if normalize:
            valid = s.dropna()
            if valid.empty or valid.iloc[0] == 0:
                continue
            y = (s / valid.iloc[0]) * 100
        ax.plot(s.index, y, label=label, linewidth=1.5)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Indexed to 100 at start" if normalize else y_label)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, f"{_safe_filename(filename_stem)}.png")
    return _finish(fig, out_path, return_fig)


def plot_macro_series(label, series, y_label=None, title=None, return_fig=False):
    """Standalone time-series chart for one FRED macro series."""
    ensure_output_dir()
    if series is None or series.empty:
        print(f"[SKIP] macro series '{label}': no data available")
        return None

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(series.index, series.values, color="teal", linewidth=1.5)
    ax.set_title(title or label)
    ax.set_xlabel("Date")
    ax.set_ylabel(y_label or "Value")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, f"macro_{_safe_filename(label)}.png")
    return _finish(fig, out_path, return_fig)


def plot_regression(label, result, return_fig=False):
    """Combined scatter+fit / residual chart for one simple linear regression."""
    ensure_output_dir()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.scatter(result["x_clean"], result["y_clean"], alpha=0.6, color="blue", s=30)
    order = np.argsort(result["x_clean"])
    ax1.plot(np.array(result["x_clean"])[order], np.array(result["y_pred"])[order], "r-", linewidth=2)
    ax1.set_xlabel(result["x_name"])
    ax1.set_ylabel(result["y_name"])
    ax1.set_title("Linear Regression")
    ax1.grid(True, alpha=0.3)

    eq = f"y = {result['slope']:.4g}x + {result['intercept']:.4g}"
    r2 = f"R² = {result['r_squared']:.3f}   n = {result['n']}"
    ax1.text(0.05, 0.95, f"{eq}\n{r2}", transform=ax1.transAxes,
              bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8), verticalalignment="top")

    residuals = result["y_clean"] - result["y_pred"]
    ax2.scatter(result["y_pred"], residuals, alpha=0.6, color="green")
    ax2.axhline(y=0, color="red", linestyle="--")
    ax2.set_xlabel("Predicted Values")
    ax2.set_ylabel("Residuals")
    ax2.set_title("Residual Plot")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(label)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, f"regression_{_safe_filename(label)}.png")
    return _finish(fig, out_path, return_fig)


def plot_multiple_regression(label, result, return_fig=False):
    """Actual-vs-predicted scatter for a multiple regression fit."""
    ensure_output_dir()
    fig, ax = plt.subplots(figsize=(7, 6))

    y_actual = result["y_actual"].values
    y_pred = result["y_pred"].values
    ax.scatter(y_actual, y_pred, alpha=0.6, color="blue", s=30)

    lo, hi = min(y_actual.min(), y_pred.min()), max(y_actual.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=2, label="y = x")

    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    r2 = f"R² = {result['r_squared']:.3f}   Adj. R² = {result['adj_r_squared']:.3f}   n = {result['n']}"
    ax.set_title(f"{label}\n{r2}")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, f"multiregression_{_safe_filename(label)}.png")
    return _finish(fig, out_path, return_fig)


def plot_market_share_bar(labels, values, title, filename_stem, return_fig=False):
    """Horizontal bar chart of market-cap shares (percent) -- used for the
    'share of sub industry / share of industry in market cap' statistics."""
    ensure_output_dir()
    pairs = [(l, v) for l, v in zip(labels, values) if v is not None]
    if not pairs:
        print(f"[SKIP] {title}: no market-cap share data")
        return None
    pairs.sort(key=lambda p: p[1])
    bar_labels, bar_values = zip(*pairs)

    fig, ax = plt.subplots(figsize=(9, max(3, 0.4 * len(bar_labels))))
    ax.barh(bar_labels, bar_values, color="steelblue")
    ax.set_xlabel("Share of Market Cap (%)")
    ax.set_title(title)
    for i, v in enumerate(bar_values):
        ax.text(v, i, f" {v:.1f}%", va="center", fontsize=8)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, f"{_safe_filename(filename_stem)}.png")
    return _finish(fig, out_path, return_fig)
