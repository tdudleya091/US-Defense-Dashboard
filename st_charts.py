"""
st_charts.py
------------
Streamlit-native chart helpers for main.py -- renders directly via
st.line_chart / st.bar_chart / st.altair_chart, all fed by plain pandas
DataFrames/Series. These auto-scale to the container width by default
(width='stretch' is the default for st.line_chart/bar_chart/scatter_chart
in this Streamlit version), unlike a matplotlib figure embedded via
st.pyplot(), which renders as a fixed-size static image. Altair (also
pandas-driven) fills the gaps Streamlit's built-in chart types can't cover
natively: pie/donut charts (no st.pie_chart) and layered scatter+fit-line
regression charts.

pipeline.py's static PNG output does NOT use this module -- see charts.py
(seaborn) for that, since scaling isn't a concern for a saved image file.
"""

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st


def _safe_field_name(s):
    """
    Strip characters that break Vega-Lite's 'field:Type' shorthand parser
    when a raw label is used as a DataFrame column name / Series index for
    st.line_chart/st.bar_chart. Most FRED series titles are "Category:
    Subcategory" (e.g. "New Orders: Ships and Boats"), and Streamlit's
    native chart elements build Vega-Lite shorthand internally from column
    names -- a colon in that name gets misread as the start of a type code
    (":Q"/":N"/etc.), raising a ValueError. Only used for chart rendering;
    the original label (with colon) is still shown in "Underlying data"
    tables and everywhere else, since st.dataframe doesn't use shorthand
    encoding and colons there are harmless.
    """
    return s.replace(":", " -") if isinstance(s, str) else s


def time_series_chart(df):
    """Multi-series line chart from a wide DataFrame (columns = series, index = date)."""
    if df is None or df.empty:
        st.warning("No data available for the selected series.")
        return
    st.line_chart(df.rename(columns=_safe_field_name))


def format_dollar(v):
    """Format a raw USD amount as $X.XXB / $X.XXM / $X.XXK."""
    abs_v = abs(v)
    if abs_v >= 1e9:
        return f"${v / 1e9:.2f}B"
    if abs_v >= 1e6:
        return f"${v / 1e6:.2f}M"
    if abs_v >= 1e3:
        return f"${v / 1e3:.2f}K"
    return f"${v:.2f}"


def bar_chart(series, y_axis_label="Value"):
    """Bar chart from a pd.Series (index = category labels, values = bar heights)."""
    if series is None or series.empty:
        st.warning("No data available.")
        return
    series = series.rename(index=_safe_field_name)
    st.bar_chart(series, y_label=y_axis_label, horizontal=True)


def pie_chart(labels, values, title=None, display_fn=None):
    """
    Pie chart via Altair's arc mark (Streamlit has no native st.pie_chart).

    Args:
        display_fn (callable or None): formats a raw value for the tooltip
            (e.g. dollar formatting for a market-cap-value pie). Defaults to
            a plain percent-of-total display, matching the arc proportions.
    """
    pairs = [(l, v) for l, v in zip(labels, values) if v is not None]
    if not pairs:
        st.warning("No data available.")
        return
    total = sum(v for _l, v in pairs)
    if display_fn is None:
        display_fn = lambda v: f"{v / total * 100:.1f}%" if total else "0%"

    df = pd.DataFrame(pairs, columns=["Label", "Value"])
    df["Display"] = df["Value"].apply(display_fn)
    chart = (
        alt.Chart(df)
        .mark_arc(outerRadius=140)
        .encode(
            theta=alt.Theta("Value:Q", stack=True),
            color=alt.Color("Label:N", legend=alt.Legend(title=None)),
            tooltip=["Label", "Display"],
        )
    )
    if title:
        chart = chart.properties(title=title)
    st.altair_chart(chart, width="stretch")


def regression_scatter_chart(result):
    """
    Layered Altair scatter (observed points) + fit line for a simple linear
    regression result (regression.linear_regression's return dict).
    """
    obs_df = pd.DataFrame({"x": result["x_clean"], "y": result["y_clean"]})
    order = np.argsort(result["x_clean"])
    fit_df = pd.DataFrame({
        "x": np.array(result["x_clean"])[order],
        "fit": np.array(result["y_pred"])[order],
    })

    points = alt.Chart(obs_df).mark_point(opacity=0.6).encode(
        x=alt.X("x:Q", title=result["x_name"]),
        y=alt.Y("y:Q", title=result["y_name"]),
    )
    line = alt.Chart(fit_df).mark_line(color="red").encode(x="x:Q", y="fit:Q")

    subtitle = f"y = {result['slope']:.4g}x + {result['intercept']:.4g}   R² = {result['r_squared']:.3f}   n = {result['n']}"
    chart = (points + line).properties(
        title=alt.TitleParams(text=f"{result['y_name']} vs {result['x_name']}", subtitle=subtitle)
    )
    st.altair_chart(chart, width="stretch")


def residual_chart(predicted, residuals, title="Residuals"):
    """Predicted-vs-residuals scatter with a zero reference line."""
    df = pd.DataFrame({"Predicted": np.asarray(predicted), "Residual": np.asarray(residuals)})
    points = alt.Chart(df).mark_point(opacity=0.6, color="green").encode(x="Predicted:Q", y="Residual:Q")
    zero_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="red", strokeDash=[4, 4]).encode(y="y:Q")
    chart = (points + zero_line).properties(title=title)
    st.altair_chart(chart, width="stretch")


def actual_vs_predicted_chart(y_actual, y_pred, title="Actual vs Predicted"):
    """Actual-vs-predicted scatter with a y=x reference line."""
    y_actual = np.asarray(y_actual)
    y_pred = np.asarray(y_pred)
    df = pd.DataFrame({"Actual": y_actual, "Predicted": y_pred})
    lo, hi = float(min(y_actual.min(), y_pred.min())), float(max(y_actual.max(), y_pred.max()))
    ref_df = pd.DataFrame({"Actual": [lo, hi], "Predicted": [lo, hi]})

    points = alt.Chart(df).mark_point(opacity=0.6).encode(x="Actual:Q", y="Predicted:Q")
    ref_line = alt.Chart(ref_df).mark_line(color="red", strokeDash=[4, 4]).encode(x="Actual:Q", y="Predicted:Q")
    chart = (points + ref_line).properties(title=title)
    st.altair_chart(chart, width="stretch")
