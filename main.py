"""
main.py
-------
Streamlit front-end for the US Defense Dashboard -- named main.py (rather
than app.py) so it matches whatever "Main file path" Streamlit Community
Cloud has configured once deployed (that setting isn't always reachable
from the app's own dashboard after the fact).

Loads a pre-built data snapshot (data/snapshot.parquet + data/snapshot_meta.json,
built by running `python build_snapshot.py` locally) by default rather than
fetching live from Yahoo Finance / FRED on every cold start -- Yahoo Finance
commonly rate-limits shared cloud datacenter IPs even though the same
fetches complete instantly run locally. A "Try live fetch" button in the
sidebar opts into a live pull.

All charts use matplotlib (st.pyplot), per user spec -- not plotly.

Usage:
    streamlit run main.py
"""

import json
import os
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

try:
    if "FRED_API_KEY" in st.secrets:
        os.environ.setdefault("FRED_API_KEY", st.secrets["FRED_API_KEY"])
except Exception:
    pass

import pipeline as MN
import regression as R
import frames as F
import charts as C
import tickers as T

st.set_page_config(page_title="US Defense Dashboard", layout="wide")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PARQUET_PATH = os.path.join(DATA_DIR, "snapshot.parquet")
META_PATH = os.path.join(DATA_DIR, "snapshot_meta.json")

SUB_INDUSTRIES = ["Aviation", "Shipbuilding"]


def load_snapshot():
    if not (os.path.exists(PARQUET_PATH) and os.path.exists(META_PATH)):
        return None, None
    try:
        wide_df = pd.read_parquet(PARQUET_PATH)
        with open(META_PATH) as f:
            meta = json.load(f)
    except Exception as e:
        st.sidebar.error(f"Failed to load snapshot: {e}")
        return None, None

    catalog = {}
    for label, info in meta["series"].items():
        if label not in wide_df.columns:
            continue
        series = wide_df[label].dropna()
        if series.empty:
            continue
        catalog[label] = {"series": series, "category": info["category"], "freq": info["freq"], "sub_industry": info.get("sub_industry")}

    return catalog, meta["built_at"]


@st.cache_resource(show_spinner="Fetching live stock, index, and macro data (can be slow/unreliable on cloud)...")
def fetch_live_catalog():
    company_data, index_data, sub_industry_avg, industry_avg, fred_data = MN.fetch_all_raw()
    return F.build_catalog(company_data, index_data, sub_industry_avg, industry_avg, fred_data)


st.title("US Defense Dashboard")
st.caption(
    "Aviation & Shipbuilding sub-industries: stock price, P/B, P/E, Market Cap, and market-cap share, "
    "compared against sub-industry/industry averages, DJIA/NASDAQ, and sub-industry-specific FRED macro factors."
)

if "catalog" not in st.session_state:
    snapshot_catalog, snapshot_built_at = load_snapshot()
    st.session_state["catalog"] = snapshot_catalog
    st.session_state["data_source"] = "snapshot" if snapshot_catalog else None
    st.session_state["built_at"] = snapshot_built_at

if st.sidebar.button("Try live fetch (Yahoo Finance / FRED)"):
    try:
        st.session_state["catalog"] = fetch_live_catalog()
        st.session_state["data_source"] = "live"
        st.session_state["built_at"] = datetime.now(timezone.utc).isoformat(timespec="minutes")
    except Exception as e:
        st.sidebar.error(f"Live fetch failed: {e}")

catalog = st.session_state.get("catalog")

if catalog:
    st.sidebar.caption(f"Data source: **{st.session_state['data_source']}** (built {st.session_state['built_at']} UTC)")

if not catalog:
    st.error(
        "No data available. No snapshot found at data/snapshot.parquet -- run "
        "`python build_snapshot.py` locally and commit data/snapshot.parquet + "
        "data/snapshot_meta.json, or click 'Try live fetch' in the sidebar."
    )
    st.stop()

all_categories = sorted({v["category"] for v in catalog.values()})
picked_categories = st.sidebar.multiselect("Filter by category", all_categories, default=all_categories)
all_labels = sorted(l for l, v in catalog.items() if v["category"] in picked_categories)

if not all_labels:
    st.warning("No series match the selected categories.")
    st.stop()

tab_ts, tab_share, tab_reg, tab_mreg = st.tabs(
    ["Time Series Comparison", "Market-Cap Share", "Linear Regression", "Multiple Regression"]
)

# --------------------------------------------------------------------------
# Tab 1: Time Series Comparison -- overlay any combination of series.
# --------------------------------------------------------------------------
with tab_ts:
    st.subheader("Compare any combination of series over time")
    col1, col2 = st.columns([3, 1])
    with col1:
        default_ts = [l for l in all_labels if l == "Industry Avg — Price (USD)"] or all_labels[:1]
        selected = st.multiselect("Series to plot", all_labels, default=default_ts, key="ts_select")
    with col2:
        freq_choice = st.selectbox("Resample", ["Native", "Monthly"], index=0)
        normalize = st.checkbox("Normalize to indexed 100", value=len(selected) > 1)

    freq_map = {"Native": None, "Monthly": "M"}

    if not selected:
        st.info("Pick one or more series above to plot.")
    else:
        df = F.wide_frame(catalog, selected, freq=freq_map[freq_choice], normalize=False)
        series_dict = {col: df[col].dropna() for col in df.columns}
        fig = C.plot_timeseries_comparison(
            series_dict, title="Series Comparison",
            y_label="Value (native units)", filename_stem="streamlit_ts",
            normalize=normalize, return_fig=True,
        )
        if fig is None:
            st.warning("No data available for the selected series.")
        else:
            st.pyplot(fig, clear_figure=True)
            with st.expander("Underlying data"):
                st.dataframe(df, use_container_width=True)

# --------------------------------------------------------------------------
# Tab 2: Market-Cap Share -- each company's share of its sub-industry and
# of the combined industry, plus each sub-industry's share of the industry.
# --------------------------------------------------------------------------
with tab_share:
    st.subheader("Market-cap share (latest available value)")
    sub_choice = st.radio("Sub-industry", SUB_INDUSTRIES, horizontal=True)

    share_labels, share_values = [], []
    for name in T.ALL_COMPANIES:
        if T.SUB_INDUSTRY_MAP.get(name) != sub_choice:
            continue
        label = f"{name} — Share of Sub-Industry Market Cap"
        entry = catalog.get(label)
        if entry is not None and not entry["series"].empty:
            share_labels.append(name)
            share_values.append(entry["series"].dropna().iloc[-1])

    fig = C.plot_market_share_bar(
        share_labels, share_values, title=f"{sub_choice}: Share of Sub-Industry Market Cap",
        filename_stem="streamlit_share", return_fig=True,
    )
    if fig:
        st.pyplot(fig, clear_figure=True)
    else:
        st.info("No market-cap share data available for this sub-industry.")

    st.markdown("#### Sub-industry share of combined defense-industry market cap")
    ind_labels, ind_values = [], []
    for sub in SUB_INDUSTRIES:
        label = f"{sub} — Share of Industry Market Cap"
        entry = catalog.get(label)
        if entry is not None and not entry["series"].empty:
            ind_labels.append(sub)
            ind_values.append(entry["series"].dropna().iloc[-1])
    fig2 = C.plot_market_share_bar(
        ind_labels, ind_values, title="Sub-Industry Share of Combined Industry Market Cap",
        filename_stem="streamlit_share_industry", return_fig=True,
    )
    if fig2:
        st.pyplot(fig2, clear_figure=True)

# --------------------------------------------------------------------------
# Tab 3: Linear Regression -- Y is always a company/sub-industry/industry
# series; X may be any stock/index/average series (any sub-industry) OR a
# FRED factor, but FRED factors are restricted to Y's own sub-industry.
# --------------------------------------------------------------------------
with tab_reg:
    st.subheader("Regress one or more series against a dependent variable")
    st.caption("FRED macro factors are restricted to the same sub-industry as the dependent variable (Y), per project spec. Stock/index series may be compared across sub-industries.")

    col1, col2, col3 = st.columns([2, 3, 1])
    with col1:
        y_label = st.selectbox("Dependent variable (Y)", all_labels, index=0, key="reg_y")
    with col2:
        y_sub = catalog[y_label].get("sub_industry")
        stock_choices = [l for l in F.stock_labels(catalog) if l != y_label]
        fred_choices = F.allowed_fred_labels_for(catalog, y_sub) if y_sub else list(F.allowed_fred_labels_for(catalog, "Aviation")) + list(F.allowed_fred_labels_for(catalog, "Shipbuilding"))
        x_choices = stock_choices + fred_choices
        x_default = x_choices[:3]
        x_labels = st.multiselect("Independent variable(s) (X)", x_choices, default=x_default, key="reg_x")
    with col3:
        reg_freq = st.selectbox("Frequency", ["Daily", "Monthly"], index=1, key="reg_freq")

    freq_code = {"Daily": "D", "Monthly": "M"}[reg_freq]

    if not x_labels:
        st.info("Pick one or more X series above to regress against Y.")
    else:
        y_series = catalog[y_label]["series"]
        results = []
        for x_label in x_labels:
            x_series = catalog[x_label]["series"]
            result = R.linear_regression(x_series, y_series, freq_code, x_name=x_label, y_name=y_label)
            if result is None:
                st.warning(f"Skipped **{x_label}**: fewer than 3 overlapping data points at {reg_freq.lower()} frequency.")
                continue
            results.append((x_label, result))

        if results:
            summary_df = pd.DataFrame([{
                "X variable": x_label, "Slope": r["slope"], "Intercept": r["intercept"],
                "R²": r["r_squared"], "P-value": r["p_value"], "N": r["n"],
            } for x_label, r in results]).sort_values("R²", ascending=False).reset_index(drop=True)

            st.dataframe(
                summary_df.style.format({"Slope": "{:.4g}", "Intercept": "{:.4g}", "R²": "{:.3f}", "P-value": "{:.3g}"}),
                use_container_width=True,
            )

            st.markdown("#### Scatter plots with fitted line")
            n_cols = 2
            cols = st.columns(n_cols)
            for i, (x_label, r) in enumerate(results):
                with cols[i % n_cols]:
                    fig = C.plot_regression(f"{y_label} vs {x_label}", r, return_fig=True)
                    st.pyplot(fig, clear_figure=True)

# --------------------------------------------------------------------------
# Tab 4: Multiple Regression -- same sub-industry restriction on FRED
# factors as the linear-regression tab.
# --------------------------------------------------------------------------
with tab_mreg:
    st.subheader("Multiple regression: one Y against several X variables at once")
    st.caption("Same sub-industry restriction on FRED macro factors as the Linear Regression tab.")

    col1, col2, col3 = st.columns([2, 3, 1])
    with col1:
        y_label_m = st.selectbox("Dependent variable (Y)", all_labels, index=0, key="mreg_y")
    with col2:
        y_sub_m = catalog[y_label_m].get("sub_industry")
        stock_choices_m = [l for l in F.stock_labels(catalog) if l != y_label_m]
        fred_choices_m = F.allowed_fred_labels_for(catalog, y_sub_m) if y_sub_m else list(F.allowed_fred_labels_for(catalog, "Aviation")) + list(F.allowed_fred_labels_for(catalog, "Shipbuilding"))
        x_choices_m = stock_choices_m + fred_choices_m
        x_labels_m = st.multiselect("Independent variables (X) -- pick 2 or more", x_choices_m, default=x_choices_m[:3], key="mreg_x")
    with col3:
        mreg_freq = st.selectbox("Frequency", ["Daily", "Monthly"], index=1, key="mreg_freq")

    freq_code_m = {"Daily": "D", "Monthly": "M"}[mreg_freq]

    if len(x_labels_m) < 2:
        st.info("Pick at least two X series for a multiple regression.")
    else:
        x_dict = {x: catalog[x]["series"] for x in x_labels_m}
        y_series_m = catalog[y_label_m]["series"]
        result = R.multiple_regression(x_dict, y_series_m, freq_code_m)

        if result is None:
            st.warning("Insufficient overlapping data points for this combination of variables.")
        else:
            st.metric("R²", f"{result['r_squared']:.3f}")
            st.metric("Adjusted R²", f"{result['adj_r_squared']:.3f}")
            st.caption(f"n = {result['n']}")

            coef_df = pd.DataFrame([
                {"Variable": name, "Coefficient": coef, "P-value": result["p_values"][name]}
                for name, coef in result["coefficients"].items()
            ])
            st.dataframe(
                coef_df.style.format({"Coefficient": "{:.4g}", "P-value": "{:.3g}"}),
                use_container_width=True,
            )

            fig = C.plot_multiple_regression(f"{y_label_m} ~ {' + '.join(x_labels_m)}", result, return_fig=True)
            st.pyplot(fig, clear_figure=True)
