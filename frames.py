"""
frames.py
---------
Flattens the raw fetched data (pipeline.fetch_all_raw()'s output) into one
catalog of {label: {"series", "category", "freq", "sub_industry"}}, plus
helpers to reshape a subset into wide DataFrames for the Streamlit UI.

The "sub_industry" tag on every entry is what lets main.py enforce the
checklist's explicit rule -- "Do not compare against non-sub-industry
federal reserve factors" -- via allowed_fred_labels_for(): an Aviation
company may only be regressed against Aviation FRED series (or any stock,
either sub-industry), and likewise for Shipbuilding.
"""

import pandas as pd

import tickers as T
import macro as M

FRED_FREQ = {label: "M" for label in M.ALL_FRED_SERIES}
# ship_boat_building_production (IPG3366S) and a couple of employment series
# are monthly too, so the blanket "M" default above covers everything --
# nothing here is quarterly or daily.


def _strip_tz(series):
    if series is None or series.empty:
        return series
    if series.index.tz is not None:
        return pd.Series(series.values, index=series.index.tz_localize(None))
    return series


def build_catalog(company_data, index_data, sub_industry_avg, industry_avg, fred_data):
    """
    Args:
        company_data (dict): {company_name: record} from pipeline.fetch_company()
        index_data (dict): {index_name: {"close": pd.Series}} from pipeline.fetch_indices()
        sub_industry_avg (dict): {"Aviation"/"Shipbuilding": {"price":.., "pb":.., "pe":.., "market_cap":..}}
        industry_avg (dict): {"price":.., "pb":.., "pe":.., "market_cap":..} -- combined both sub-industries
        fred_data (dict): output of macro.fetch_all_fred_series()

    Returns:
        dict: {label: {"series", "category", "freq", "sub_industry"}}
        Entries with no usable data are omitted.
    """
    catalog = {}

    def _add(label, series, category, freq="D", sub_industry=None):
        series = _strip_tz(series)
        if series is None or series.empty:
            return
        catalog[label] = {"series": series, "category": category, "freq": freq, "sub_industry": sub_industry}

    # --- companies ---
    for name, record in company_data.items():
        sub = T.SUB_INDUSTRY_MAP.get(name)
        _add(f"{name} — Price (USD)", record.get("close_usd"), f"Stock Price ({sub})", sub_industry=sub)
        _add(f"{name} — P/B", record.get("pb"), f"P/B Ratio ({sub})", sub_industry=sub)
        _add(f"{name} — P/E", record.get("pe"), f"P/E Ratio ({sub})", sub_industry=sub)
        _add(f"{name} — Market Cap ($)", record.get("market_cap"), f"Market Cap ({sub})", sub_industry=sub)
        _add(f"{name} — % of Sub-Industry Market Cap", record.get("share_of_sub_industry"),
             f"Market Cap % ({sub})", sub_industry=sub)
        _add(f"{name} — % of Industry Market Cap", record.get("share_of_industry"),
             f"Market Cap % ({sub})", sub_industry=sub)

    # --- benchmark indices ---
    for idx_name, info in index_data.items():
        _add(f"{idx_name}", info.get("close"), "Benchmark Index")

    # --- sub-industry averages ---
    for sub, metrics in sub_industry_avg.items():
        _add(f"{sub} Avg — Price (USD)", metrics.get("price"), f"Stock Price ({sub})", sub_industry=sub)
        _add(f"{sub} Avg — P/B", metrics.get("pb"), f"P/B Ratio ({sub})", sub_industry=sub)
        _add(f"{sub} Avg — P/E", metrics.get("pe"), f"P/E Ratio ({sub})", sub_industry=sub)
        _add(f"{sub} — Total Market Cap ($)", metrics.get("market_cap"), f"Market Cap ({sub})", sub_industry=sub)
        _add(f"{sub} — % of Industry Market Cap", metrics.get("share_of_industry"),
             f"Market Cap % ({sub})", sub_industry=sub)

    # --- combined industry average (both sub-industries) ---
    _add("Industry Avg — Price (USD)", industry_avg.get("price"), "Stock Price (Industry)")
    _add("Industry Avg — P/B", industry_avg.get("pb"), "P/B Ratio (Industry)")
    _add("Industry Avg — P/E", industry_avg.get("pe"), "P/E Ratio (Industry)")
    _add("Industry — Total Market Cap ($)", industry_avg.get("market_cap"), "Market Cap (Industry)")

    # --- FRED macro series, tagged with the sub-industry that's allowed to use them ---
    # A couple of series (e.g. "New Orders: Machinery" / A33SNO) are listed
    # under BOTH sub-industries in the checklist -- always suffix the title
    # with its sub-industry so those don't collide on the same catalog key
    # (which would silently drop one sub-industry's access to that factor).
    for label, series in fred_data.items():
        is_aviation = label in M.AVIATION_FRED_SERIES
        sub = "Aviation" if is_aviation else "Shipbuilding"
        _, unit, title = M.ALL_FRED_SERIES[label]
        freq = FRED_FREQ.get(label, "M")
        _add(f"{title} ({sub})", series, f"Macro (FRED) - {sub}", freq=freq, sub_industry=sub)

    return catalog


def allowed_fred_labels_for(catalog, sub_industry):
    """
    Return the catalog labels of FRED series permitted for a company/average
    in the given sub-industry, per the checklist's "do not compare against
    non-sub-industry federal reserve factors" rule.
    """
    return sorted(
        label for label, entry in catalog.items()
        if entry["category"].startswith("Macro (FRED)") and entry["sub_industry"] == sub_industry
    )


def stock_labels(catalog):
    """All stock/index/average labels (i.e. everything that is NOT a FRED macro series) --
    these may be compared against companies of ANY sub-industry."""
    return sorted(
        label for label, entry in catalog.items()
        if not entry["category"].startswith("Macro (FRED)")
    )


def _resample(series, freq):
    if freq == "D":
        return series
    pandas_freq = {"M": "ME", "Q": "QE"}.get(freq, freq)
    return series.resample(pandas_freq).mean().dropna()


def wide_frame(catalog, labels, freq=None, normalize=False):
    """
    Build a wide DataFrame (one column per label) for a multi-series
    Streamlit line chart, optionally resampled to a common frequency and/or
    normalized to indexed-100 at each column's first valid observation.
    """
    columns = {}
    for label in labels:
        entry = catalog.get(label)
        if entry is None:
            continue
        series = entry["series"]
        if freq is not None:
            series = _resample(series, freq)
        columns[label] = series

    if not columns:
        return pd.DataFrame()

    df = pd.concat(columns, axis=1).sort_index()
    if freq is None:
        df = df.ffill()

    if normalize:
        first_valid = df.apply(lambda col: col.dropna().iloc[0] if col.notna().any() else None)
        df = df.apply(lambda col: (col / first_valid[col.name]) * 100 if first_valid[col.name] else col)

    return df
