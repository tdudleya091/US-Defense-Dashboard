"""
pipeline.py
-----------
CLI entry point for the US Defense Dashboard: fetches every Aviation and
Shipbuilding company (yfinance) plus FRED macro data, computes sub-industry
and industry averages/market-cap shares, generates the required static
matplotlib charts, and runs the linear + multiple regressions.

fetch_all_raw() -- the fetch step, no chart/regression side effects -- is
also imported by main.py (the Streamlit app) and build_snapshot.py, so all
three share one fetch implementation (same pattern as the sibling
ford-global-eval project).

(Named pipeline.py rather than main.py so main.py can be reserved for the
Streamlit app -- Streamlit Community Cloud's "Main file path" setting isn't
always reachable/editable after the fact.)

Usage:
    python pipeline.py
"""

import tickers as T
import data as D
import macro as M
import charts as C
import regression as R
import frames as F

SUB_INDUSTRIES = ["Aviation", "Shipbuilding"]


def fetch_company(name, fx_cache):
    """
    Fetch price history (resolving ADR/local fallback candidates where
    applicable), convert to USD, and compute P/B, P/E, Market Cap.

    Returns:
        dict record, or None if no ticker or no data could be fetched.
    """
    if name in T.TICKER_CANDIDATES:
        candidates = T.TICKER_CANDIDATES[name]
        hist_df, ticker_obj, currency, ticker_symbol = D.resolve_best_ticker(candidates)
    else:
        ticker_symbol = T.TICKERS.get(name)
        if ticker_symbol is None:
            print(f"[SKIP] {name}: ticker is None, skipping")
            return None
        currency = T.CURRENCY_MAP.get(name, "USD")
        print(f"[FETCH] {name} ({ticker_symbol}, {currency}) ...")
        hist_df, ticker_obj = D.fetch_ticker_data(ticker_symbol)

    if hist_df is None or ticker_obj is None:
        print(f"[SKIP] {name}: no usable price data from any candidate ticker")
        return None

    close_usd = D.convert_to_usd(hist_df["Close"], currency, fx_cache.get(currency))
    if close_usd is None:
        print(f"[SKIP] {name}: USD conversion failed; omitted from USD comparisons")

    pb_series = D.get_pb_series(hist_df, ticker_obj)
    pe_series = D.get_pe_series(hist_df, ticker_obj)
    market_cap_local = D.get_market_cap_series(hist_df, ticker_obj)
    market_cap_usd = D.convert_to_usd(market_cap_local, currency, fx_cache.get(currency)) if market_cap_local is not None else None

    return {
        "ticker_symbol": ticker_symbol,
        "currency": currency,
        "hist_local": hist_df,
        "ticker_obj": ticker_obj,
        "close_usd": close_usd,
        "pb": pb_series,
        "pe": pe_series,
        "market_cap": market_cap_usd,
    }


def fetch_indices():
    """Fetch DJIA and NASDAQ Composite (both already USD-native)."""
    index_data = {}
    for idx_name, idx_ticker in T.INDICES.items():
        print(f"[FETCH] Index {idx_name} ({idx_ticker}) ...")
        hist_df, _ = D.fetch_ticker_data(idx_ticker)
        if hist_df is None:
            print(f"[SKIP] Index {idx_name}: no price data")
            continue
        index_data[idx_name] = {"close": hist_df["Close"]}
    return index_data


def fetch_all_raw(start_date="2010-01-01"):
    """
    Run the full fetch pipeline (Austal's AUD fallback FX, every company,
    both benchmark indices, sub-industry + combined-industry averages and
    market-cap shares, FRED macro series).

    Returns:
        tuple: (company_data, index_data, sub_industry_avg, industry_avg, fred_data)
    """
    print("--- Fetching FX (AUD, for Austal's ASX fallback only) ---")
    fx_cache = {"AUD": D.get_fx_series("AUD")}

    print("\n--- Fetching company price data ---")
    company_data = {}
    for name in T.ALL_COMPANIES:
        try:
            record = fetch_company(name, fx_cache)
        except Exception as e:
            print(f"WARNING: Unexpected error fetching {name}: {e}")
            record = None
        if record is not None:
            company_data[name] = record
            print(f"  [OK] {name} ({record['ticker_symbol']}, {record['currency']})")

    print("\n--- Fetching benchmark indices ---")
    index_data = fetch_indices()

    print("\n--- Computing sub-industry and industry averages / market-cap shares ---")
    sub_industry_avg = {}
    sub_industry_mcap_total = {}
    for sub in SUB_INDUSTRIES:
        names = [n for n in T.ALL_COMPANIES if T.SUB_INDUSTRY_MAP.get(n) == sub and n in company_data]
        close_dict = {n: company_data[n]["close_usd"] for n in names}
        pb_dict = {n: company_data[n]["pb"] for n in names if company_data[n]["pb"] is not None}
        pe_dict = {n: company_data[n]["pe"] for n in names if company_data[n]["pe"] is not None}
        mcap_dict = {n: company_data[n]["market_cap"] for n in names if company_data[n]["market_cap"] is not None}

        mcap_total = D.compute_industry_sum(mcap_dict, metric_key=f"{sub} market cap")
        sub_industry_avg[sub] = {
            "price": D.compute_industry_averages(close_dict, metric_key=f"{sub} close"),
            "pb": D.compute_industry_averages(pb_dict, metric_key=f"{sub} P/B"),
            "pe": D.compute_industry_averages(pe_dict, metric_key=f"{sub} P/E"),
            "market_cap": mcap_total,
        }
        sub_industry_mcap_total[sub] = mcap_total

    # combined industry total market cap = sum of both sub-industry totals
    all_mcap = {n: r["market_cap"] for n, r in company_data.items() if r["market_cap"] is not None}
    industry_mcap_total = D.compute_industry_sum(all_mcap, metric_key="industry market cap")

    all_close = {n: r["close_usd"] for n, r in company_data.items() if r["close_usd"] is not None}
    all_pb = {n: r["pb"] for n, r in company_data.items() if r["pb"] is not None}
    all_pe = {n: r["pe"] for n, r in company_data.items() if r["pe"] is not None}
    industry_avg = {
        "price": D.compute_industry_averages(all_close, metric_key="industry close"),
        "pb": D.compute_industry_averages(all_pb, metric_key="industry P/B"),
        "pe": D.compute_industry_averages(all_pe, metric_key="industry P/E"),
        "market_cap": industry_mcap_total,
    }

    # per-company share of sub-industry / industry market cap, and each
    # sub-industry's share of the combined industry market cap. mcap is
    # tz-aware (straight from yfinance) while the sub/industry totals coming
    # out of compute_industry_sum are tz-naive (stripped internally) --
    # strip tz here too so the division aligns instead of raising.
    for name, record in company_data.items():
        sub = T.SUB_INDUSTRY_MAP.get(name)
        mcap = F._strip_tz(record["market_cap"])
        sub_total = sub_industry_mcap_total.get(sub)
        record["share_of_sub_industry"] = (
            (mcap / sub_total * 100).dropna() if mcap is not None and sub_total is not None else None
        )
        record["share_of_industry"] = (
            (mcap / industry_mcap_total * 100).dropna() if mcap is not None and industry_mcap_total is not None else None
        )

    for sub in SUB_INDUSTRIES:
        sub_total = sub_industry_mcap_total.get(sub)
        sub_industry_avg[sub]["share_of_industry"] = (
            (sub_total / industry_mcap_total * 100).dropna() if sub_total is not None and industry_mcap_total is not None else None
        )

    print("\n--- Fetching FRED macro data ---")
    fred_data = M.fetch_all_fred_series(start_date=start_date)

    return company_data, index_data, sub_industry_avg, industry_avg, fred_data


def main():
    C.ensure_output_dir()
    print("=== US Defense Dashboard ===\n")

    company_data, index_data, sub_industry_avg, industry_avg, fred_data = fetch_all_raw()

    catalog = F.build_catalog(company_data, index_data, sub_industry_avg, industry_avg, fred_data)
    print(f"\n[CATALOG] {len(catalog)} plottable series")

    # --- standalone company price charts, one per sub-industry ---
    print("\n--- Generating sub-industry stock price comparison charts ---")
    for sub in SUB_INDUSTRIES:
        names = [n for n in T.ALL_COMPANIES if T.SUB_INDUSTRY_MAP.get(n) == sub and n in company_data]
        series_dict = {n: company_data[n]["close_usd"] for n in names}
        series_dict[f"{sub} Avg"] = sub_industry_avg[sub]["price"]
        series_dict["DJIA"] = index_data.get("Dow Jones Industrial Average", {}).get("close")
        series_dict["NASDAQ"] = index_data.get("NASDAQ Composite", {}).get("close")
        C.plot_timeseries_comparison(
            series_dict, title=f"{sub} Stock Price Comparison (USD)", y_label="Close Price (USD)",
            filename_stem=f"{sub}_price_comparison", normalize=True,
        )

    # --- P/B, P/E comparison charts per sub-industry ---
    for sub in SUB_INDUSTRIES:
        names = [n for n in T.ALL_COMPANIES if T.SUB_INDUSTRY_MAP.get(n) == sub and n in company_data]
        for metric, key in [("P/B", "pb"), ("P/E", "pe")]:
            series_dict = {n: company_data[n][key] for n in names if company_data[n][key] is not None}
            series_dict[f"{sub} Avg"] = sub_industry_avg[sub][key]
            series_dict["Industry Avg"] = industry_avg[key]
            C.plot_timeseries_comparison(
                series_dict, title=f"{sub} {metric} Comparison", y_label=f"{metric} Ratio",
                filename_stem=f"{sub}_{key}_comparison",
            )

    # --- market-cap share charts: percent (bar + pie) AND total dollar value (bar + pie) ---
    print("\n--- Generating market-cap share charts ---")
    for sub in SUB_INDUSTRIES:
        names = [n for n in T.ALL_COMPANIES if T.SUB_INDUSTRY_MAP.get(n) == sub and n in company_data]
        pct_labels, pct_values = [], []
        value_labels, value_values = [], []
        for n in names:
            share = company_data[n].get("share_of_sub_industry")
            if share is not None and not share.empty:
                pct_labels.append(n)
                pct_values.append(share.dropna().iloc[-1])
            mcap = company_data[n].get("market_cap")
            if mcap is not None and not mcap.dropna().empty:
                value_labels.append(n)
                value_values.append(mcap.dropna().iloc[-1])

        C.plot_market_share_bar(pct_labels, pct_values, title=f"{sub}: % of Sub-Industry Market Cap (latest)",
                                 filename_stem=f"{sub}_share_of_subindustry")
        C.plot_market_share_pie(pct_labels, pct_values, title=f"{sub}: % of Sub-Industry Market Cap (latest)",
                                 filename_stem=f"{sub}_share_of_subindustry")
        C.plot_market_value_bar(value_labels, value_values, title=f"{sub}: Total Market Cap by Company (latest)",
                                 filename_stem=f"{sub}_market_cap")
        C.plot_market_value_pie(value_labels, value_values, title=f"{sub}: Total Market Cap by Company (latest)",
                                 filename_stem=f"{sub}_market_cap")

    industry_pct_labels, industry_pct_values = [], []
    industry_value_labels, industry_value_values = [], []
    for sub in SUB_INDUSTRIES:
        share = sub_industry_avg[sub].get("share_of_industry")
        if share is not None and not share.empty:
            industry_pct_labels.append(sub)
            industry_pct_values.append(share.dropna().iloc[-1])
        mcap = sub_industry_avg[sub].get("market_cap")
        if mcap is not None and not mcap.dropna().empty:
            industry_value_labels.append(sub)
            industry_value_values.append(mcap.dropna().iloc[-1])

    C.plot_market_share_bar(industry_pct_labels, industry_pct_values,
                             title="Sub-Industry % of Combined Defense-Industry Market Cap (latest)",
                             filename_stem="industry_subindustry_shares")
    C.plot_market_share_pie(industry_pct_labels, industry_pct_values,
                             title="Sub-Industry % of Combined Defense-Industry Market Cap (latest)",
                             filename_stem="industry_subindustry_shares")
    C.plot_market_value_bar(industry_value_labels, industry_value_values,
                             title="Sub-Industry Total Market Cap (latest)",
                             filename_stem="industry_subindustry_market_cap_total")
    C.plot_market_value_pie(industry_value_labels, industry_value_values,
                             title="Sub-Industry Total Market Cap (latest)",
                             filename_stem="industry_subindustry_market_cap_total")

    # --- standalone FRED macro series charts ---
    print("\n--- Generating standalone FRED macro series charts ---")
    for label, series in fred_data.items():
        _series_id, unit, title = M.ALL_FRED_SERIES[label]
        C.plot_macro_series(label, series, y_label=unit, title=title)

    # --- linear regressions: each company's price vs every allowed FRED
    # factor in its OWN sub-industry only ---
    print("\n--- Running linear regressions (company price vs sub-industry FRED factors) ---")
    for name, record in company_data.items():
        sub = T.SUB_INDUSTRY_MAP.get(name)
        allowed = M.AVIATION_FRED_SERIES if sub == "Aviation" else M.SHIPBUILDING_FRED_SERIES
        for fred_label, (_sid, _unit, fred_title) in allowed.items():
            x_series = fred_data.get(fred_label)
            if x_series is None:
                continue
            result = R.linear_regression(x_series, record["close_usd"], freq="M", x_name=fred_title, y_name=f"{name} Price")
            if result is None:
                print(f"[SKIP] {name} vs {fred_title}: insufficient overlapping data")
                continue
            C.plot_regression(f"{name} vs {fred_title}", result)

    # --- one example multiple regression per sub-industry: sub-industry avg
    # price ~ all of that sub-industry's MONTHLY FRED factors. Annual
    # series (e.g. aerospace_employment) are excluded here: this regression
    # inner-joins every predictor onto one date index, so a single annual
    # series drags the whole combined fit down to ~1 usable row/year --
    # e.g. Aviation's n went from 198 to 16 (barely above the "n >=
    # predictors + 2" floor) the moment aerospace_employment joined the
    # dict, with R² collapsing into a near-meaningless overfit at that
    # sample size. The annual series is still fully available individually
    # (standalone chart above, per-company linear regression, and the
    # Streamlit multiple-regression tab where the user picks variables and
    # gets an explicit "insufficient data" warning instead of a silent
    # overfit).
    print("\n--- Running multiple regressions (sub-industry avg price ~ monthly FRED factors) ---")
    for sub in SUB_INDUSTRIES:
        allowed = M.AVIATION_FRED_SERIES if sub == "Aviation" else M.SHIPBUILDING_FRED_SERIES
        x_dict = {
            title: fred_data[label] for label, (_sid, _unit, title) in allowed.items()
            if fred_data.get(label) is not None and F.FRED_FREQ.get(label, "M") == "M"
        }
        y_series = sub_industry_avg[sub]["price"]
        if y_series is None or not x_dict:
            print(f"[SKIP] {sub} multiple regression: missing data")
            continue
        result = R.multiple_regression(x_dict, y_series, freq="M")
        if result is None:
            print(f"[SKIP] {sub} multiple regression: insufficient overlapping data")
            continue
        print(f"  {sub}: R²={result['r_squared']:.3f}, n={result['n']}")
        C.plot_multiple_regression(f"{sub} Avg Price ~ FRED factors", result)

    print(f"\n[DONE] All charts saved to {C.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
