# US Defense Dashboard

Interactive comparison of US defense-industry stocks across two sub-industries
-- **Aviation** and **Shipbuilding** -- against sub-industry/industry averages,
DJIA/NASDAQ, and sub-industry-specific FRED macro factors. Built from the
checklist in `US defense dashboard v0.1.md`.

## Companies

**Aviation:** Boeing (BA), Textron (TXT), Northrop Grumman (NOC), RTX (RTX),
Honeywell (HON), Lockheed Martin (LMT), GE Aviation (GE), Astronics (ATRO)

**Shipbuilding:** General Dynamics (GD), HII (HII), BAE Systems (BAESY),
L3Harris (LHX), Austal (AUTLY, ASX fallback ASB.AX), Vision Marine
Technologies (VMAR), Kirby Corporation (KEX)

## Metrics per company

Stock price, P/B, P/E, Market Cap, share of sub-industry market cap, share
of combined industry market cap.

## FRED macro factors

Aviation and Shipbuilding each have their own set of FRED series (see
`macro.py`). **A company is only ever regressed against its own
sub-industry's FRED factors** -- this is enforced in `frames.py`
(`allowed_fred_labels_for`) and in the Streamlit regression tabs, per the
checklist's explicit rule. Stock-vs-stock comparisons are not restricted by
sub-industry.

## Entry points

- **`main.py`** -- the Streamlit app (`streamlit run main.py`). Loads a
  pre-built snapshot (`data/snapshot.parquet` + `data/snapshot_meta.json`)
  by default; a sidebar button opts into a live Yahoo Finance / FRED fetch.
- **`pipeline.py`** -- CLI pipeline (`python pipeline.py`), saves static
  matplotlib charts to `output/` (gitignored, regenerable).
- **`build_snapshot.py`** -- run locally to refresh the data snapshot:
  ```
  python build_snapshot.py
  git add data/snapshot.parquet data/snapshot_meta.json
  git commit -m "Refresh data snapshot"
  git push
  ```

## Setup

```
pip install -r requirements.txt
cp .env.example .env   # then fill in FRED_API_KEY
python pipeline.py     # or: streamlit run main.py
```

## Notes

- All prices are fetched via yfinance with `period="max"`, so every run is
  backed up through the most recent completed trading session.
- Market Cap = Close price x `sharesOutstanding` (a constant, latest-known
  value -- same simplification as the P/B and P/E ratios).
- "Dead stock" handling: a company's contribution to any sub-industry or
  industry average/sum is dropped (not held flat) past its own last
  observed trading date, so a delisted company doesn't silently flat-line
  an average forever (see `data.compute_industry_averages`).
