"""
build_snapshot.py
------------------
Runs the full fetch pipeline (same one pipeline.py and main.py use) and
saves the resulting catalog to data/snapshot.parquet + data/snapshot_meta.json.

Why a snapshot at all: Yahoo Finance commonly rate-limits or blocks requests
from shared cloud datacenter IPs (which is what Streamlit Community Cloud
runs on), even though the same fetches complete in well under a second run
locally. main.py (the Streamlit app) loads this pre-built snapshot by
default (instant, reliable) and only attempts a live fetch if asked to via
the sidebar button.

Why Parquet+JSON instead of a pickle: a pickled pandas object isn't
reliably readable by a different pandas major version, and requirements.txt
doesn't pin pandas -- so a snapshot pickled locally can fail to load once
Streamlit Cloud installs a newer pandas at deploy time. Parquet is a stable
columnar format independent of pandas' internal version.

Usage:
    python build_snapshot.py

Re-run this locally and push the two data/ files whenever you want the
deployed app to show newer data:

    python build_snapshot.py
    git add data/snapshot.parquet data/snapshot_meta.json
    git commit -m "Refresh data snapshot"
    git push
"""

import json
import os
from datetime import datetime, timezone

import pandas as pd

import pipeline as MN
import frames as F

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PARQUET_PATH = os.path.join(DATA_DIR, "snapshot.parquet")
META_PATH = os.path.join(DATA_DIR, "snapshot_meta.json")


def main():
    print("=== Building data snapshot for the Streamlit app ===\n")

    company_data, index_data, sub_industry_avg, industry_avg, fred_data = MN.fetch_all_raw()
    catalog = F.build_catalog(company_data, index_data, sub_industry_avg, industry_avg, fred_data)

    labels = list(catalog.keys())
    wide_df = pd.concat({label: catalog[label]["series"] for label in labels}, axis=1)

    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        "series": {
            label: {
                "category": catalog[label]["category"],
                "freq": catalog[label]["freq"],
                "sub_industry": catalog[label]["sub_industry"],
            }
            for label in labels
        },
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    wide_df.to_parquet(PARQUET_PATH)
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[DONE] Snapshot saved: {PARQUET_PATH} + {META_PATH} ({len(labels)} series)")


if __name__ == "__main__":
    main()
