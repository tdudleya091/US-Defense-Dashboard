"""
macro.py
--------
Fetches FRED (Federal Reserve Economic Data) series for the two defense
sub-industries (Aviation, Shipbuilding), using the fredapi package per user
spec (rather than raw requests, as the sibling ford-global-eval /
japan-shipbuilder-eval projects do).

Every series ID below was verified live against the FRED API
(fred.get_series_info) before being added here -- the checklist's own codes
had two typos (zero-vs-letter-O: A34SN0/A34HN0 -> A34SNO/A34HNO) and one
mislabeling (the two "nondefense aircraft and parts" line items reused the
*defense* code ADAPNO; corrected to ANAPNO/ANAPTI, the actual nondefense
series). Shipbuilding series had no codes at all in the checklist (only
descriptions) -- IDs below were found via fred.search() and confirmed with
fred.get_series_info().

Per the checklist's explicit rule: Aviation stocks must only be compared
against AVIATION_FRED_SERIES, and Shipbuilding stocks only against
SHIPBUILDING_FRED_SERIES -- never across sub-industries. See
frames.allowed_fred_labels_for().
"""

import os
import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv

load_dotenv()

FRED_API_KEY = os.environ.get("FRED_API_KEY", "")

_fred = None


def _client():
    """Lazily construct the Fred client so import doesn't fail with no key set."""
    global _fred
    if _fred is None:
        if not FRED_API_KEY:
            raise RuntimeError("FRED_API_KEY not set (see .env.example)")
        _fred = Fred(api_key=FRED_API_KEY)
    return _fred


# label -> (FRED series ID, y-axis unit, chart title)
# Units below follow "patch v3.md" exactly -- each parenthetical in that
# checklist is the intended y-axis label for that variable's graph.
AVIATION_FRED_SERIES = {
    "new_computer_orders":                ("A34SNO", "Millions of USD (Seasonally Adjusted)", "New Orders: Computers and Electronic Products"),
    "new_electronics_orders":             ("A34HNO", "Millions of USD", "New Orders: Other Electronic Component Manufacturing"),
    "computer_electronics_shipments":     ("A34SVS", "Millions of USD", "Value of Shipments: Computers and Electronic Products"),
    "new_machinery_orders":               ("A33SNO", "Millions of USD", "New Orders: Machinery"),
    "defense_aircraft_parts_orders":      ("ADAPNO", "Millions of USD", "New Orders: Defense Aircraft and Parts"),
    "defense_aircraft_parts_inventories": ("ADAPTI", "Millions of USD", "Total Inventories: Defense Aircraft and Parts"),
    "defense_aircraft_parts_shipments":   ("ADAPVS", "Millions of USD", "Value of Shipments: Defense Aircraft and Parts"),
    "defense_capital_orders":             ("ADEFNO", "Millions of USD", "New Orders: Defense Capital Goods"),
    "defense_aircraft_parts_unfilled":    ("ADAPUO", "Millions of USD", "Unfilled Orders: Defense Aircraft and Parts"),
    "nondefense_aircraft_parts_orders":       ("ANAPNO", "Millions of USD", "New Orders: Nondefense Aircraft and Parts"),
    "nondefense_aircraft_parts_shipments":    ("ANAPVS", "Millions of USD", "Value of Shipments: Nondefense Aircraft and Parts"),
    "nondefense_aircraft_parts_unfilled":     ("ANAPUO", "Millions of USD", "Unfilled Orders: Nondefense Aircraft and Parts"),
    "nondefense_aircraft_parts_inventories":  ("ANAPTI", "Millions of USD", "Total Inventories: Nondefense Aircraft and Parts"),
    # annual (not monthly like the rest of this dict) -- see frames.FRED_FREQ override
    "aerospace_employment": ("IPUEN3364W200000000", "Thous. of Jobs",
                              "Employment for Manufacturing: Aerospace Product and Parts Manufacturing (NAICS 3364) in the United States"),
}

SHIPBUILDING_FRED_SERIES = {
    "industrial_machinery_shipments": ("A33EVS", "Millions of USD (Monthly)", "Value of Shipments: Industrial Machinery Manufacturing"),
    "defense_capital_goods_shipments": ("ADEFVS", "Millions of USD (Monthly, Seasonally Adjusted)", "Value of Shipments: Defense Capital Goods"),
    "ship_new_orders":                 ("A36ZNO", "Millions of USD (Seasonally Adjusted)", "New Orders: Ships and Boats (SA)"),
    "ships_boats_unfilled_orders_nsa": ("U36ZUO", "Millions of USD (Not Seasonally Adjusted)", "Unfilled Orders: Ships and Boats (NSA)"),
    "ships_boats_shipments_nsa":       ("U36ZVS", "Millions of USD (Not Seasonally Adjusted)", "Value of Shipments: Ships and Boats (NSA)"),
    "ship_boat_building_employees":    ("CES3133660001", "Thousands of People", "Employees: Ship and Boat Building"),
    # explicit label per checklist: "Label this as Ship and Boat building
    # Production Capacity compared to 2017"
    "ship_boat_building_production":   ("IPG3366S", "Index 2017=100", "Ship and Boat Building Production Capacity compared to 2017"),
    "ships_boats_unfilled_orders_sa":  ("A36ZUO", "Millions of USD (Seasonally Adjusted)", "Unfilled Orders: Ships and Boats (SA)"),
    "metal_new_orders":                ("A31SNO", "Mil. of $", "New Orders: Primary Metals"),
    "machinery_new_orders":            ("A33SNO", "Millions of USD", "New Orders: Machinery"),
}

ALL_FRED_SERIES = {**AVIATION_FRED_SERIES, **SHIPBUILDING_FRED_SERIES}


def fetch_fred_series(series_id, start_date="2010-01-01"):
    """
    Download one observation series via fredapi.

    Returns:
        pd.Series indexed by DatetimeIndex, or None on any failure. Never raises.
    """
    try:
        client = _client()
        series = client.get_series(series_id, observation_start=start_date)
    except Exception as e:
        print(f"WARNING: FRED fetch failed for {series_id}: {e}")
        return None

    if series is None or series.empty:
        print(f"WARNING: FRED returned no observations for {series_id}")
        return None

    series = series.dropna()
    series.name = series_id
    return series if not series.empty else None


def fetch_all_fred_series(start_date="2010-01-01"):
    """
    Download every series in ALL_FRED_SERIES.

    Returns:
        dict: {label: pd.Series or None}
    """
    results = {}
    for label, (series_id, _unit, _title) in ALL_FRED_SERIES.items():
        print(f"[FETCH] FRED {label} ({series_id}) ...")
        results[label] = fetch_fred_series(series_id, start_date=start_date)
    return results
