"""
regression.py
-------------
Linear and multiple regression for the US defense dashboard: any company
metric (Price, P/B, P/E, Market Cap) or sub-industry/industry average
regressed against another stock series or an allowed FRED factor. Simple
fits use scipy.stats.linregress; multiple regression uses statsmodels OLS
(gives per-coefficient p-values for a proper summary table).

Frequency alignment (resampling both sides to a common D/M/Q frequency
before joining) follows the same pattern as the sibling ford-global-eval
project -- FRED series are monthly and stock data is daily, so a direct
join without resampling would silently produce near-zero overlap.
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


def _resample_mean(series, freq):
    """Resample to a common frequency via mean, stripping tz first (stocks are
    tz-aware, FRED series are tz-naive -- joining mismatched tz silently
    yields zero overlap instead of raising)."""
    if series is None or series.empty:
        return None
    if series.index.tz is not None:
        series = series.tz_localize(None)
    freq_map = {"M": "ME", "Q": "QE", "D": "D"}
    pandas_freq = freq_map.get(freq, freq)
    resampled = series.resample(pandas_freq).mean().dropna()
    return resampled if not resampled.empty else None


def simple_linear_regression(x_data, y_data, x_name="X", y_name="Y"):
    """Fit y = slope*x + intercept via scipy.stats.linregress."""
    mask = ~(pd.isna(x_data) | pd.isna(y_data))
    x_clean = np.array(x_data)[mask]
    y_clean = np.array(y_data)[mask]

    slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
    y_pred = slope * x_clean + intercept

    return {
        "slope": slope, "intercept": intercept, "r_value": r_value,
        "r_squared": r_value ** 2, "p_value": p_value, "std_err": std_err,
        "x_clean": x_clean, "y_clean": y_clean, "y_pred": y_pred,
        "x_name": x_name, "y_name": y_name, "n": len(x_clean),
    }


def linear_regression(x_series, y_series, freq, x_name="X", y_name="Y"):
    """
    Align two series to a common frequency, then fit with
    simple_linear_regression(). Returns None if fewer than 3 overlapping points.
    """
    x_resampled = _resample_mean(x_series, freq)
    y_resampled = _resample_mean(y_series, freq)
    if x_resampled is None or y_resampled is None:
        return None

    aligned = pd.concat([x_resampled, y_resampled], axis=1, join="inner").dropna()
    aligned.columns = ["x", "y"]
    if len(aligned) < 3:
        return None

    return simple_linear_regression(aligned["x"], aligned["y"], x_name=x_name, y_name=y_name)


def multiple_regression(x_series_dict, y_series, freq):
    """
    Multiple linear regression: y ~ x1 + x2 + ... via statsmodels OLS.

    Args:
        x_series_dict (dict): {x_name: pd.Series} -- independent variables
        y_series (pd.Series): dependent variable
        freq (str): "D", "M", or "Q" -- common resample frequency for alignment

    Returns:
        dict with keys 'coefficients' (dict name->coef, includes 'const'),
        'p_values' (dict name->p), 'r_squared', 'adj_r_squared', 'n',
        'x_names', 'y_pred', 'y_actual', or None if fewer than
        (n_predictors + 2) overlapping rows survive alignment.
    """
    y_resampled = _resample_mean(y_series, freq)
    if y_resampled is None:
        return None

    x_resampled = {}
    for name, s in x_series_dict.items():
        rs = _resample_mean(s, freq)
        if rs is not None:
            x_resampled[name] = rs

    if not x_resampled:
        return None

    combined = pd.concat({"y": y_resampled, **x_resampled}, axis=1, join="inner").dropna()
    x_names = list(x_resampled.keys())

    if len(combined) < len(x_names) + 2:
        # need more observations than predictors for a meaningful fit
        return None

    X = sm.add_constant(combined[x_names])
    y = combined["y"]
    model = sm.OLS(y, X).fit()

    return {
        "coefficients": model.params.to_dict(),
        "p_values": model.pvalues.to_dict(),
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "n": int(model.nobs),
        "x_names": x_names,
        "y_pred": model.predict(X),
        "y_actual": y,
    }
