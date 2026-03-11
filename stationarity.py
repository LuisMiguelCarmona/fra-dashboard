import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss


def run_adf(series):
    """Augmented Dickey-Fuller test. Returns (statistic, p-value)."""
    s = pd.Series(series).dropna()
    stat, pvalue, _, _, _, _ = adfuller(s)
    return stat, pvalue


def run_kpss(series):
    """KPSS test for level stationarity. Returns (statistic, p-value)."""
    s = pd.Series(series).dropna()
    stat, pvalue, _, _ = kpss(s, regression="c", nlags="auto")
    return stat, pvalue


def compute_half_life(series):
    s = pd.Series(series).dropna()
    lagged = s.shift(1).dropna()
    delta = s.diff().dropna()
    aligned = pd.concat([lagged, delta], axis=1).dropna()
    aligned.columns = ["lagged", "delta"]
    X = sm.add_constant(aligned["lagged"])
    y = aligned["delta"]
    model = sm.OLS(y, X).fit()
    beta = model.params["lagged"]
    if beta >= 0:
        return np.nan
    return -np.log(2) / beta


def stationarity_table(series, name="Spread"):
    adf_stat, adf_p = run_adf(series)
    kpss_stat, kpss_p = run_kpss(series)
    hl = compute_half_life(series)
    return pd.DataFrame({"Series": [name], "ADF Stat": [round(adf_stat, 4)],"ADF p-value": [round(adf_p, 4)],"KPSS Stat": [round(kpss_stat, 4)],"KPSS p-value": [round(kpss_p, 4)],"Half-Life (days)": [round(hl, 1) if not np.isnan(hl) else np.nan]})
