import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss


def run_pca(curve_df):
    X = curve_df[["1y", "2y", "5y", "10y"]].dropna()

    pca = PCA(n_components=3)
    pcs = pca.fit_transform(X)

    pca_df = curve_df.loc[X.index, ["closeDate"]].copy()
    pca_df["Level"] = pcs[:, 0]
    pca_df["Slope"] = pcs[:, 1]
    pca_df["Curvature"] = pcs[:, 2]

    loadings = pd.DataFrame(pca.components_.T,index=["1y", "2y", "5y", "10y"],columns=["Level", "Slope", "Curvature"])

    explained = pd.DataFrame({"Component": ["Level", "Slope", "Curvature"],"Explained Variance": pca.explained_variance_ratio_})

    for pc in ["Level", "Slope", "Curvature"]:
        mean = pca_df[pc].mean()
        std = pca_df[pc].std()
        pca_df[f"{pc}_z"] = (pca_df[pc] - mean) / std

    return pca_df, loadings, explained

def add_rolling_zscore(df, cols, windows=[60, 120, 252]):
    out = df.copy()
    for col in cols:
        for w in windows:
            rolling_mean = out[col].rolling(w).mean()
            rolling_std = out[col].rolling(w).std()
            out[f"{col}_z_{w}d"] = (out[col] - rolling_mean) / rolling_std

    return out

def run_adf(series):
    s = pd.Series(series).dropna()
    stat, pvalue, _, _, _, _ = adfuller(s)
    return stat, pvalue

def run_kpss(series):
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

    half_life = -np.log(2) / beta
    return half_life


def stationarity_table(series, name="Spread"):
    adf_stat, adf_p = run_adf(series)
    kpss_stat, kpss_p = run_kpss(series)
    hl = compute_half_life(series)
    table = pd.DataFrame({"Series": [name],"ADF Stat": [adf_stat],"ADF p-value": [adf_p],"KPSS Stat": [kpss_stat],"KPSS p-value": [kpss_p],"Half-Life": [hl]})
    return table
