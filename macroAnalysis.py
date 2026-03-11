"""
Macro Fair Value Analytics
Feature engineering, VIF, Granger causality, regression models, cointegration.
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


# =================== Feature Engineering ===================

def build_macro_features(macro_df, level_cols, momentum_windows=None, zscore_window=252):
    if momentum_windows is None:
        momentum_windows = [21, 63]

    out = macro_df[["closeDate"]].copy()

    for col in level_cols:
        s = macro_df[col]
        roll_mean = s.rolling(zscore_window, min_periods=60).mean()
        roll_std = s.rolling(zscore_window, min_periods=60).std()
        out[f"{col}_z"] = (s - roll_mean) / roll_std

        for w in momentum_windows:
            out[f"{col}_mom{w}d"] = s.diff(w)

    return out


# =================== VIF ===================

def compute_vif(X_df):
    X_clean = X_df.dropna()
    if len(X_clean) < 10 or len(X_clean.columns) < 2:
        return pd.DataFrame({"Feature": X_clean.columns, "VIF": [np.nan] * len(X_clean.columns)})

    X_const = sm.add_constant(X_clean)
    vif_data = []
    for i, col in enumerate(X_const.columns):
        if col == "const":
            continue
        vif_data.append({"Feature": col,"VIF": variance_inflation_factor(X_const.values, i)})
    return pd.DataFrame(vif_data).sort_values("VIF", ascending=False).reset_index(drop=True)


# =================== Granger Causality ===================
def run_granger_tests(spread_series, feature_df, feature_cols, max_lag=10, lag=5):
    results = []
    for col in feature_cols:
        try:
            data = pd.DataFrame({"spread": spread_series, col: feature_df[col]}).dropna()
            if len(data) < max_lag + 50:
                continue
            gc = grangercausalitytests(data[["spread", col]], maxlag=max_lag, verbose=False)
            f_stat = gc[lag][0]["ssr_ftest"][0]
            p_val = gc[lag][0]["ssr_ftest"][1]
        except Exception:
            lag, f_stat, p_val = np.nan, np.nan, np.nan
        results.append({
            "Feature": col,
            "Lag": lag,
            "F-stat": round(f_stat, 2) if not np.isnan(f_stat) else np.nan,
            "p-value": round(p_val, 4) if not np.isnan(p_val) else np.nan,
            "Significant (5%)": "✓" if (not np.isnan(p_val) and p_val < 0.05) else "✗"})
    return pd.DataFrame(results).sort_values("p-value").reset_index(drop=True)

# =================== Cointegration ===================

def run_engle_granger(y, X_df):
    common = y.dropna().index.intersection(X_df.dropna().index)
    y_c = y.loc[common]
    X_c = X_df.loc[common]

    if len(common) < 100:
        return {"stat": np.nan, "pvalue": np.nan}

    X_const = sm.add_constant(X_c)
    model = sm.OLS(y_c, X_const).fit()
    residuals = model.resid

    adf_stat, adf_p, _, _, crit, _ = adfuller(residuals)

    return {"stat": adf_stat,"pvalue": adf_p,"crit_values": crit,"residuals": residuals,}


# =================== Static OLS ===================

def compute_macro_residual_ols(spread_df, features_df, feature_cols, train_end="2017-12-31"):
    """
    Spread = f(macro features), OLS fit on training only.
    Returns (df with fair_value + macro_residual, betas, stats).
    """
    train_end = pd.to_datetime(train_end)

    df = spread_df.merge(features_df[["closeDate"] + feature_cols],on="closeDate", how="inner").dropna().copy()
    df["is_train"] = df["closeDate"] <= train_end
    train = df[df["is_train"]]

    if train.empty:
        raise ValueError("Training sample is empty.")

    X_train = sm.add_constant(train[feature_cols])
    y_train = train["spread"]
    model = sm.OLS(y_train, X_train).fit()

    X_full = sm.add_constant(df[feature_cols], has_constant="add")
    df["fair_value"] = model.predict(X_full)
    df["macro_residual"] = df["spread"] - df["fair_value"]

    betas = pd.DataFrame({"Variable": model.params.index,"Beta": model.params.values,"Std Err": model.bse.values,"p-value": model.pvalues.values})

    # Test R²
    test = df[~df["is_train"]]
    r2_test = np.nan
    if len(test) > 0:
        ss_res = ((test["spread"] - test["fair_value"]) ** 2).sum()
        ss_tot = ((test["spread"] - test["spread"].mean()) ** 2).sum()
        r2_test = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    stats = {"r2_train": model.rsquared,"adj_r2_train": model.rsquared_adj,"r2_test": r2_test}

    return df, betas, stats


# =================== Rolling Ridge ===================

def compute_macro_residual_rolling_ridge(spread_df, features_df, feature_cols,window=504, alpha=1.0, min_train=252):
    df = spread_df.merge(features_df[["closeDate"] + feature_cols],on="closeDate", how="inner",).dropna().reset_index(drop=True)
    n = len(df)
    fair_values = np.full(n, np.nan)
    rolling_r2 = np.full(n, np.nan)
    betas_list = []

    for t in range(min_train, n):
        start = max(0, t - window)
        train_idx = slice(start, t-1)

        X_train = df.iloc[start:t][feature_cols].values
        y_train = df.iloc[start:t]["spread"].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        model = Ridge(alpha=alpha)
        model.fit(X_scaled, y_train)

        y_pred_train = model.predict(X_scaled)
        ss_res = ((y_train - y_pred_train) ** 2).sum()
        ss_tot = ((y_train - y_train.mean()) ** 2).sum()
        rolling_r2[t] = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        X_t = df.iloc[[t]][feature_cols].values
        X_t_scaled = scaler.transform(X_t)
        fair_values[t] = model.predict(X_t_scaled)[0]

        betas_unscaled = model.coef_ / scaler.scale_
        betas_list.append({"closeDate": df.loc[t, "closeDate"],**{col: b for col, b in zip(feature_cols, betas_unscaled)}})

    df["fair_value"] = fair_values
    df["macro_residual"] = df["spread"] - df["fair_value"]
    df["rolling_r2"] = rolling_r2

    rolling_betas = pd.DataFrame(betas_list) if betas_list else pd.DataFrame()

    return df, rolling_betas