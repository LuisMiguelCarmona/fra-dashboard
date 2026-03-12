from config import TRAIN_END
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests, adfuller, coint
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


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
            used_lag = lag
        except Exception:
            used_lag, f_stat, p_val = np.nan, np.nan, np.nan
        results.append({
            "Feature": col,
            "Lag": used_lag,
            "F-stat": round(f_stat, 2) if not np.isnan(f_stat) else np.nan,
            "p-value": round(p_val, 4) if not np.isnan(p_val) else np.nan,
            "Significant (5%)": "✓" if (not np.isnan(p_val) and p_val < 0.05) else "✗"})
    return pd.DataFrame(results).sort_values("p-value").reset_index(drop=True)

# =================== Cointegration ===================

def run_engle_granger(y, X_df):
    common = y.dropna().index.intersection(X_df.dropna().index)
    y_c = y.loc[common].values
    X_c = X_df.loc[common].values

    if len(common) < 100:
        return {"stat": np.nan, "pvalue": np.nan, "fallback": False}
    
    try:
        eg_stat, eg_pvalue, eg_crit = coint(y_c, X_c)
        return {"stat": eg_stat, "pvalue": eg_pvalue, "crit_values": eg_crit, "fallback": False}
    except (IndexError, ValueError):
        X_const = sm.add_constant(X_c)
        model = sm.OLS(y_c, X_const).fit()
        adf_stat, adf_p, _, _, crit, _ = adfuller(model.resid)
        return {"stat": adf_stat, "pvalue": adf_p, "crit_values": crit, "fallback": True}


# =================== Johansen Cointegration ===================

def run_johansen_test(spread_series, feature_df, feature_cols, det_order=0, k_ar_diff=2):
    """
    Johansen multivariate cointegration test.
    Tests whether spread + macro features share cointegrating relationships.

    det_order: -1 (no deterministic), 0 (constant), 1 (linear trend)
    k_ar_diff: number of lagged differences in the VECM

    Returns dict with trace/eigenvalue stats, critical values, n_coint.
    """
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    common = spread_series.dropna().index.intersection(feature_df.dropna().index)
    y = spread_series.loc[common].values
    X = feature_df.loc[common, feature_cols].values

    data = np.column_stack([y, X])

    if len(data) < 100:
        return {"error": "Insufficient observations", "n_coint_trace": 0, "n_coint_eigen": 0}

    try:
        result = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)

        var_names = ["spread"] + list(feature_cols)
        n_vars = len(var_names)

        # Trace test
        trace_stats = result.lr1   # trace statistics
        trace_cvs = result.cvt     # critical values (90%, 95%, 99%)

        # Max eigenvalue test
        eigen_stats = result.lr2
        eigen_cvs = result.cvm

        # Count cointegrating vectors at 5% level
        n_coint_trace = sum(1 for i in range(n_vars) if trace_stats[i] > trace_cvs[i, 1])
        n_coint_eigen = sum(1 for i in range(n_vars) if eigen_stats[i] > eigen_cvs[i, 1])

        # Build results table
        rows = []
        for i in range(n_vars):
            rows.append({
                "H0": f"r ≤ {i}",
                "Trace Stat": trace_stats[i],
                "Trace CV (5%)": trace_cvs[i, 1],
                "Trace Reject": "✓" if trace_stats[i] > trace_cvs[i, 1] else "✗",
                "Eigen Stat": eigen_stats[i],
                "Eigen CV (5%)": eigen_cvs[i, 1],
                "Eigen Reject": "✓" if eigen_stats[i] > eigen_cvs[i, 1] else "✗",
            })

        return {
            "table": pd.DataFrame(rows),
            "n_coint_trace": n_coint_trace,
            "n_coint_eigen": n_coint_eigen,
            "eigenvectors": result.evec,
            "var_names": var_names,
        }
    except Exception as e:
        return {"error": str(e), "n_coint_trace": 0, "n_coint_eigen": 0}


# =================== Static OLS ===================

def compute_macro_residual_ols(spread_df, features_df, feature_cols, train_end=TRAIN_END):
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


# =================== Lasso / Elastic Net ===================

def compute_macro_residual_lasso(spread_df, features_df, feature_cols,
                                  train_end=TRAIN_END, alpha=0.001):
    train_end = pd.to_datetime(train_end)
    df = spread_df.merge(
        features_df[["closeDate"] + feature_cols],
        on="closeDate", how="inner"
    ).dropna().copy()
    df["is_train"] = df["closeDate"] <= train_end
    train = df[df["is_train"]]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feature_cols])
    y_train = train["spread"].values

    model = Lasso(alpha=alpha, fit_intercept=True, max_iter=10000)
    model.fit(X_train, y_train)

    X_full = scaler.transform(df[feature_cols])
    df["fair_value"] = model.predict(X_full)
    df["macro_residual"] = df["spread"] - df["fair_value"]

    betas_unscaled = model.coef_ / scaler.scale_
    betas = pd.DataFrame({
        "Variable": feature_cols,
        "Beta": betas_unscaled,
        "Selected": betas_unscaled != 0,
    })

    r2_train = r2_score(y_train, model.predict(X_train))

    test = df[~df["is_train"]]
    r2_test = np.nan
    if len(test) > 0:
        ss_res = ((test["spread"] - test["fair_value"]) ** 2).sum()
        ss_tot = ((test["spread"] - test["spread"].mean()) ** 2).sum()
        r2_test = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    stats = {
        "model_type": "Lasso",
        "alpha": alpha,
        "r2_train": r2_train,
        "r2_test": r2_test,
        "n_selected": (betas_unscaled != 0).sum(),
        "n_features": len(feature_cols),
    }

    return df, betas, stats


def compare_regularization_models(spread_df, features_df, feature_cols,
                                   window=504, min_train=252):
    """
    Compare OLS, Ridge, Lasso, ElasticNet in the SAME rolling walk-forward
    framework. At each step t, trains on [t-window : t], predicts t.

    This is the only fair comparison — a static train/test split is meaningless
    for macro relationships that drift over time.

    Returns summary DataFrame + rolling OOS error DataFrame.
    """
    df = spread_df.merge(
        features_df[["closeDate"] + feature_cols],
        on="closeDate", how="inner"
    ).dropna().reset_index(drop=True)

    n = len(df)

    model_configs = {
        "OLS": lambda: Ridge(alpha=1e-10, fit_intercept=True),
        "Ridge (α=1)": lambda: Ridge(alpha=1.0, fit_intercept=True),
        "Ridge (α=10)": lambda: Ridge(alpha=10.0, fit_intercept=True),
        "Lasso (α=0.001)": lambda: Lasso(alpha=0.001, fit_intercept=True, max_iter=5000),
        "ElasticNet": lambda: ElasticNet(alpha=0.001, l1_ratio=0.5, fit_intercept=True, max_iter=5000),
    }

    # Accumulate OOS predictions per model
    oos_errors = {name: np.full(n, np.nan) for name in model_configs}
    oos_preds = {name: np.full(n, np.nan) for name in model_configs}

    for t in range(min_train, n):
        start = max(0, t - window)

        X_train = df.iloc[start:t][feature_cols].values
        y_train = df.iloc[start:t]["spread"].values

        X_t = df.iloc[[t]][feature_cols].values
        y_t = df.iloc[t]["spread"]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_t_s = scaler.transform(X_t)

        for name, model_fn in model_configs.items():
            model = model_fn()
            model.fit(X_train_s, y_train)
            pred = model.predict(X_t_s)[0]
            oos_preds[name][t] = pred
            oos_errors[name][t] = y_t - pred

    # Build summary
    results = []
    for name in model_configs:
        errs = oos_errors[name]
        valid = ~np.isnan(errs)
        e = errs[valid]

        if len(e) == 0:
            continue

        mae = np.mean(np.abs(e))
        rmse = np.sqrt(np.mean(e ** 2))

        y_actual = df.loc[valid, "spread"].values
        ss_res = np.sum(e ** 2)
        ss_tot = np.sum((y_actual - y_actual.mean()) ** 2)
        r2_oos = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        results.append({
            "Model": name,
            "OOS R²": r2_oos,
            "OOS RMSE": rmse,
            "OOS MAE": mae,
            "OOS Obs": int(valid.sum()),
        })

    # Rolling OOS error DataFrame (for Diebold-Mariano or plotting)
    error_df = pd.DataFrame({"closeDate": df["closeDate"]})
    for name in model_configs:
        error_df[f"error_{name}"] = oos_errors[name]

    return pd.DataFrame(results), error_df