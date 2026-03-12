import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


# =================== Walk-Forward Expanding Window ===================

def walk_forward_backtest(spread_df, features_df, feature_cols,initial_train=504, step=63, alpha=1.0,model_type="ridge"):
    df = spread_df.merge(features_df[["closeDate"] + feature_cols],on="closeDate", how="inner").dropna().reset_index(drop=True)

    n = len(df)
    fair_values = np.full(n, np.nan)
    oos_r2_windows = []
    fold_metrics = []

    t = initial_train
    fold_id = 0

    while t < n:
        end_oos = min(t + step, n)

        X_train = df.iloc[:t][feature_cols].values
        y_train = df.iloc[:t]["spread"].values
        X_oos = df.iloc[t:end_oos][feature_cols].values
        y_oos = df.iloc[t:end_oos]["spread"].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_oos_s = scaler.transform(X_oos)

        if model_type == "ridge":
            model = Ridge(alpha=alpha, fit_intercept=True)
        elif model_type == "lasso":
            model = Lasso(alpha=alpha, fit_intercept=True, max_iter=5000)
        elif model_type == "elasticnet":
            model = ElasticNet(alpha=alpha, l1_ratio=0.5, fit_intercept=True, max_iter=5000)
        else:  # ols
            model = Ridge(alpha=1e-10, fit_intercept=True)

        model.fit(X_train_s, y_train)

        y_pred_oos = model.predict(X_oos_s)
        fair_values[t:end_oos] = y_pred_oos

        y_pred_train = model.predict(X_train_s)
        r2_train = r2_score(y_train, y_pred_train)

        ss_res = ((y_oos - y_pred_oos) ** 2).sum()
        ss_tot = ((y_oos - y_oos.mean()) ** 2).sum()
        r2_oos = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        fold_metrics.append({
            "fold": fold_id,
            "train_start": df.loc[0, "closeDate"],
            "train_end": df.loc[t - 1, "closeDate"],
            "oos_start": df.loc[t, "closeDate"],
            "oos_end": df.loc[end_oos - 1, "closeDate"],
            "train_size": t,
            "oos_size": end_oos - t,
            "r2_train": r2_train,
            "r2_oos": r2_oos,
        })

        t = end_oos
        fold_id += 1

    df["wf_fair_value"] = fair_values
    df["wf_residual"] = df["spread"] - df["wf_fair_value"]

    fold_df = pd.DataFrame(fold_metrics)
    return df, fold_df


# =================== Monte Carlo Signal Validation ===================

def run_monte_carlo(signal_df, backtest_func, n_sims=500, seed=42):
    """
    Validates signal by randomizing direction while preserving
    position frequency and execution infrastructure.

    Parameters
    ----------
    signal_df : DataFrame with 'position' column
    backtest_func : callable that takes signal_df and returns backtest_df with 'net_pnl'
    n_sims : number of simulations

    Returns
    -------
    dict with real_sharpe, mc_sharpes, percentile, z_score, p_value
    """
    rng = np.random.RandomState(seed)

    real_bt = backtest_func(signal_df)
    real_pnl = real_bt["net_pnl"]
    real_sharpe = _compute_sharpe(real_pnl)

    # Empirical position distribution
    pos_vals = signal_df["position"].values
    unique, counts = np.unique(pos_vals[~np.isnan(pos_vals)], return_counts=True)
    probs = counts / counts.sum()

    mc_sharpes = []
    for _ in range(n_sims):
        sim_df = signal_df.copy()
        random_pos = rng.choice(unique, size=len(sim_df), p=probs)
        sim_df["position"] = random_pos

        sim_bt = backtest_func(sim_df)
        sim_sharpe = _compute_sharpe(sim_bt["net_pnl"])
        mc_sharpes.append(sim_sharpe)

    mc_sharpes = np.array(mc_sharpes)
    mc_mean = np.nanmean(mc_sharpes)
    mc_std = np.nanstd(mc_sharpes)

    percentile = (mc_sharpes < real_sharpe).sum() / len(mc_sharpes) * 100
    z_score = (real_sharpe - mc_mean) / mc_std if mc_std > 0 else np.nan

    return {
        "real_sharpe": real_sharpe,
        "mc_mean": mc_mean,
        "mc_std": mc_std,
        "mc_5th": np.nanpercentile(mc_sharpes, 5),
        "mc_95th": np.nanpercentile(mc_sharpes, 95),
        "percentile": percentile,
        "z_score": z_score,
        "mc_sharpes": mc_sharpes,
    }


def _compute_sharpe(pnl_series, annual_factor=252):
    pnl = pnl_series.dropna()
    if len(pnl) == 0 or pnl.std() == 0:
        return np.nan
    return (pnl.mean() / pnl.std()) * np.sqrt(annual_factor)


# =================== Structural Break Tests ===================

def run_zivot_andrews(series, max_lags=None, trim=0.15):
    """
    Zivot-Andrews structural break test (model='c' — level break).
    Returns dict with statistic, break_date, critical values.
    """
    s = pd.Series(series).dropna()
    try:
        from statsmodels.tsa.stattools import zivot_andrews
        result = zivot_andrews(s, maxlag=max_lags, regression="c", autolag="AIC")
        return {"statistic": result[0],"p_value": result[1],"break_index": result[4],"lags": result[2],"critical_values": result[3]}
    except Exception as e:
        return {"statistic": np.nan, "p_value": np.nan, "break_index": None,"error": str(e)}


def run_chow_test(y, X, break_date, dates):
    """
    Chow test for structural break at a given date.
    Returns F-statistic and p-value.
    """
    from scipy import stats as scipy_stats

    mask = dates <= pd.to_datetime(break_date)
    y1, X1 = y[mask], X[mask]
    y2, X2 = y[~mask], X[~mask]

    n1, n2 = len(y1), len(y2)
    k = X.shape[1] + 1  # including constant

    if n1 < k + 5 or n2 < k + 5:
        return {"f_stat": np.nan, "p_value": np.nan, "error": "Insufficient obs"}

    X_full = sm.add_constant(X)
    X1_c = sm.add_constant(X1)
    X2_c = sm.add_constant(X2)

    ssr_full = sm.OLS(y, X_full).fit().ssr
    ssr1 = sm.OLS(y1, X1_c).fit().ssr
    ssr2 = sm.OLS(y2, X2_c).fit().ssr
    ssr_sub = ssr1 + ssr2

    f_stat = ((ssr_full - ssr_sub) / k) / (ssr_sub / (n1 + n2 - 2 * k))
    p_value = 1 - scipy_stats.f.cdf(f_stat, k, n1 + n2 - 2 * k)

    return {"f_stat": f_stat, "p_value": p_value}


# =================== Stress Test Periods ===================

STRESS_PERIODS = {
    "2008 GFC": ("2008-09-01", "2009-03-31"),
    "2013 Taper Tantrum": ("2013-05-01", "2013-09-30"),
    "2015-16 Brazil Crisis": ("2015-06-01", "2016-06-30"),
    "2020 COVID": ("2020-02-15", "2020-06-30"),
    "2021-22 Tightening": ("2021-03-01", "2022-06-30"),
    "2024-25 Fiscal Stress": ("2024-06-01", "2025-06-30"),
}

def compute_stress_test(backtest_df, trade_log_df=None, periods=None):
    """
    Computes performance metrics for each stress period.
    """
    if periods is None:
        periods = STRESS_PERIODS

    results = []
    for name, (start, end) in periods.items():
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)

        mask = (backtest_df["closeDate"] >= start_dt) & (backtest_df["closeDate"] <= end_dt)
        sub = backtest_df[mask]

        if len(sub) == 0:
            continue

        pnl = sub["net_pnl"]
        cum = pnl.cumsum()
        peak = cum.cummax()
        max_dd = (cum - peak).min()

        ann_ret = pnl.mean() * 252 if len(pnl) > 0 else 0
        ann_vol = pnl.std() * np.sqrt(252) if len(pnl) > 1 else np.nan
        sharpe = ann_ret / ann_vol if ann_vol and ann_vol > 0 else np.nan

        n_active = (sub["position_prev"].abs() > 0).sum() if "position_prev" in sub.columns else np.nan
        pct_active = n_active / len(sub) * 100 if len(sub) > 0 else 0

        # Hit rate on active days
        if "position_prev" in sub.columns:
            active = sub[sub["position_prev"].abs() > 0]
            if len(active) > 0:
                hit = (active["net_pnl"] > 0).mean()
            else:
                hit = np.nan
        else:
            hit = np.nan

        results.append({
            "Period": name,
            "Start": start,
            "End": end,
            "Trading Days": len(sub),
            "Total P&L (bps)": pnl.sum() * 10000,
            "Ann. Return (bps)": ann_ret * 10000,
            "Ann. Vol (bps)": ann_vol * 10000 if not np.isnan(ann_vol) else np.nan,
            "Sharpe": sharpe,
            "Max DD (bps)": max_dd * 10000,
            "% Active": pct_active,
            "Hit Rate": hit,
        })

    return pd.DataFrame(results)


# =================== Position-Level Attribution ===================

def compute_position_attribution(backtest_df):
    """
    Decomposes P&L by position direction: Long, Short, Flat.
    Similar to Mauro's position-level analysis.
    """
    df = backtest_df.copy()

    categories = {
        "Long": df["position_prev"] > 0,
        "Short": df["position_prev"] < 0,
        "Flat": df["position_prev"] == 0,
    }

    rows = []
    for cat, mask in categories.items():
        sub = df[mask]
        if len(sub) == 0:
            continue

        gross = sub["daily_pnl"].sum()
        slip = sub["slippage_cost"].sum()
        roll = sub["roll_cost"].sum()
        net = sub["net_pnl"].sum()

        hit = (sub["net_pnl"] > 0).mean() if len(sub) > 0 else np.nan

        rows.append({
            "Direction": cat,
            "Trading Days": len(sub),
            "% of Sample": len(sub) / len(df) * 100,
            "Gross P&L (bps)": gross * 10000,
            "Slippage (bps)": slip * 10000,
            "Roll Cost (bps)": roll * 10000,
            "Net P&L (bps)": net * 10000,
            "Hit Rate": hit,
        })

    return pd.DataFrame(rows)


# =================== Rolling Parameter Stability ===================

def compute_rolling_ols_stability(spread_df, features_df, feature_cols,
                                   window=504, step=21):
    """
    Runs rolling OLS and returns time-varying betas + t-stats.
    Shows parameter stability over time.
    """
    df = spread_df.merge(
        features_df[["closeDate"] + feature_cols],
        on="closeDate", how="inner"
    ).dropna().reset_index(drop=True)

    n = len(df)
    results = []

    for t in range(window, n, step):
        start = t - window
        sub = df.iloc[start:t]

        X = sm.add_constant(sub[feature_cols])
        y = sub["spread"]

        try:
            model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
            row = {"closeDate": df.loc[t - 1, "closeDate"], "r2": model.rsquared}
            for col in feature_cols:
                row[f"beta_{col}"] = model.params.get(col, np.nan)
                row[f"tstat_{col}"] = model.tvalues.get(col, np.nan)
            results.append(row)
        except Exception:
            continue

    return pd.DataFrame(results)


# =================== Cross-Validation (Time Series K-Fold) ===================

def time_series_cv(spread_df, features_df, feature_cols,
                   n_splits=5, alpha=1.0):
    """
    Time-series cross-validation with expanding window.
    Returns fold-level R² metrics.
    """
    df = spread_df.merge(
        features_df[["closeDate"] + feature_cols],
        on="closeDate", how="inner"
    ).dropna().reset_index(drop=True)

    n = len(df)
    fold_size = n // (n_splits + 1)

    results = []
    for k in range(n_splits):
        train_end = fold_size * (k + 1)
        test_start = train_end
        test_end = min(train_end + fold_size, n)

        if test_end <= test_start:
            continue

        X_train = df.iloc[:train_end][feature_cols].values
        y_train = df.iloc[:train_end]["spread"].values
        X_test = df.iloc[test_start:test_end][feature_cols].values
        y_test = df.iloc[test_start:test_end]["spread"].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(X_train_s, y_train)

        r2_train = r2_score(y_train, model.predict(X_train_s))
        r2_test = r2_score(y_test, model.predict(X_test_s))

        results.append({
            "fold": k + 1,
            "train_end": df.loc[train_end - 1, "closeDate"],
            "test_start": df.loc[test_start, "closeDate"],
            "test_end": df.loc[test_end - 1, "closeDate"],
            "train_size": train_end,
            "test_size": test_end - test_start,
            "r2_train": r2_train,
            "r2_test": r2_test,
        })

    return pd.DataFrame(results)