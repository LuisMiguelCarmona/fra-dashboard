import pandas as pd
import numpy as np
from config import TRAIN_END
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy import stats as scipy_stats


# =================== Walk-Forward Expanding Window ===================

def walk_forward_backtest(spread_df, features_df, feature_cols,
                          initial_train=504, step=63, alpha=1.0,
                          model_type="ridge"):
    df = spread_df.merge(
        features_df[["closeDate"] + feature_cols],
        on="closeDate", how="inner"
    ).dropna().reset_index(drop=True)

    n = len(df)
    fair_values = np.full(n, np.nan)
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
        else:
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


# =================== Feature Selection: Stability Selection ===================

def stability_selection(spread_df, features_df, feature_cols,
                        train_end=TRAIN_END, n_subsamples=100,
                        subsample_frac=0.7, alpha_range=None, threshold=0.6,
                        seed=42):
    """
    Stability Selection (Meinshausen & Buhlmann, 2010).

    Runs LASSO on random subsamples of the training data across a grid of
    regularization strengths. Features selected in >= threshold fraction
    of all (subsample x alpha) runs are considered stable.

    Controls expected false positives: E[V] <= q^2 / (2*p*threshold - q).
    """
    rng = np.random.RandomState(seed)
    train_end_dt = pd.to_datetime(train_end)

    df = spread_df.merge(features_df[["closeDate"] + feature_cols],on="closeDate", how="inner").dropna()
    train = df[df["closeDate"] <= train_end_dt]
    X = train[feature_cols].values
    y = train["spread"].values
    n, p = X.shape

    if alpha_range is None:
        X_scaled = StandardScaler().fit_transform(X)
        lambda_max = np.max(np.abs(X_scaled.T @ (y - y.mean()))) / n
        alpha_range = np.logspace(np.log10(lambda_max), np.log10(lambda_max * 0.01), 20)

    selection_counts = np.zeros(p)
    total_runs = 0
    subsample_size = int(n * subsample_frac)

    for _ in range(n_subsamples):
        idx = rng.choice(n, size=subsample_size, replace=False)
        X_sub = X[idx]
        y_sub = y[idx]

        scaler = StandardScaler()
        X_sub_s = scaler.fit_transform(X_sub)

        for alpha in alpha_range:
            model = Lasso(alpha=alpha, fit_intercept=True, max_iter=5000)
            model.fit(X_sub_s, y_sub)
            selected = np.abs(model.coef_) > 1e-10
            selection_counts += selected.astype(float)
            total_runs += 1

    selection_prob = selection_counts / total_runs

    results = pd.DataFrame({
        "Feature": feature_cols,
        "Selection Probability": selection_prob,
        "Selected": selection_prob >= threshold,
    }).sort_values("Selection Probability", ascending=False).reset_index(drop=True)

    return results


# =================== Structural Break: Zivot-Andrews ===================

def run_zivot_andrews(series, max_lags=None, trim=0.15):
    s = pd.Series(series).dropna()
    try:
        from statsmodels.tsa.stattools import zivot_andrews
        result = zivot_andrews(s, maxlag=max_lags, regression="c", autolag="AIC")
        return {
            "statistic": result[0], "p_value": result[1],
            "break_index": result[4], "lags": result[2],
            "critical_values": result[3],
        }
    except Exception as e:
        return {"statistic": np.nan, "p_value": np.nan, "break_index": None, "error": str(e)}


# =================== Structural Break: Chow Test ===================

def run_chow_test(y, X, break_date, dates):
    """
    Chow (1960) test for structural break at a known date.

    Compares SSR of a single pooled regression vs two sub-period regressions.
    Significant F-stat means the relationship is structurally different
    before and after the break date.

    Parameters
    ----------
    y : array-like — dependent variable
    X : array-like — regressors (no constant, added internally)
    break_date : str or Timestamp — the candidate break date
    dates : array-like — date index aligned with y and X
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    dates = pd.to_datetime(dates)

    mask = dates <= pd.to_datetime(break_date)
    y1, X1 = y[mask], X[mask]
    y2, X2 = y[~mask], X[~mask]

    n1, n2 = len(y1), len(y2)
    k = X.shape[1] + 1  # including constant

    if n1 < k + 5 or n2 < k + 5:
        return {"f_stat": np.nan, "p_value": np.nan, "error": "Insufficient obs in one sub-period"}

    X_full = sm.add_constant(X)
    X1_c = sm.add_constant(X1)
    X2_c = sm.add_constant(X2)

    ssr_full = sm.OLS(y, X_full).fit().ssr
    ssr1 = sm.OLS(y1, X1_c).fit().ssr
    ssr2 = sm.OLS(y2, X2_c).fit().ssr
    ssr_sub = ssr1 + ssr2

    f_stat = ((ssr_full - ssr_sub) / k) / (ssr_sub / (n1 + n2 - 2 * k))
    p_value = 1 - scipy_stats.f.cdf(f_stat, k, n1 + n2 - 2 * k)

    return {"f_stat": f_stat, "p_value": p_value, "n_pre": n1, "n_post": n2}


# =================== Structural Break: Bai-Perron ===================

def run_bai_perron(series, max_breaks=5, min_segment=60, pen_factor=3.0):
    s = np.asarray(pd.Series(series).dropna(), dtype=float)
    n = len(s)

    if n < 2 * min_segment:
        return {"n_breaks": 0, "break_indices": [], "error": "Series too short"}

    # cumulative sums for O(1) segment SSR
    csum = np.zeros(n + 1)
    csum2 = np.zeros(n + 1)
    csum[1:] = np.cumsum(s)
    csum2[1:] = np.cumsum(s ** 2)

    def segment_ssr(start, end):
        # inclusive [start, end]
        if end < start:
            return 0.0
        m = end - start + 1
        sx = csum[end + 1] - csum[start]
        sx2 = csum2[end + 1] - csum2[start]
        return max(0.0, sx2 - (sx * sx) / m)

    bic_results = []

    # m = 0
    ssr_0 = segment_ssr(0, n - 1)
    k_0 = 2
    bic_0 = n * np.log(ssr_0 / n + 1e-15) + pen_factor * k_0 * np.log(n)
    bic_results.append({"n_breaks": 0, "ssr": ssr_0, "bic": bic_0, "breaks": []})

    max_m = min(max_breaks, n // min_segment - 1)

    for m in range(1, max_m + 1):
        best_ssr = np.inf
        best_breaks = []

        if m == 1:
            for bp in range(min_segment, n - min_segment + 1):
                ssr = segment_ssr(0, bp - 1) + segment_ssr(bp, n - 1)
                if ssr < best_ssr:
                    best_ssr = ssr
                    best_breaks = [bp]

        elif m == 2:
            for bp1 in range(min_segment, n - 2 * min_segment + 1):
                ssr1 = segment_ssr(0, bp1 - 1)
                for bp2 in range(bp1 + min_segment, n - min_segment + 1):
                    ssr = ssr1 + segment_ssr(bp1, bp2 - 1) + segment_ssr(bp2, n - 1)
                    if ssr < best_ssr:
                        best_ssr = ssr
                        best_breaks = [bp1, bp2]

        else:
            prev = bic_results[-1]["breaks"]
            if len(prev) < m - 1:
                continue

            boundaries = [0] + prev + [n]
            seg_lens = [boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1)]
            longest = int(np.argmax(seg_lens))
            seg_s, seg_e = boundaries[longest], boundaries[longest + 1]

            best_insert_ssr = np.inf
            best_new_bp = None

            for bp in range(seg_s + min_segment, seg_e - min_segment + 1):
                ssr_lr = segment_ssr(seg_s, bp - 1) + segment_ssr(bp, seg_e - 1)
                if ssr_lr < best_insert_ssr:
                    best_insert_ssr = ssr_lr
                    best_new_bp = bp

            if best_new_bp is None:
                continue

            new_breaks = sorted(prev + [best_new_bp])
            bounds = [0] + new_breaks + [n]
            best_ssr = sum(
                segment_ssr(bounds[i], bounds[i + 1] - 1)
                for i in range(len(bounds) - 1)
            )
            best_breaks = new_breaks

        k_m = 2 * (m + 1)
        bic_m = n * np.log(best_ssr / n + 1e-15) + pen_factor * k_m * np.log(n)
        bic_results.append({"n_breaks": m, "ssr": best_ssr, "bic": bic_m, "breaks": best_breaks})

    bic_df = pd.DataFrame(bic_results)
    best_idx = bic_df["bic"].idxmin()
    best = bic_results[best_idx]

    boundaries = [0] + best["breaks"] + [n]
    segments = []
    for i in range(len(boundaries) - 1):
        a, b = boundaries[i], boundaries[i + 1] - 1
        seg = s[a:b + 1]
        segments.append({
            "Segment": i + 1,
            "Start Idx": a,
            "End Idx": b,
            "Obs": len(seg),
            "Mean": seg.mean(),
            "Std": seg.std(ddof=0),
        })

    return {
        "n_breaks": best["n_breaks"],
        "break_indices": best["breaks"],
        "segments": pd.DataFrame(segments),
        "bic_table": bic_df[["n_breaks", "ssr", "bic"]],
    }

# =================== Residual Diagnostics: Ljung-Box ===================

def run_ljung_box(residuals, lags=None):
    """
    Ljung-Box test for residual autocorrelation.
    H0: no autocorrelation up to lag k.
    Rejection means the model is incomplete.
    """
    resid = pd.Series(residuals).dropna()
    if lags is None:
        lags = [5, 10, 20]

    result = acorr_ljungbox(resid, lags=lags, return_df=True)
    result = result.reset_index()
    result.columns = ["Lag", "LB Statistic", "p-value"]
    result["Significant (5%)"] = result["p-value"].apply(lambda p: "Y" if p < 0.05 else "")
    return result


# =================== Residual Diagnostics: CUSUM / CUSUMSQ ===================

def compute_cusum(y, X, dates=None):
    """
    CUSUM test (Brown, Durbin & Evans, 1975) for parameter stability.
    Tracks cumulative sum of recursive residuals.
    Breach of 5% bands => structural change in parameters.
    """
    y = np.asarray(y, dtype=float)
    X_c = sm.add_constant(np.asarray(X, dtype=float))
    n, k = X_c.shape
    start = k + 5

    recursive_residuals = []
    for t in range(start, n):
        model = sm.OLS(y[:t], X_c[:t]).fit()
        y_hat = model.predict(X_c[[t]])[0]
        resid = y[t] - y_hat

        x_t = X_c[t:t + 1]
        try:
            h_t = (x_t @ np.linalg.inv(X_c[:t].T @ X_c[:t]) @ x_t.T)[0, 0]
        except np.linalg.LinAlgError:
            h_t = 0

        sigma = np.sqrt(model.mse_resid * (1 + h_t)) if model.mse_resid > 0 else 1e-10
        recursive_residuals.append(resid / sigma)

    w = np.array(recursive_residuals)
    T = len(w)
    cusum = np.cumsum(w)
    cusum_norm = cusum / np.sqrt(T)

    # 5% significance bands
    a = 0.948
    t_frac = np.arange(1, T + 1) / T
    upper_band = a + 2 * a * t_frac
    lower_band = -(a + 2 * a * t_frac)

    breach = bool(np.any(cusum_norm > upper_band) or np.any(cusum_norm < lower_band))
    date_index = dates[start:n] if dates is not None else np.arange(start, n)

    return {
        "cusum": cusum_norm, "upper_band": upper_band, "lower_band": lower_band,
        "dates": date_index, "breach": breach, "start_idx": start,
    }


def compute_cusumsq(y, X, dates=None):
    """
    CUSUMSQ test for variance stability.
    Tracks cumulative sum of squared recursive residuals.
    Deviation from expected linear path => heteroskedasticity / variance break.
    """
    y = np.asarray(y, dtype=float)
    X_c = sm.add_constant(np.asarray(X, dtype=float))
    n, k = X_c.shape
    start = k + 5

    sq_resids = []
    for t in range(start, n):
        model = sm.OLS(y[:t], X_c[:t]).fit()
        y_hat = model.predict(X_c[[t]])[0]
        sq_resids.append((y[t] - y_hat) ** 2)

    w_sq = np.array(sq_resids)
    T = len(w_sq)
    total = w_sq.sum()

    if total == 0:
        return {"cusumsq": np.zeros(T), "expected": np.zeros(T),
                "upper_band": np.ones(T), "lower_band": -np.ones(T),
                "dates": np.arange(start, n), "breach": False, "start_idx": start}

    cusumsq = np.cumsum(w_sq) / total
    expected = np.arange(1, T + 1) / T

    c_alpha = 1.358 / np.sqrt(T)
    upper_band = expected + c_alpha
    lower_band = expected - c_alpha

    breach = bool(np.any(cusumsq > upper_band) or np.any(cusumsq < lower_band))
    date_index = dates[start:n] if dates is not None else np.arange(start, n)

    return {
        "cusumsq": cusumsq, "expected": expected,
        "upper_band": upper_band, "lower_band": lower_band,
        "dates": date_index, "breach": breach, "start_idx": start,
    }


# =================== Model Comparison: Diebold-Mariano ===================

def diebold_mariano_test(errors_1, errors_2, horizon=1, loss="mse", harvey_correction=True):
    """
    Diebold-Mariano (1995) test for equal predictive accuracy.
    H0: Model 1 and Model 2 have equal forecast accuracy.
    Harvey, Leybourne & Newbold (1997) small-sample correction applied.
    """
    e1 = np.asarray(errors_1)
    e2 = np.asarray(errors_2)
    n = min(len(e1), len(e2))
    e1, e2 = e1[:n], e2[:n]

    if loss == "mse":
        d = e1 ** 2 - e2 ** 2
    elif loss == "mae":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError("loss must be 'mse' or 'mae'")

    d_mean = d.mean()

    # HAC variance (Newey-West with bandwidth = horizon - 1)
    gamma_0 = np.mean((d - d_mean) ** 2)
    gamma_sum = 0
    for k in range(1, horizon):
        gamma_k = np.mean((d[k:] - d_mean) * (d[:-k] - d_mean))
        gamma_sum += 2 * gamma_k

    var_d = (gamma_0 + gamma_sum) / n

    if var_d <= 0:
        return {"dm_stat": np.nan, "p_value": np.nan, "error": "Non-positive variance"}

    dm_stat = d_mean / np.sqrt(var_d)

    if harvey_correction:
        correction = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
        dm_stat = dm_stat * correction

    p_value = 2 * (1 - scipy_stats.norm.cdf(abs(dm_stat)))
    better = "Model 1" if d_mean < 0 else "Model 2" if d_mean > 0 else "Equal"

    return {"dm_stat": dm_stat, "p_value": p_value, "mean_loss_diff": d_mean, "better_model": better}


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
    if periods is None:
        periods = STRESS_PERIODS
    results = []
    for name, (start, end) in periods.items():
        start_dt, end_dt = pd.to_datetime(start), pd.to_datetime(end)
        mask = (backtest_df["closeDate"] >= start_dt) & (backtest_df["closeDate"] <= end_dt)
        sub = backtest_df[mask]
        if len(sub) == 0:
            continue
        pnl = sub["net_pnl"]
        cum = pnl.cumsum()
        max_dd = (cum - cum.cummax()).min()
        ann_ret = pnl.mean() * 252
        ann_vol = pnl.std() * np.sqrt(252) if len(pnl) > 1 else np.nan
        sharpe = ann_ret / ann_vol if ann_vol and ann_vol > 0 else np.nan
        n_active = (sub["position_prev"].abs() > 0).sum() if "position_prev" in sub.columns else np.nan
        pct_active = n_active / len(sub) * 100 if len(sub) > 0 else 0
        if "position_prev" in sub.columns:
            active = sub[sub["position_prev"].abs() > 0]
            hit = (active["net_pnl"] > 0).mean() if len(active) > 0 else np.nan
        else:
            hit = np.nan
        results.append({
            "Period": name, "Trading Days": len(sub),
            "Total P&L (bps)": pnl.sum() * 10000, "Ann. Return (bps)": ann_ret * 10000,
            "Ann. Vol (bps)": ann_vol * 10000 if not np.isnan(ann_vol) else np.nan,
            "Sharpe": sharpe, "Max DD (bps)": max_dd * 10000,
            "% Active": pct_active, "Hit Rate": hit,
        })
    return pd.DataFrame(results)


# =================== Position-Level Attribution ===================

def compute_position_attribution(backtest_df):
    df = backtest_df.copy()
    categories = {"Long": df["position_prev"] > 0, "Short": df["position_prev"] < 0, "Flat": df["position_prev"] == 0}
    rows = []
    for cat, mask in categories.items():
        sub = df[mask]
        if len(sub) == 0:
            continue
        rows.append({
            "Direction": cat, "Trading Days": len(sub),
            "% of Sample": len(sub) / len(df) * 100,
            "Gross P&L (bps)": sub["daily_pnl"].sum() * 10000,
            "Slippage (bps)": sub["slippage_cost"].sum() * 10000,
            "Roll Cost (bps)": sub["roll_cost"].sum() * 10000,
            "Net P&L (bps)": sub["net_pnl"].sum() * 10000,
            "Hit Rate": (sub["net_pnl"] > 0).mean() if len(sub) > 0 else np.nan,
        })
    return pd.DataFrame(rows)


# =================== Rolling Parameter Stability ===================

def compute_rolling_ols_stability(spread_df, features_df, feature_cols, window=504, step=21):
    df = spread_df.merge(features_df[["closeDate"] + feature_cols], on="closeDate", how="inner").dropna().reset_index(drop=True)
    n = len(df)
    results = []
    for t in range(window, n, step):
        sub = df.iloc[t - window:t]
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



# =================== Block Bootstrap for Sharpe Inference ===================

def _optimal_block_length(series, max_lag=None):
    s = np.asarray(series)
    n = len(s)
    if max_lag is None:
        max_lag = min(int(10 * np.log10(n)), n // 3)
    threshold = 1.96 / np.sqrt(n)
    s_demeaned = s - s.mean()
    var = np.dot(s_demeaned, s_demeaned) / n
    if var == 0:
        return max(1, int(np.sqrt(n)))
    cutoff_lag = 1
    for lag in range(1, max_lag + 1):
        acf_val = np.dot(s_demeaned[lag:], s_demeaned[:-lag]) / (n * var)
        if abs(acf_val) < threshold:
            cutoff_lag = lag
            break
    else:
        cutoff_lag = max_lag
    return max(5, min(int(2 * cutoff_lag), n // 4))


def block_bootstrap_sharpe(pnl_series, n_bootstrap=5000, block_length=None,
                            confidence_level=0.95, annual_factor=252, seed=42):
    rng = np.random.RandomState(seed)
    pnl = np.asarray(pnl_series).copy()
    pnl = pnl[~np.isnan(pnl)]
    n = len(pnl)
    if n < 30:
        return {"sharpe_point": np.nan, "ci_lower": np.nan, "ci_upper": np.nan,
                "p_value": np.nan, "bootstrap_sharpes": np.array([]),
                "block_length": 0, "se": np.nan, "error": "Series too short (<30 obs)"}
    sharpe_point = (pnl.mean() / pnl.std()) * np.sqrt(annual_factor) if pnl.std() > 0 else np.nan
    if block_length is None:
        block_length = _optimal_block_length(pnl)
    n_blocks = int(np.ceil(n / block_length))
    max_start = n - block_length
    if max_start < 1:
        block_length = max(1, n // 3)
        max_start = n - block_length
    bootstrap_sharpes = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        starts = rng.randint(0, max_start + 1, size=n_blocks)
        boot_sample = np.concatenate([pnl[s:s + block_length] for s in starts])[:n]
        std_b = boot_sample.std()
        bootstrap_sharpes[b] = (boot_sample.mean() / std_b) * np.sqrt(annual_factor) if std_b > 0 else 0.0
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_sharpes, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_sharpes, 100 * (1 - alpha / 2))
    p_value = np.mean(bootstrap_sharpes <= 0)
    se = np.std(bootstrap_sharpes)
    return {
        "sharpe_point": sharpe_point, "ci_lower": ci_lower, "ci_upper": ci_upper,
        "p_value": p_value, "bootstrap_sharpes": bootstrap_sharpes,
        "block_length": block_length, "se": se,
        "n_bootstrap": n_bootstrap, "confidence_level": confidence_level,
    }