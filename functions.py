import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import adfuller, kpss
import statsmodels.api as sm

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


# =================== PCA ===================
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

def run_pca_training(curve_df, train_end="2017-12-31"):
    train_end = pd.to_datetime(train_end)
    cols = ["1y", "2y", "5y", "10y"]

    X_all = curve_df[cols].dropna()
    train_mask = curve_df.loc[X_all.index, "closeDate"] <= train_end
    X_train = X_all[train_mask]

    pca = PCA(n_components=3)
    pca.fit(X_train)
    pcs = pca.transform(X_all)

    pca_df = curve_df.loc[X_all.index, ["closeDate"]].copy()
    pca_df["Level"]     = pcs[:, 0]
    pca_df["Slope"]     = pcs[:, 1]
    pca_df["Curvature"] = pcs[:, 2]

    loadings = pd.DataFrame(pca.components_.T, index=cols,columns=["Level", "Slope", "Curvature"])

    explained = pd.DataFrame({"Component": ["Level", "Slope", "Curvature"],"Explained Variance": pca.explained_variance_ratio_})

    train_pcs = pca_df[pca_df["closeDate"] <= train_end]
    for pc in ["Level", "Slope", "Curvature"]:
        mean = train_pcs[pc].mean()
        std  = train_pcs[pc].std()
        pca_df[f"{pc}_z"] = (pca_df[pc] - mean) / std
    return pca_df, loadings, explained

# =================== Stationarity ===================

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


# =================== Residual (OLS Train/Test) ===================

def compute_residual_spread_oos(spread_df,pca_df,train_end="2017-12-31"):
    df = spread_df.merge(pca_df[["closeDate", "Level", "Slope", "Curvature"]],on="closeDate",how="inner").dropna().copy()
    train_end = pd.to_datetime(train_end)
    df["is_train"] = df["closeDate"] <= train_end
    train_df = df[df["is_train"]].copy()
    test_df = df[~df["is_train"]].copy()

    if train_df.empty:
        raise ValueError("Training sample is empty for residual OLS.")
    if test_df.empty:
        raise ValueError("Test sample is empty for residual OLS.")

    X_train = sm.add_constant(train_df[["Level", "Slope", "Curvature"]])
    y_train = train_df["spread"]

    model = sm.OLS(y_train, X_train).fit()

    X_full = sm.add_constant(df[["Level", "Slope", "Curvature"]], has_constant="add")
    df["fair_spread"] = model.predict(X_full)
    df["residual_spread"] = df["spread"] - df["fair_spread"]

    betas = pd.DataFrame({"Variable": model.params.index,"Beta": model.params.values})

    stats = {"r2_train": model.rsquared,"adj_r2_train": model.rsquared_adj}

    return df, betas, stats


# =================== K-means & GMM ===================

def run_regime_model(df, feature_cols, model_type="kmeans", n_regimes=3, random_state=42):
    out = df.copy()

    X = out[["closeDate"] + feature_cols].dropna().copy()
    idx = X.index
    train_end = pd.to_datetime('2017-12-31')
    train_mask = X["closeDate"] <= train_end
    X_train = X.loc[train_mask, feature_cols]
    X_full  = X[feature_cols]

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_full_scaled  = scaler.transform(X_full)

    if model_type == "kmeans":
        model = KMeans(n_clusters=n_regimes, n_init=20, random_state=random_state)
        model.fit(X_train_scaled)
        labels = model.predict(X_full_scaled)
        out.loc[idx, "regime"] = labels
        out.loc[idx, "is_train"] = train_mask.values
        centers = pd.DataFrame(
            scaler.inverse_transform(model.cluster_centers_),
            columns=feature_cols)
        centers["regime"] = range(n_regimes)

    elif model_type == "gmm":
        model = GaussianMixture(n_components=n_regimes, random_state=random_state)
        model.fit(X_train_scaled)
        labels = model.predict(X_full_scaled)
        probs  = model.predict_proba(X_full_scaled)
        out.loc[idx, "regime"] = labels
        out.loc[idx, "is_train"] = train_mask.values
        for i in range(n_regimes):
            out.loc[idx, f"prob_regime_{i}"] = probs[:, i]
        centers = pd.DataFrame(scaler.inverse_transform(model.means_),columns=feature_cols)
        centers["regime"] = range(n_regimes)

    else:
        raise ValueError("model_type must be 'kmeans' or 'gmm'")

    return out, centers, model


# =================== Trading ===================

def compute_trading_signal(residual_df, z_window, entry_long, entry_short, exit_band, stop_loss):
    df = residual_df.copy()

    roll_mean = df["residual_spread"].rolling(z_window).mean()
    roll_std  = df["residual_spread"].rolling(z_window).std()
    df["residual_z"] = (df["residual_spread"] - roll_mean) / roll_std

    position = pd.Series(0.0, index=df.index)
    prev = 0.0

    z = df["residual_z"].values

    for i in range(len(z)):
        if np.isnan(z[i]):
            position.iloc[i] = 0.0
            prev = 0.0
            continue

        if prev != 0 and abs(z[i]) >= stop_loss:
            position.iloc[i] = 0.0
            prev = 0.0
            continue

        if prev != 0 and abs(z[i]) <= exit_band:
            position.iloc[i] = 0.0
            prev = 0.0
            continue

        if prev == 0:
            if z[i] <= entry_long:
                position.iloc[i] = 1.0
                prev = 1.0
                continue
            elif z[i] >= entry_short:
                position.iloc[i] = -1.0
                prev = -1.0
                continue
        position.iloc[i] = prev

    df["position"] = position
    return df

# =========== Regime z-score ===========
def compute_regime_zscore(regime_df, train_end="2017-12-31"):
    df = regime_df.copy()
    train_end = pd.to_datetime(train_end)

    train = df[df["closeDate"] <= train_end]

    regime_stats = train.groupby("regime")["residual_spread"].agg(["mean", "std"]).reset_index()
    regime_stats.columns = ["regime", "regime_mean", "regime_std"]

    df = df.merge(regime_stats, on="regime", how="left")

    df["regime_z"] = (df["residual_spread"] - df["regime_mean"]) / df["regime_std"]

    return df, regime_stats

def compute_trading_signal_regime(df, entry_long, entry_short, exit_band, stop_loss=None, tradeable_regimes=None):
    out = df.copy()
    n = len(out)
    position  = np.zeros(n)
    exit_type = [None] * n
    prev = 0.0
    z       = out["regime_z"].values
    regimes = out["regime"].values
    for i in range(n):
        r = regimes[i]
        zi = z[i]
        regime_ok = True
        if tradeable_regimes is not None:
            regime_ok = (not np.isnan(r)) and (int(r) in tradeable_regimes)
        if not regime_ok:
            if prev != 0:
                exit_type[i] = "regime_exit"
            position[i] = 0.0
            prev = 0.0
            continue
        if np.isnan(zi):
            position[i] = prev
            continue
        if stop_loss is not None and prev != 0 and abs(zi) >= stop_loss:
            position[i] = 0.0
            exit_type[i] = "stop"
            prev = 0.0
            continue
        if prev != 0 and abs(zi) <= exit_band:
            position[i] = 0.0
            exit_type[i] = "signal"
            prev = 0.0
            continue
        if prev == 0:
            if zi <= entry_long:
                position[i] = 1.0
                prev = 1.0
                continue
            elif zi >= entry_short:
                position[i] = -1.0
                prev = -1.0
                continue
        position[i] = prev
    out["position"]  = position
    out["exit_type"] = exit_type
    return out


def run_backtest(signal_df, slippage_bps=0.0, roll_freq=90, roll_cost_bps=0.0, train_end="2017-12-31"):
    df = signal_df.copy()
    train_end = pd.to_datetime(train_end)

    df["spread_change"]   = df["spread"].diff()
    df["position_prev"]   = df["position"].shift(1).fillna(0.0)
    df["position_change"] = df["position"].diff().fillna(0.0)

    df["daily_pnl"] = df["position_prev"] * df["spread_change"]

    slippage_per_unit = slippage_bps / 10000.0
    df["slippage_cost"] = abs(df["position_change"]) * slippage_per_unit

    roll_per_unit = roll_cost_bps / 10000.0
    days_in_position = np.zeros(len(df))
    roll_cost = np.zeros(len(df))

    for i in range(1, len(df)):
        if df["position_prev"].iloc[i] != 0:
            days_in_position[i] = days_in_position[i - 1] + 1
            if roll_freq > 0 and days_in_position[i] % roll_freq == 0:
                roll_cost[i] = roll_per_unit
        else:
            days_in_position[i] = 0

    df["days_in_position"] = days_in_position
    df["roll_cost"] = roll_cost

    df["net_pnl"] = df["daily_pnl"] - df["slippage_cost"] - df["roll_cost"]
    df["cumulative_pnl"] = df["net_pnl"].cumsum()

    peak = df["cumulative_pnl"].cummax()
    df["drawdown"] = df["cumulative_pnl"] - peak
    df["is_train"] = df["closeDate"] <= train_end
    return df





def build_trade_log(signal_df):
    df = signal_df.copy().reset_index(drop=True)
    trades = []
    in_trade = False
    entry_idx = None

    z_col = "regime_z" if "regime_z" in df.columns else "residual_z"
    has_regime = "regime" in df.columns

    for i in range(len(df)):
        pos      = df["position"].iloc[i]
        prev_pos = df["position"].iloc[i - 1] if i > 0 else 0.0

        if prev_pos == 0 and pos != 0:
            in_trade = True
            entry_idx = i

        if in_trade and prev_pos != 0 and pos == 0:
            direction = "long" if df["position"].iloc[entry_idx] == 1.0 else "short"
            sign = 1.0 if direction == "long" else -1.0

            entry_spread = df["residual_spread"].iloc[entry_idx]
            exit_spread  = df["residual_spread"].iloc[i]
            pnl = sign * (exit_spread - entry_spread)

            trade = {
                "entry_date":    df["closeDate"].iloc[entry_idx],
                "exit_date":     df["closeDate"].iloc[i],
                "direction":     direction,
                "entry_z":       df[z_col].iloc[entry_idx],
                "exit_z":        df[z_col].iloc[i],
                "entry_spread":  entry_spread,
                "exit_spread":   exit_spread,
                "pnl":           pnl,
                "holding_days":  (df["closeDate"].iloc[i] - df["closeDate"].iloc[entry_idx]).days,
                "exit_type":     df["exit_type"].iloc[i] if "exit_type" in df.columns else None,
            }
            if has_regime:
                trade["entry_regime"] = int(df["regime"].iloc[entry_idx])

            trades.append(trade)
            in_trade = False
            entry_idx = None

    return pd.DataFrame(trades)


def compute_performance_metrics(backtest_df, trade_log_df, annual_factor=252):
    def _metrics(pnl_series, trades_sub):
        if len(pnl_series) == 0 or pnl_series.std() == 0:
            return {k: np.nan for k in ["total_pnl", "annual_return", "annual_vol","sharpe", "sortino", "max_drawdown","n_trades", "avg_holding_days", "pnl_per_trade"]}

        total_pnl     = pnl_series.sum()
        n_days        = len(pnl_series)
        annual_return = total_pnl * (annual_factor / n_days) if n_days > 0 else 0
        annual_vol    = pnl_series.std() * np.sqrt(annual_factor)
        sharpe        = annual_return / annual_vol if annual_vol > 0 else np.nan

        downside     = pnl_series[pnl_series < 0]
        downside_vol = downside.std() * np.sqrt(annual_factor) if len(downside) > 0 else np.nan
        sortino      = annual_return / downside_vol if downside_vol and downside_vol > 0 else np.nan

        cum    = pnl_series.cumsum()
        peak   = cum.cummax()
        max_dd = (cum - peak).min()

        n_trades = len(trades_sub)
        if n_trades > 0:
            avg_holding   = trades_sub["holding_days"].mean()
            pnl_per_trade = trades_sub["pnl"].mean()
        else:
            avg_holding = pnl_per_trade = np.nan

        return {
            "total_pnl":        total_pnl,
            "annual_return":    annual_return,
            "annual_vol":       annual_vol,
            "sharpe":           sharpe,
            "sortino":          sortino,
            "max_drawdown":     max_dd,
            "n_trades":         n_trades,
            "avg_holding_days": avg_holding,
            "pnl_per_trade":    pnl_per_trade,
        }

    train_end = pd.to_datetime("2017-12-31")

    train_pnl = backtest_df.loc[backtest_df["is_train"], "net_pnl"]
    test_pnl  = backtest_df.loc[~backtest_df["is_train"], "net_pnl"]

    if len(trade_log_df) > 0:
        train_trades = trade_log_df[trade_log_df["entry_date"] <= train_end]
        test_trades  = trade_log_df[trade_log_df["entry_date"] > train_end]
    else:
        train_trades = trade_log_df
        test_trades  = trade_log_df

    return {
        "all":   _metrics(backtest_df["net_pnl"], trade_log_df),
        "train": _metrics(train_pnl, train_trades),
        "test":  _metrics(test_pnl, test_trades),
    }