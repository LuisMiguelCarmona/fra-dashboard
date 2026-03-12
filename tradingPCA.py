import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from config import TRAIN_END


# =================== Residual (OLS, Ridge Train/Test) ===================

def compute_residual_spread(spread_df,pca_df,train_end=TRAIN_END,model_type="ols",alpha=1.0,):

    model_type = model_type.lower()

    if model_type not in ["ols", "ridge"]:
        raise ValueError("model_type must be one of: 'ols', 'ridge'")

    df = spread_df.merge(pca_df[["closeDate", "Level", "Slope", "Curvature"]],on="closeDate",how="inner",).dropna().copy()

    train_end = pd.to_datetime(train_end)
    df["is_train"] = df["closeDate"] <= train_end
    train_df = df[df["is_train"]].copy()

    if train_df.empty:
        raise ValueError(f"Training sample is empty for residual {model_type.upper()}.")

    feature_cols = ["Level", "Slope", "Curvature"]

    X_train = train_df[feature_cols].copy()
    y_train = train_df["spread"].copy()

    X_full = df[feature_cols].copy()

    if model_type == "ols":
        X_train_sm = sm.add_constant(X_train, has_constant="add")
        X_full_sm = sm.add_constant(X_full, has_constant="add")

        model = sm.OLS(y_train, X_train_sm).fit()

        df["fair_spread"] = model.predict(X_full_sm)
        df["residual_spread"] = df["spread"] - df["fair_spread"]

        betas = pd.DataFrame({"Variable": model.params.index,"Beta": model.params.values})

        stats = {"model_type": "OLS","alpha": None,"r2_train": model.rsquared,"adj_r2_train": model.rsquared_adj}

    elif model_type == "ridge":
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(X_train, y_train)

        df["fair_spread"] = model.predict(X_full)
        df["residual_spread"] = df["spread"] - df["fair_spread"]

        betas = pd.DataFrame({"Variable": ["const"] + feature_cols,"Beta": [model.intercept_] + list(model.coef_)})

        y_train_pred = model.predict(X_train)
        stats = {"model_type": "Ridge","alpha": alpha,"r2_train": r2_score(y_train, y_train_pred),"adj_r2_train": None}

    return df, betas, stats



# =================== Trading Signals ===================

def compute_trading_signal(residual_df, z_window, entry_long, entry_short, exit_band, stop_loss):
    df = residual_df.copy()

    roll_mean = df["residual_spread"].rolling(z_window).mean()
    roll_std = df["residual_spread"].rolling(z_window).std()
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


def compute_trading_signal_regime(df, entry_long, entry_short, exit_band, stop_loss=None, tradeable_regimes=None):
    out = df.copy()
    n = len(out)
    position = np.zeros(n)
    exit_type = [None] * n
    prev = 0.0
    z = out["regime_z"].values
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

    out["position"] = position
    out["exit_type"] = exit_type
    return out


# =================== Backtest ===================

def run_backtest(signal_df, slippage_bps=0.0, roll_freq=90, roll_cost_bps=0.0, train_end=TRAIN_END):
    df = signal_df.copy()
    train_end = pd.to_datetime(train_end)

    df["spread_change"] = df["spread"].diff()
    df["position_prev"] = df["position"].shift(1).fillna(0)
    df["position_change"] = df["position"].diff().fillna(0)

    df["daily_pnl"] = df["position_prev"] * df["spread_change"]

    slippage_per_unit = slippage_bps / 10000.0
    raw_slippage = abs(df["position_change"]) * slippage_per_unit

    # Fix: entry slippage (flat→active) should be charged on first ACTIVE day,
    # not on the transition day where position_prev is still 0.
    # This way it's attributed to the trade, not to "Flat".
    is_entry = (df["position_prev"] == 0) & (df["position"] != 0)
    entry_slippage = raw_slippage.where(is_entry, 0.0)
    non_entry_slippage = raw_slippage.where(~is_entry, 0.0)
    df["slippage_cost"] = non_entry_slippage + entry_slippage.shift(-1).fillna(0)

    roll_per_unit = roll_cost_bps / 10000.0

    days_in_position = np.zeros(len(df), dtype=int)
    roll_cost = np.zeros(len(df))

    for i in range(len(df)):
        if df["position_prev"].iloc[i] != 0:
            if i > 0 and df["position_prev"].iloc[i - 1] != 0:
                days_in_position[i] = days_in_position[i - 1] + 1
            else:
                days_in_position[i] = 1

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


# =================== Trade Log ===================

def build_trade_log(signal_df, backtest_df=None):
    """
    Build trade log. If backtest_df is provided, compute net P&L per trade
    by summing daily slippage and roll costs between entry and exit.
    """
    df = signal_df.copy().reset_index(drop=True)
    trades = []
    in_trade = False
    entry_idx = None

    z_col = "regime_z" if "regime_z" in df.columns else "residual_z"
    has_regime = "regime" in df.columns

    # Align backtest_df index if provided
    bt = None
    if backtest_df is not None:
        bt = backtest_df.copy().reset_index(drop=True)

    for i in range(len(df)):
        pos = df["position"].iloc[i]
        prev_pos = df["position"].iloc[i - 1] if i > 0 else 0.0

        if prev_pos == 0 and pos != 0:
            in_trade = True
            entry_idx = i

        if in_trade and prev_pos != 0 and pos == 0:
            direction = "long" if df["position"].iloc[entry_idx] == 1.0 else "short"
            sign = 1.0 if direction == "long" else -1.0

            pnl_col = "spread" if "spread" in df.columns else "residual_spread"
            entry_spread = df[pnl_col].iloc[entry_idx]
            exit_spread = df[pnl_col].iloc[i]
            gross_pnl = sign * (exit_spread - entry_spread)

            # Compute trade costs from backtest_df
            trade_slippage = 0.0
            trade_roll = 0.0
            net_pnl = gross_pnl
            if bt is not None:
                # Trade occupies days entry_idx to i (inclusive of exit day)
                trade_slice = bt.iloc[entry_idx:i + 1]
                trade_slippage = trade_slice["slippage_cost"].sum()
                trade_roll = trade_slice["roll_cost"].sum()
                net_pnl = gross_pnl - trade_slippage - trade_roll

            trade = {
                "entry_date": df["closeDate"].iloc[entry_idx],
                "exit_date": df["closeDate"].iloc[i],
                "direction": direction,
                "entry_z": df[z_col].iloc[entry_idx],
                "exit_z": df[z_col].iloc[i],
                "entry_spread": entry_spread,
                "exit_spread": exit_spread,
                "gross_pnl": gross_pnl,
                "slippage": trade_slippage,
                "roll_cost": trade_roll,
                "pnl": net_pnl,
                "holding_days": (df["closeDate"].iloc[i] - df["closeDate"].iloc[entry_idx]).days,
                "exit_type": df["exit_type"].iloc[i] if "exit_type" in df.columns else None,
            }
            if has_regime:
                trade["entry_regime"] = int(df["regime"].iloc[entry_idx])

            trades.append(trade)
            in_trade = False
            entry_idx = None

    return pd.DataFrame(trades)


# =================== Performance Metrics ===================

def compute_performance_metrics(backtest_df, trade_log_df, annual_factor=252):
    def _metrics(pnl_series, trades_sub):
        if len(pnl_series) == 0 or pnl_series.std() == 0:
            return {k: np.nan for k in [
                "total_pnl", "annual_return", "annual_vol",
                "sharpe", "sortino", "max_drawdown",
                "n_trades", "avg_holding_days", "pnl_per_trade",
            ]}

        total_pnl = pnl_series.sum()
        n_days = len(pnl_series)
        annual_return = total_pnl * (annual_factor / n_days) if n_days > 0 else 0
        annual_vol = pnl_series.std() * np.sqrt(annual_factor)
        sharpe = annual_return / annual_vol if annual_vol > 0 else np.nan

        downside = pnl_series[pnl_series < 0]
        downside_vol = downside.std() * np.sqrt(annual_factor) if len(downside) > 0 else np.nan
        sortino = annual_return / downside_vol if downside_vol and downside_vol > 0 else np.nan

        cum = pnl_series.cumsum()
        peak = cum.cummax()
        max_dd = (cum - peak).min()

        n_trades = len(trades_sub)
        if n_trades > 0:
            avg_holding = trades_sub["holding_days"].mean()
            pnl_per_trade = trades_sub["pnl"].mean()
        else:
            avg_holding = pnl_per_trade = np.nan

        return {
            "total_pnl": total_pnl,
            "annual_return": annual_return,
            "annual_vol": annual_vol,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_dd,
            "n_trades": n_trades,
            "avg_holding_days": avg_holding,
            "pnl_per_trade": pnl_per_trade,
        }

    train_end = pd.to_datetime(TRAIN_END)

    train_pnl = backtest_df.loc[backtest_df["is_train"], "net_pnl"]
    test_pnl = backtest_df.loc[~backtest_df["is_train"], "net_pnl"]

    if len(trade_log_df) > 0:
        train_trades = trade_log_df[trade_log_df["entry_date"] <= train_end]
        test_trades = trade_log_df[trade_log_df["entry_date"] > train_end]
    else:
        train_trades = trade_log_df
        test_trades = trade_log_df

    return {
        "all": _metrics(backtest_df["net_pnl"], trade_log_df),
        "train": _metrics(train_pnl, train_trades),
        "test": _metrics(test_pnl, test_trades),
    }