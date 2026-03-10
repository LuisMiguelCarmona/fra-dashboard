
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from clustering import compute_regime_zscore
from tradingPCA import compute_trading_signal_regime, run_backtest, build_trade_log, compute_performance_metrics

TRAINING_DATE = pd.to_datetime("2017-12-31")

REGIME_LABELS = {0: "Tightening", 1: "Easing", 2: "Transitional"}
REGIME_COLORS_BG = {
    0: "rgba(239,68,68,0.20)",     # red
    1: "rgba(34,197,94,0.20)",     # green
    2: "rgba(156,163,175,0.20)",   # gray
}


def classify_macro_regime(result_df, inflation_curve, window=21):
    """
    Rule-based macro regime using inflation momentum:
      - Tightening (0): inflation expectations rising → 1y1y reprices up → spread compresses
      - Easing (1): inflation expectations falling → 1y1y drops → spread widens
      - Transitional (2): no clear trend
    """
    df = result_df.copy()

    infl = inflation_curve[["closeDate", "Inflation_1y"]].copy()
    df = df.merge(infl, on="closeDate", how="left")

    df["infl_delta"] = df["Inflation_1y"].diff(window)

    # Classify
    threshold = df["infl_delta"].rolling(252, min_periods=60).std() * 0.5
    conditions = [
        df["infl_delta"] > threshold,    # rising inflation
        df["infl_delta"] < -threshold,   # falling inflation
    ]
    df["regime"] = np.select(conditions, [0, 1], default=2).astype(float)
    df["regime_label"] = df["regime"].map(REGIME_LABELS)

    return df


def render(result_df, inflation_curve, spread_df, train_end="2017-12-31"):
    """
    Render macro trading section.

    Parameters
    ----------
    result_df : from macro.render_model() — has closeDate, spread, macro_residual
    inflation_curve : for regime classification
    spread_df : for backtest P&L
    train_end : training cutoff
    """
    if result_df is None or "macro_residual" not in result_df.columns:
        st.warning("Run the Macro Fair Value model first.")
        return

    st.divider()
    st.header("Trading the Macro Residual")

    # =================== Macro Regime Classification ===================

    st.subheader("Macro Regime Classification")
    st.markdown("""
    **Rule-based** regime using inflation expectations momentum (21-day change):
    - **Tightening**: inflation rising → short-end reprices up → spread compresses
    - **Easing**: inflation falling → short-end drops → spread widens  
    - **Transitional**: no clear direction — avoid trading
    """)

    regime_df = classify_macro_regime(result_df, inflation_curve)

    # Rename for compatibility with trading functions
    regime_df = regime_df.rename(columns={"macro_residual": "residual_spread"})

    # Regime timeline
    fig_regime = go.Figure()
    for reg_id, label in REGIME_LABELS.items():
        mask = regime_df["regime"] == reg_id
        fig_regime.add_trace(go.Scatter(
            x=regime_df.loc[mask, "closeDate"],
            y=regime_df.loc[mask, "spread"],
            mode="markers", marker=dict(size=3, color=["red", "green", "gray"][reg_id]),
            name=label,
        ))
    fig_regime.update_layout(title="Spread Colored by Macro Regime", yaxis_tickformat=".2%", height=350)
    st.plotly_chart(fig_regime, use_container_width=True)

    # Regime distribution
    col1, col2, col3 = st.columns(3)
    total = len(regime_df)
    for col_st, (reg_id, label) in zip([col1, col2, col3], REGIME_LABELS.items()):
        count = (regime_df["regime"] == reg_id).sum()
        with col_st:
            st.metric(label, f"{count:,} days ({count/total*100:.1f}%)")

    st.divider()

    # =================== Regime Z-Score ===================

    st.subheader("Regime-Conditioned Trading")

    trade_input = regime_df[["closeDate", "regime", "spread", "residual_spread"]].dropna().copy()
    trade_input, regime_stats = compute_regime_zscore(trade_input, train_end=train_end)

    st.markdown("**Residual distribution per regime (training sample)**")
    display_stats = regime_stats.copy()
    display_stats["regime_label"] = display_stats["regime"].map(REGIME_LABELS)
    display_stats["regime_mean_bps"] = (display_stats["regime_mean"] * 10000).round(2)
    display_stats["regime_std_bps"] = (display_stats["regime_std"] * 10000).round(2)
    st.dataframe(
        display_stats[["regime", "regime_label", "regime_mean_bps", "regime_std_bps"]],
        hide_index=True, use_container_width=True,
    )

    # =================== Controls ===================

    all_regimes = sorted(trade_input["regime"].dropna().unique().astype(int).tolist())

    ctrl1, ctrl2, ctrl3 = st.columns(3)
    with ctrl1:
        entry_threshold = st.slider("Entry (σ)", 0.5, 3.0, 1.5, 0.1, key="macro_t_entry")
    with ctrl2:
        exit_band = st.slider("Exit Band (σ)", 0.0, 1.5, 0.25, 0.05, key="macro_t_exit")
    with ctrl3:
        use_stop = st.checkbox("Enable Stop-Loss", value=True, key="macro_t_stop_check")
        stop_loss = st.slider("Stop-Loss (σ)", 2.0, 5.0, 3.0, 0.25, key="macro_t_stop") if use_stop else None

    ctrl4, ctrl5, ctrl6 = st.columns(3)
    with ctrl4:
        roll_freq = st.slider("Roll Frequency (days)", 30, 180, 30, 10, key="macro_t_roll_freq")
    with ctrl5:
        roll_cost_bps = st.slider("Roll Cost (bps)", 0.0, 10.0, 2.0, 0.5, key="macro_t_roll_cost")
    with ctrl6:
        slippage_bps = st.slider("Slippage per trade (bps)", 0.0, 20.0, 1.0, 1.0, key="macro_t_slippage")

    st.markdown("**Tradeable Regimes**")
    tradeable = []
    cols = st.columns(len(all_regimes))
    for i, reg in enumerate(all_regimes):
        label = REGIME_LABELS.get(reg, f"Regime {reg}")
        with cols[i]:
            if st.checkbox(label, value=(reg != 2), key=f"macro_t_regime_{reg}"):
                tradeable.append(reg)
    tradeable_regimes = tradeable if len(tradeable) < len(all_regimes) else None

    # =================== Signal ===================

    signal_df = compute_trading_signal_regime(
        trade_input,
        entry_long=-entry_threshold,
        entry_short=entry_threshold,
        exit_band=exit_band,
        stop_loss=stop_loss,
        tradeable_regimes=tradeable_regimes,
    )

    # Z-Score chart
    fig_z = go.Figure()
    fig_z.add_trace(go.Scatter(x=signal_df["closeDate"], y=signal_df["regime_z"], mode="lines", name="Regime Z-Score"))
    fig_z.add_hline(y=entry_threshold, line_dash="dash", line_color="red")
    fig_z.add_hline(y=-entry_threshold, line_dash="dash", line_color="green")
    fig_z.add_hline(y=exit_band, line_dash="dot", line_color="gray")
    fig_z.add_hline(y=-exit_band, line_dash="dot", line_color="gray")
    fig_z.add_hline(y=0, line_color="lightgray", line_width=0.5)
    if stop_loss is not None:
        fig_z.add_hline(y=stop_loss, line_dash="dashdot", line_color="black")
        fig_z.add_hline(y=-stop_loss, line_dash="dashdot", line_color="black")

    tick_vals = [entry_threshold, -entry_threshold, exit_band, -exit_band]
    tick_labels = [f"Short +{entry_threshold}σ", f"Long −{entry_threshold}σ", f"Exit +{exit_band}σ", f"Exit −{exit_band}σ"]
    if stop_loss is not None:
        tick_vals += [stop_loss, -stop_loss]
        tick_labels += [f"Stop +{stop_loss}σ", f"Stop −{stop_loss}σ"]

    fig_z.update_layout(
        title="Regime-Conditioned Z-Score with Trading Bands",
        yaxis_title="Z-Score",
        yaxis2=dict(overlaying="y", side="right", tickmode="array", tickvals=tick_vals, ticktext=tick_labels, showgrid=False),
        height=420,
    )
    fig_z.add_trace(go.Scatter(x=[signal_df["closeDate"].iloc[0]], y=[0], yaxis="y2", mode="markers", marker=dict(opacity=0), showlegend=False))
    st.plotly_chart(fig_z, use_container_width=True)

    # =================== Backtest ===================

    backtest_df = run_backtest(
        signal_df, slippage_bps=slippage_bps,
        roll_freq=roll_freq, roll_cost_bps=roll_cost_bps,
        train_end=train_end,
    )

    # Position + Regime chart
    fig_pos = go.Figure()
    for reg in all_regimes:
        mask = signal_df["regime"] == reg
        label = REGIME_LABELS.get(reg, f"Regime {reg}")
        fig_pos.add_trace(go.Bar(
            x=signal_df.loc[mask, "closeDate"], y=[1] * mask.sum(),
            marker_color=REGIME_COLORS_BG.get(reg, "rgba(200,200,200,0.15)"),
            name=label, yaxis="y2",
        ))
    pos_colors = backtest_df["position"].map({1.0: "green", -1.0: "red", 0.0: "lightgray"}).fillna("lightgray")
    fig_pos.add_trace(go.Bar(x=backtest_df["closeDate"], y=backtest_df["position"], marker_color=pos_colors, name="Position", showlegend=False))
    fig_pos.update_layout(
        title="Position & Macro Regime",
        yaxis_title="Position",
        yaxis2=dict(overlaying="y", side="right", showticklabels=False, showgrid=False, range=[0, 1]),
        height=280, bargap=0, barmode="overlay",
    )
    st.plotly_chart(fig_pos, use_container_width=True)

    # Cost summary
    total_slippage = backtest_df["slippage_cost"].sum()
    total_roll = backtest_df["roll_cost"].sum()
    total_gross = backtest_df["daily_pnl"].sum()
    total_net = backtest_df["net_pnl"].sum()
    st.caption(
        f"Gross P&L: {total_gross * 10000:.2f} bps | "
        f"Slippage: {total_slippage * 10000:.2f} bps | "
        f"Roll costs: {total_roll * 10000:.2f} bps | "
        f"Net P&L: {total_net * 10000:.2f} bps"
    )

    # =================== Cumulative P&L ===================

    fig_pnl = go.Figure()
    train_bt = backtest_df[backtest_df["is_train"]]
    test_bt = backtest_df[~backtest_df["is_train"]]
    fig_pnl.add_trace(go.Scatter(x=train_bt["closeDate"], y=train_bt["cumulative_pnl"] * 10000, mode="lines", name="Train", line=dict(color="blue", width=1.5)))
    fig_pnl.add_trace(go.Scatter(x=test_bt["closeDate"], y=test_bt["cumulative_pnl"] * 10000, mode="lines", name="Test", line=dict(color="green", width=1.5)))
    fig_pnl.add_hline(y=0, line_color="lightgray", line_width=0.5)
    fig_pnl.add_vline(x=str(TRAINING_DATE.date()), line_dash="dash", line_color="orange")
    fig_pnl.update_layout(title="Cumulative Net P&L (Macro Strategy)", yaxis_title="P&L (bps)", height=400)
    st.plotly_chart(fig_pnl, use_container_width=True)

    # =================== Drawdown ===================

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=backtest_df["closeDate"], y=backtest_df["drawdown"] * 10000,
        mode="lines", name="Drawdown",
        fill="tozeroy", fillcolor="rgba(239,68,68,0.15)",
        line=dict(color="red", width=1),
    ))
    fig_dd.add_vline(x=str(TRAINING_DATE.date()), line_dash="dash", line_color="orange")
    fig_dd.update_layout(title="Drawdown", yaxis_title="Drawdown (bps)", height=300)
    st.plotly_chart(fig_dd, use_container_width=True)

    # =================== Performance Metrics ===================

    st.subheader("Performance Metrics")
    trade_log = build_trade_log(signal_df)
    metrics = compute_performance_metrics(backtest_df, trade_log)

    def _fmt(val, fmt_type="f2"):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "—"
        if fmt_type == "bps":
            return f"{val * 10000:.2f}"
        if fmt_type == "int":
            return f"{int(val)}"
        return f"{val:.2f}"

    metric_rows = [
        ("Total P&L (bps)", "bps"), ("Ann. Return (bps)", "bps"),
        ("Ann. Vol (bps)", "bps"), ("Sharpe", "f2"), ("Sortino", "f2"),
        ("Max Drawdown (bps)", "bps"), ("# Trades", "int"),
        ("Avg Holding (days)", "f2"), ("P&L / Trade (bps)", "bps"),
    ]
    keys = [
        "total_pnl", "annual_return", "annual_vol",
        "sharpe", "sortino", "max_drawdown",
        "n_trades", "avg_holding_days", "pnl_per_trade",
    ]

    perf_data = []
    for (label, fmt), key in zip(metric_rows, keys):
        perf_data.append({
            "Metric": label,
            "Full": _fmt(metrics["all"].get(key), fmt),
            "Train": _fmt(metrics["train"].get(key), fmt),
            "Test": _fmt(metrics["test"].get(key), fmt),
        })
    st.dataframe(pd.DataFrame(perf_data), hide_index=True, use_container_width=True)

    # =================== Monthly Heatmap ===================

    st.subheader("Monthly P&L Heatmap")
    monthly = backtest_df.copy()
    monthly["year"] = monthly["closeDate"].dt.year
    monthly["month"] = monthly["closeDate"].dt.month
    monthly_pnl = monthly.groupby(["year", "month"])["net_pnl"].sum().reset_index()
    monthly_pivot = monthly_pnl.pivot(index="year", columns="month", values="net_pnl")
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly_pivot.columns = [month_names[m - 1] for m in monthly_pivot.columns]

    fig_heat = go.Figure(data=go.Heatmap(
        z=monthly_pivot.values * 10000,
        x=monthly_pivot.columns,
        y=monthly_pivot.index.astype(str),
        colorscale="RdYlGn", zmid=0,
        text=np.round(monthly_pivot.values * 10000, 1),
        texttemplate="%{text}",
        textfont={"size": 10},
    ))
    fig_heat.update_layout(title="Monthly Net P&L (bps)", height=max(300, len(monthly_pivot) * 30 + 80))
    st.plotly_chart(fig_heat, use_container_width=True)

    # =================== Trade Log ===================

    st.subheader("Trade Log")
    if len(trade_log) > 0:
        display_log = trade_log.copy()
        display_log["entry_date"] = display_log["entry_date"].dt.date
        display_log["exit_date"] = display_log["exit_date"].dt.date
        display_log["pnl_bps"] = (display_log["pnl"] * 10000).round(2)
        display_log["entry_z"] = display_log["entry_z"].round(3)
        display_log["exit_z"] = display_log["exit_z"].round(3)
        if "entry_regime" in display_log.columns:
            display_log["regime_label"] = display_log["entry_regime"].map(REGIME_LABELS)
        show_cols = ["entry_date", "exit_date", "direction"]
        if "regime_label" in display_log.columns:
            show_cols.append("regime_label")
        show_cols += ["entry_z", "exit_z", "pnl_bps", "holding_days", "exit_type"]
        st.dataframe(display_log[show_cols], hide_index=True, use_container_width=True)
    else:
        st.info("No trades generated with current parameters.")