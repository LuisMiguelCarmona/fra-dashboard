import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from clustering import compute_regime_zscore
from tradingPCA import compute_trading_signal_regime, run_backtest, build_trade_log, compute_performance_metrics
from validation import run_monte_carlo, compute_stress_test, compute_position_attribution

TRAINING_DATE = pd.to_datetime("2017-12-31")

REGIME_LABELS = {0: "Tightening", 1: "Easing", 2: "Transitional"}
REGIME_COLORS_BG = {
    0: "rgba(239,68,68,0.20)",     # red
    1: "rgba(34,197,94,0.20)",     # green
    2: "rgba(156,163,175,0.20)",   # gray
}


def classify_macro_regime(result_df, inflation_curve, train_end="2017-12-31", window=21):
    df = result_df.copy()
    infl = inflation_curve[["closeDate", "Inflation_1y"]].copy()
    df = df.merge(infl, on="closeDate", how="left")
    df["infl_delta"] = df["Inflation_1y"].diff(window)
    train_end = pd.to_datetime(train_end)
    rolling_std = df["infl_delta"].rolling(252, min_periods=60).std() * 0.5
    threshold = rolling_std.where(df["closeDate"] <= train_end).ffill()
    conditions = [df["infl_delta"] > threshold,df["infl_delta"] < -threshold]
    df["regime"] = np.select(conditions, [0, 1], default=2).astype(float)
    df["regime_label"] = df["regime"].map(REGIME_LABELS)
    return df

def classify_copom_regime(result_df, copom_df):
    df = result_df.copy()
    meetings = copom_df[["meeting_date", "regime", "regime_label"]].rename(columns={"meeting_date": "closeDate"})
    df = df.merge(meetings, on="closeDate", how="left")
    df["regime"] = df["regime"].ffill().fillna(2)
    df["regime_label"] = df["regime_label"].ffill().fillna("Transitional")
    return df

def _render_copom_inspector(copom_df):
    st.subheader("COPOM Meeting Inspector")

    options = [f"{row['meeting_date'].date()}  —  {row['regime_label']}" for _, row in copom_df.iterrows()]
    
    selected_label = st.selectbox("Select a COPOM meeting", options, key="copom_inspector_select")
    selected_idx = options.index(selected_label)
    meeting = copom_df.iloc[selected_idx]


    st.divider()

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Meeting Date", str(meeting["meeting_date"].date()))
    with m2:
        icon = {"Tightening": "", "Easing": "", "Transitional": ""}.get(meeting["regime_label"], "⚪")
        st.metric("Regime", f"{icon} {meeting['regime_label']}")
    with m3:
        st.metric("Sentiment Score", f"{meeting['sentiment_score']:.2f}")

    with st.expander("Committee Reasoning", expanded=False):
        st.markdown(meeting["reasoning"])

    with st.expander("Key Passages from Minutes", expanded=False):
        for i, passage in enumerate(meeting["key_passages"], 1):
            st.markdown(f"**{i}.** _{passage}_")


def _render_trading_block(regime_df, train_end, suffix):
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
    fig_regime.update_layout(title="Spread Colored by Regime", yaxis_tickformat=".2%", height=350)
    st.plotly_chart(fig_regime)

    col1, col2, col3 = st.columns(3)
    total = len(regime_df)
    for col_st, (reg_id, label) in zip([col1, col2, col3], REGIME_LABELS.items()):
        count = (regime_df["regime"] == reg_id).sum()
        with col_st:
            st.metric(label, f"{count:,} days ({count/total*100:.1f}%)")

    st.divider()
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
        hide_index=True,
    )

    all_regimes = sorted(trade_input["regime"].dropna().unique().astype(int).tolist())

    ctrl1, ctrl2, ctrl3 = st.columns(3)
    with ctrl1:
        entry_threshold = st.slider("Entry (σ)", 0.5, 3.0, 1.5, 0.1, key=f"entry_{suffix}")
    with ctrl2:
        exit_band = st.slider("Exit Band (σ)", 0.0, 1.5, 0.25, 0.05, key=f"exit_{suffix}")
    with ctrl3:
        use_stop = st.checkbox("Enable Stop-Loss", value=True, key=f"stop_check_{suffix}")
        stop_loss = st.slider("Stop-Loss (σ)", 2.0, 5.0, 3.0, 0.25, key=f"stop_{suffix}") if use_stop else None

    ctrl4, ctrl5, ctrl6 = st.columns(3)
    with ctrl4:
        roll_freq = st.slider("Roll Frequency (days)", 30, 180, 30, 10, key=f"roll_freq_{suffix}")
    with ctrl5:
        roll_cost_bps = st.slider("Roll Cost (bps)", 0.0, 10.0, 2.0, 0.5, key=f"roll_cost_{suffix}")
    with ctrl6:
        slippage_bps = st.slider("Slippage per trade (bps)", 0.0, 20.0, 1.0, 1.0, key=f"slippage_{suffix}")

    st.markdown("**Tradeable Regimes**")
    tradeable = []
    cols = st.columns(len(all_regimes))
    for i, reg in enumerate(all_regimes):
        label = REGIME_LABELS.get(reg, f"Regime {reg}")
        with cols[i]:
            if st.checkbox(label, value=(reg != 2), key=f"regime_{reg}_{suffix}"):
                tradeable.append(reg)
    tradeable_regimes = tradeable if len(tradeable) < len(all_regimes) else None

    signal_df = compute_trading_signal_regime(
        trade_input,
        entry_long=-entry_threshold,
        entry_short=entry_threshold,
        exit_band=exit_band,
        stop_loss=stop_loss,
        tradeable_regimes=tradeable_regimes,
    )

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
    st.plotly_chart(fig_z)

    backtest_df = run_backtest(
        signal_df, slippage_bps=slippage_bps,
        roll_freq=roll_freq, roll_cost_bps=roll_cost_bps,
        train_end=train_end,
    )

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
        title="Position & Regime",
        yaxis_title="Position",
        yaxis2=dict(overlaying="y", side="right", showticklabels=False, showgrid=False, range=[0, 1]),
        height=280, bargap=0, barmode="overlay",
    )
    st.plotly_chart(fig_pos)

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

    fig_pnl = go.Figure()
    train_bt = backtest_df[backtest_df["is_train"]]
    test_bt  = backtest_df[~backtest_df["is_train"]]
    fig_pnl.add_trace(go.Scatter(x=train_bt["closeDate"], y=train_bt["cumulative_pnl"] * 10000, mode="lines", name="Train", line=dict(color="blue", width=1.5)))
    fig_pnl.add_trace(go.Scatter(x=test_bt["closeDate"],  y=test_bt["cumulative_pnl"]  * 10000, mode="lines", name="Test",  line=dict(color="green", width=1.5)))
    fig_pnl.add_hline(y=0, line_color="lightgray", line_width=0.5)
    fig_pnl.add_vline(x=str(TRAINING_DATE.date()), line_dash="dash", line_color="orange")
    fig_pnl.update_layout(title="Cumulative Net P&L (Macro Strategy)", yaxis_title="P&L (bps)", height=400)
    st.plotly_chart(fig_pnl)

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=backtest_df["closeDate"], y=backtest_df["drawdown"] * 10000,
        mode="lines", name="Drawdown",
        fill="tozeroy", fillcolor="rgba(239,68,68,0.15)",
        line=dict(color="red", width=1),
    ))
    fig_dd.add_vline(x=str(TRAINING_DATE.date()), line_dash="dash", line_color="orange")
    fig_dd.update_layout(title="Drawdown", yaxis_title="Drawdown (bps)", height=300)
    st.plotly_chart(fig_dd)

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
            "Full":  _fmt(metrics["all"].get(key),   fmt),
            "Train": _fmt(metrics["train"].get(key), fmt),
            "Test":  _fmt(metrics["test"].get(key),  fmt),
        })
    st.dataframe(pd.DataFrame(perf_data), hide_index=True)

    st.subheader("Monthly P&L Heatmap")
    monthly = backtest_df.copy()
    monthly["year"]  = monthly["closeDate"].dt.year
    monthly["month"] = monthly["closeDate"].dt.month
    monthly_pnl = monthly.groupby(["year", "month"])["net_pnl"].sum().reset_index()
    monthly_pivot = monthly_pnl.pivot(index="year", columns="month", values="net_pnl")
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
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
    st.plotly_chart(fig_heat)

    st.subheader("Trade Log")
    if len(trade_log) > 0:
        display_log = trade_log.copy()
        display_log["entry_date"] = display_log["entry_date"].dt.date
        display_log["exit_date"]  = display_log["exit_date"].dt.date
        display_log["pnl_bps"]    = (display_log["pnl"] * 10000).round(2)
        display_log["entry_z"]    = display_log["entry_z"].round(3)
        display_log["exit_z"]     = display_log["exit_z"].round(3)
        if "entry_regime" in display_log.columns:
            display_log["regime_label"] = display_log["entry_regime"].map(REGIME_LABELS)
        show_cols = ["entry_date", "exit_date", "direction"]
        if "regime_label" in display_log.columns:
            show_cols.append("regime_label")
        show_cols += ["entry_z", "exit_z", "pnl_bps", "holding_days", "exit_type"]
        st.dataframe(display_log[show_cols], hide_index=True)
    else:
        st.info("No trades generated with current parameters.")

    # =================== Position Attribution ===================
    st.divider()
    st.subheader("Position-Level P&L Attribution")
    st.markdown("Decomposing performance by direction reveals whether alpha comes from long, short, or is symmetric.")
    attrib_df = compute_position_attribution(backtest_df)
    if len(attrib_df) > 0:
        display_attrib = attrib_df.copy()
        for col in ["Gross P&L (bps)", "Slippage (bps)", "Roll Cost (bps)", "Net P&L (bps)"]:
            display_attrib[col] = display_attrib[col].map(lambda x: f"{x:.1f}")
        display_attrib["% of Sample"] = display_attrib["% of Sample"].map(lambda x: f"{x:.1f}%")
        display_attrib["Hit Rate"] = display_attrib["Hit Rate"].map(lambda x: f"{x:.1%}" if not np.isnan(x) else "—")
        st.dataframe(display_attrib, hide_index=True)

    # =================== Stress Test ===================
    st.divider()
    st.subheader("Stress Test — Key Market Episodes")
    st.markdown("How does the strategy perform during known crisis and regime-change periods?")
    stress_df = compute_stress_test(backtest_df, trade_log)
    if len(stress_df) > 0:
        display_stress = stress_df.copy()
        for col in ["Total P&L (bps)", "Ann. Return (bps)", "Ann. Vol (bps)", "Max DD (bps)"]:
            if col in display_stress.columns:
                display_stress[col] = display_stress[col].map(lambda x: f"{x:.1f}" if not np.isnan(x) else "—")
        display_stress["Sharpe"] = display_stress["Sharpe"].map(lambda x: f"{x:.2f}" if not np.isnan(x) else "—")
        display_stress["Hit Rate"] = display_stress["Hit Rate"].map(lambda x: f"{x:.1%}" if not np.isnan(x) else "—")
        display_stress["% Active"] = display_stress["% Active"].map(lambda x: f"{x:.1f}%")
        st.dataframe(display_stress.drop(columns=["Start", "End"]), hide_index=True)
    else:
        st.info("No stress periods overlap with backtest data.")

    # =================== Monte Carlo Validation ===================
    st.divider()
    st.subheader("Signal Validation — Monte Carlo Simulation")
    st.markdown("""
    To test whether the macro signal contains genuine predictive information, we construct a **directional random walk null model**.
    The procedure preserves the complete execution infrastructure (slippage, roll costs) while replacing only the signal direction
    with random draws from {−1, 0, +1} using the same empirical frequency. Repeated **500 times**.
    """)

    if st.button("Run Monte Carlo (500 sims)", key=f"mc_btn_{suffix}"):
        with st.spinner("Running 500 Monte Carlo simulations..."):
            def _bt_func(sig):
                return run_backtest(
                    sig, slippage_bps=slippage_bps,
                    roll_freq=roll_freq, roll_cost_bps=roll_cost_bps,
                    train_end=train_end,
                )

            mc_result = run_monte_carlo(signal_df, _bt_func, n_sims=500)

        col_mc1, col_mc2 = st.columns([0.4, 0.6])

        with col_mc1:
            st.markdown("**Results**")
            mc_data = {
                "Metric": ["Real Model Sharpe", "MC Mean Sharpe", "MC 5th–95th Pct",
                           "Percentile", "Z-Score"],
                "Value": [
                    f"{mc_result['real_sharpe']:.3f}",
                    f"{mc_result['mc_mean']:.3f}",
                    f"[{mc_result['mc_5th']:.3f}, {mc_result['mc_95th']:.3f}]",
                    f"{mc_result['percentile']:.1f}%",
                    f"{mc_result['z_score']:.2f}σ",
                ],
            }
            st.dataframe(pd.DataFrame(mc_data), hide_index=True)

            if mc_result["percentile"] >= 95:
                st.success(f"Signal validated: real model at {mc_result['percentile']:.1f}th percentile of random strategies.")
            elif mc_result["percentile"] >= 75:
                st.warning(f"Moderate evidence: {mc_result['percentile']:.1f}th percentile.")
            else:
                st.error(f"Weak evidence: {mc_result['percentile']:.1f}th percentile.")

        with col_mc2:
            fig_mc = go.Figure()
            fig_mc.add_trace(go.Histogram(
                x=mc_result["mc_sharpes"], nbinsx=40,
                name="Random Walks", marker_color="steelblue", opacity=0.7,
            ))
            fig_mc.add_vline(
                x=mc_result["real_sharpe"], line_dash="dash", line_color="red",
                annotation_text=f"Real Model ({mc_result['real_sharpe']:.3f})",
                annotation_position="top right",
            )
            fig_mc.add_vline(
                x=mc_result["mc_mean"], line_dash="dot", line_color="gray",
                annotation_text=f"MC Mean ({mc_result['mc_mean']:.3f})",
                annotation_position="top left",
            )
            fig_mc.update_layout(
                title="Monte Carlo Sharpe Distribution (500 sims)",
                xaxis_title="Sharpe Ratio", yaxis_title="Frequency", height=400,
            )
            st.plotly_chart(fig_mc)

    return backtest_df, signal_df



def render(result_df, inflation_curve, spread_df, copom_df, train_end="2017-12-31"):
    if result_df is None or "macro_residual" not in result_df.columns:
        st.warning("Run the Macro Fair Value model first.")
        return

    st.divider()
    st.header("Trading the Macro Residual")

    has_copom = copom_df is not None and len(copom_df) > 0

    if has_copom:
        regime_mode = st.radio(
            "Regime Classification Method",
            ["Inflation Momentum", "COPOM Sentiment Analysis"],
            horizontal=True,
            key="regime_mode_radio",
        )
    else:
        regime_mode = "Inflation Momentum"

    st.divider()

    # =================== Inflation Momentum ===================

    if regime_mode == "Inflation Momentum":
        st.subheader("Macro Regime — Inflation Momentum")
        st.markdown("""
        **Rule-based** regime using inflation expectations momentum (21-day change):
        - **Tightening**: inflation rising → short-end reprices up → spread compresses
        - **Easing**: inflation falling → short-end drops → spread widens
        - **Transitional**: no clear direction — avoid trading
        """)
        regime_df = classify_macro_regime(result_df, inflation_curve, train_end=train_end)
        _render_trading_block(regime_df, train_end, suffix="mom")

    # =================== COPOM Sentiment ===================

    else:
        st.subheader("Macro Regime — COPOM Sentiment Analysis")
        st.markdown("""
        Regime assigned from **NLP classification of COPOM meeting minutes**.
        Each meeting's regime is forward-filled until the next meeting.
        - **Tightening**: hawkish bias — no space for easing, upside inflation risks
        - **Easing**: dovish bias — room for cuts, inflation converging to target
        - **Transitional**: mixed signals, data-dependent stance
        """)

        # COPOM timeline overview
        fig_cop = go.Figure()
        color_map = {0: "red", 1: "green", 2: "gray"}
        regime_df = classify_copom_regime(result_df, copom_df)
        _render_copom_inspector(copom_df)
        _render_trading_block(regime_df, train_end, suffix="copom")

        st.divider()