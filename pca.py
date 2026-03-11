from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from tradingPCA import (compute_trading_signal,compute_residual_spread,compute_trading_signal_regime,run_backtest,build_trade_log,compute_performance_metrics)
import stationarity
import clustering


REGIME_COLORS = {
    0: "rgba(59,130,246,0.25)",
    1: "rgba(249,115,22,0.25)",
    2: "rgba(139,92,246,0.25)",
    3: "rgba(20,184,166,0.25)",
    4: "rgba(236,72,153,0.25)",
}

def run_pca(curve_df, train_end=None):
    cols = ["Nominal_1y", "Nominal_2y", "Nominal_5y", "Nominal_10y"]
    out_cols = ["1Y", "2Y", "5Y", "10Y"]
    X_all = curve_df[cols].dropna()

    pca = PCA(n_components=3)

    if train_end is not None:
        train_end = pd.to_datetime(train_end)
        train_mask = curve_df.loc[X_all.index, "closeDate"] <= train_end
        pca.fit(X_all[train_mask])
    else:
        pca.fit(X_all)

    pcs = pca.transform(X_all)

    pca_df = curve_df.loc[X_all.index, ["closeDate"]].copy()
    pca_df["Level"]     = pcs[:, 0]
    pca_df["Slope"]     = pcs[:, 1]
    pca_df["Curvature"] = pcs[:, 2]

    loadings = pd.DataFrame(
        pca.components_.T,
        index=out_cols,
        columns=["Level", "Slope", "Curvature"],
    )

    explained = pd.DataFrame({
        "Component": ["Level", "Slope", "Curvature"],
        "Explained Variance": pca.explained_variance_ratio_,
    })

    if train_end is not None:
        ref = pca_df[pca_df["closeDate"] <= train_end]
    else:
        ref = pca_df

    for pc in ["Level", "Slope", "Curvature"]:
        pca_df[f"{pc}_z"] = (pca_df[pc] - ref[pc].mean()) / ref[pc].std()

    return pca_df, loadings, explained


def add_rolling_zscore(df, cols, windows=[60, 120, 252]):
    out = df.copy()
    for col in cols:
        for w in windows:
            rolling_mean = out[col].rolling(w).mean()
            rolling_std = out[col].rolling(w).std()
            out[f"{col}_z_{w}d"] = (out[col] - rolling_mean) / rolling_std
    return out



def plot_pca(nominal_curve):
    st.subheader("PCA Analysis (Exploratory, Window-Dependent)")

    left_panel, middle_panel, right_panel = st.columns([0.4, 0.3, 0.3])
    with left_panel:
        mode = st.radio("Select analysis period",["Predefined Periods", "User defined Dates"],horizontal=True)
    with middle_panel:
        min_date = nominal_curve["closeDate"].min().date()
        max_date = nominal_curve["closeDate"].max().date()
        if mode == "User defined Dates":
            d1, d2 = st.columns(2)
            with d1:
                start_date = st.date_input("Start date",value=min_date,min_value=min_date,max_value=max_date)
            with d2:
                end_date = st.date_input("End date",value=max_date,min_value=min_date,max_value=max_date)

        else:
            period = st.selectbox("Predefined period", ["3Y", "5Y", "10Y", "15Y", "Max Period"], index = 4)

            end_date = max_date
            if period == "3Y": start_date = (pd.Timestamp(end_date) - pd.DateOffset(years=3)).date()
            elif period == "5Y": start_date = (pd.Timestamp(end_date) - pd.DateOffset(years=5)).date()
            elif period == "10Y": start_date = (pd.Timestamp(end_date) - pd.DateOffset(years=10)).date()
            elif period == "15Y": start_date = (pd.Timestamp(end_date) - pd.DateOffset(years=15)).date()
            else: start_date = min_date

            if start_date < min_date:
                start_date = min_date

    with right_panel:
        st.markdown(f"<div style='text-align: right;'>{start_date} to {end_date}</div>", unsafe_allow_html=True)

    curve_pca = nominal_curve[(nominal_curve["closeDate"].dt.date >= start_date) & (nominal_curve["closeDate"].dt.date <= end_date)].copy()

    left_panel, right_panel = st.columns([0.3, 0.7])

    pca_df, loadings, explained = run_pca(curve_pca)
    pca_df = add_rolling_zscore(pca_df, cols=["Level", "Slope", "Curvature"], windows=[60, 120, 252])

    with left_panel:
        st.dataframe(loadings, width="stretch")

        explained_display = explained.copy()
        explained_display["Explained Variance"] = explained_display["Explained Variance"].map(lambda x: f"{x:.2%}")
        st.dataframe(explained_display, width="stretch", hide_index=True)

    with right_panel:
        fig_loadings = go.Figure()

        fig_loadings.add_trace(go.Scatter(x=loadings.index, y=loadings["Level"], name="Level"))
        fig_loadings.add_trace(go.Scatter(x=loadings.index, y=loadings["Slope"], name="Slope"))
        fig_loadings.add_trace(go.Scatter(x=loadings.index, y=loadings["Curvature"], name="Curvature"))

        fig_loadings.update_layout(title="PCA Weights by Vertex",xaxis_title="Vertex",yaxis_title="Weight")

        st.plotly_chart(fig_loadings, width="stretch")


    left_panel, middle_panel, right_panel = st.columns([0.6, 0.2, 0.2])
    with left_panel:
        option = st.radio("How would you like to see the results?",key='visibility',options=["Principal Components", "Z-Scored Principal Components", "Rolling z-score window"],horizontal=True)
    with middle_panel:
        z_window = None
        if option == 'Rolling z-score window':
            z_window = st.selectbox("Rolling z-score window",["60d", "120d", "252d"],index=0)

    row_pc1, row_pc2, row_pc3 = st.columns([0.33, 0.33, 0.33])
    with row_pc1:
        fig_pc1 = go.Figure()
        if option == 'Principal Components':
            fig_pc1.add_trace(go.Scatter(x=pca_df["closeDate"], y=pca_df["Level"], mode="lines", name="Level"))
        elif option == 'Z-Scored Principal Components':
            fig_pc1.add_trace(go.Scatter(x=pca_df["closeDate"], y=pca_df["Level_z"], mode="lines", name="Z-Score"))
        elif option == 'Rolling z-score window':
            if z_window == "60d":
                fig_pc1.add_trace(go.Scatter(x=pca_df["closeDate"], y=pca_df["Level_z_60d"], mode="lines", name="Rolling Z-Score"))
            if z_window == "120d":
                fig_pc1.add_trace(go.Scatter(x=pca_df["closeDate"], y=pca_df["Level_z_120d"], mode="lines", name="Rolling Z-Score"))
            if z_window == "252d":
                fig_pc1.add_trace(go.Scatter(x=pca_df["closeDate"], y=pca_df["Level_z_252d"], mode="lines", name="Rolling Z-Score"))
        fig_pc1.update_layout(title="Level Time Series", xaxis_title="Date", yaxis_title="Level")
        st.plotly_chart(fig_pc1, width="stretch")

    with row_pc2:
        fig_pc2 = go.Figure()
        if option == 'Principal Components':
            fig_pc2.add_trace(go.Scatter(x=pca_df["closeDate"], y=pca_df["Slope"], mode="lines", name="Slope"))
        elif option == 'Z-Scored Principal Components':
            fig_pc2.add_trace(go.Scatter(x=pca_df["closeDate"], y=pca_df["Slope_z"], mode="lines", name="Z-Score"))
        elif option == 'Rolling z-score window':
            if z_window == "60d":
                fig_pc2.add_trace(go.Scatter(x=pca_df["closeDate"], y=pca_df["Slope_z_60d"], mode="lines", name="Rolling Z-Score"))
            if z_window == "120d":
                fig_pc2.add_trace(go.Scatter(x=pca_df["closeDate"], y=pca_df["Slope_z_120d"], mode="lines", name="Rolling Z-Score"))
            if z_window == "252d":
                fig_pc2.add_trace(go.Scatter(x=pca_df["closeDate"], y=pca_df["Slope_z_252d"], mode="lines", name="Rolling Z-Score"))
        fig_pc2.update_layout(title="Slope Time Series", xaxis_title="Date", yaxis_title="Slope")
        st.plotly_chart(fig_pc2, width="stretch")

    with row_pc3:
        fig_pc3 = go.Figure()
        if option == 'Principal Components':
            fig_pc3.add_trace(go.Scatter(x=pca_df["closeDate"], y=pca_df["Curvature"], mode="lines", name="Curvature"))
        elif option == 'Z-Scored Principal Components':
            fig_pc3.add_trace(go.Scatter(x=pca_df["closeDate"], y=pca_df["Curvature_z"], mode="lines", name="Z-Score"))
        elif option == 'Rolling z-score window':
            if z_window == "60d":
                fig_pc3.add_trace(go.Scatter(x=pca_df["closeDate"], y=pca_df["Curvature_z_60d"], mode="lines", name="Rolling Z-Score"))
            if z_window == "120d":
                fig_pc3.add_trace(go.Scatter(x=pca_df["closeDate"], y=pca_df["Curvature_z_120d"], mode="lines", name="Rolling Z-Score"))
            if z_window == "252d":
                fig_pc3.add_trace(go.Scatter(x=pca_df["closeDate"], y=pca_df["Curvature_z_252d"], mode="lines", name="Rolling Z-Score"))
        fig_pc3.update_layout(title="Curvature Time Series", xaxis_title="Date", yaxis_title="Curvature")
        st.plotly_chart(fig_pc3, width="stretch")
    
    return pca_df, start_date,end_date





def render(nominal_curve, spread_df, TRAINING_DATE):
    startdate = spread_df["closeDate"].min().date()
    enddate = spread_df["closeDate"].max().date()

    # =================== Residual ===================
    st.subheader(f"Signal Construction using PCA Arbitrage — Using Full Sample. The results bellow are not period sensitive according to the assigned period above")
    st.caption("The results below use a fixed training period (up to 2017-12-31) and are not affected by the PCA window above.")

    full_pca_df, full_loadings, full_explained = run_pca(nominal_curve, train_end="2017-12-31")
    full_pca_df = add_rolling_zscore(full_pca_df, cols=["Level", "Slope", "Curvature"])

    col_model, col_alpha = st.columns([0.4, 0.6])
    with col_model:
        model_type = st.radio("Regression model", ["ols", "ridge"], horizontal=True, key="pca_model_type")
    with col_alpha:
        if model_type in ["ridge"]:
            alpha = st.slider("Regularization Ridge (α)", 0.001, 1.0, 0.2, 0.01, key="pca_alpha")
        else:
            alpha = 1.0

    residual_df, residual_betas, residual_stats = compute_residual_spread(spread_df, full_pca_df, TRAINING_DATE, model_type=model_type, alpha=alpha)


    st.subheader(f"Residual Spread ({startdate} to {enddate})")
    st.caption("The fair spread is estimated with OLS fitted only on the training sample up to 2017-12-31, then applied to the full sample.")

    latex = r"""
    Residual Spread:
    $$
    \text{Residual}_t = \text{Spread}_t - \widehat{\text{Spread}}_t
    $$
    where
    $$
    \widehat{\text{Spread}}_t = \alpha + \beta_1 \text{Level}_t + \beta_2 \text{Slope}_t + \beta_3 \text{Curvature}_t
    $$
    """
    st.markdown(latex)

    left_res, right_res = st.columns([0.35, 0.65])

    with left_res:
        st.dataframe(residual_betas, hide_index=True)
        fit_table = pd.DataFrame({
            "Metric": ["Train R-squared", "Train Adj. R-squared"],
            "Value": [f"{residual_stats['r2_train']:.4f}",f"{residual_stats['adj_r2_train']:.4f}" if residual_stats['adj_r2_train'] is not None else "N/A"]
        })
        st.dataframe(fit_table, hide_index=True)

    with right_res:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=residual_df["closeDate"], y=residual_df["spread"], mode="lines", name="Actual Spread"))
        fig.add_trace(go.Scatter(x=residual_df["closeDate"], y=residual_df["fair_spread"], mode="lines", name="Fair Spread"))
        fig.update_layout(title="Actual vs Fair Spread", yaxis_tickformat=".2%")
        st.plotly_chart(fig)

    fig_residual = go.Figure()
    fig_residual.add_trace(go.Scatter(x=residual_df["closeDate"], y=residual_df["residual_spread"], mode="lines", name="Residual Spread"))
    fig_residual.update_layout(title="Residual Spread Time Series", yaxis_title="Residual Spread")
    st.plotly_chart(fig_residual)

    st.subheader(f"Residual Spread Stationarity Analysis ({startdate} to {enddate})")
    st.dataframe(stationarity.stationarity_table(residual_df["residual_spread"], name="1Y1Y − 5Y5Y Residual Spread"), hide_index=True)
    st.markdown("The residual helps us answer if the 1y1y−5y5y is high or low given today's level, slope, and curvature. To support this analysis we will add a regime clustering.")

    st.divider()

    # =================== Regime Clustering ===================

    regime_df = full_pca_df[["closeDate", "Level", "Slope", "Curvature"]].copy()
    regime_df = regime_df.merge(residual_df[["closeDate", "spread", "residual_spread"]], on="closeDate", how="inner")
    regime_df = regime_df.dropna().reset_index(drop=True)

    st.subheader(f"Regime Analysis ({startdate} to {enddate})")
    st.markdown("Lets cluster the regimes using 2 different models, Gaussian Mixture Model (GMM) and K-means")

    left_regime, right_regime = st.columns([0.5, 0.5])
    with left_regime:
        clustering_option = st.radio("Choose the model:", options=["gmm", "kmeans"], horizontal=True, key="pca_clustering")
    with right_regime:
        n_regimes = st.selectbox("Number of regimes", [2, 3, 4, 5], index=1, key="pca_n_regimes")

    regime_df, centers, model = clustering.run_regime_model(
        regime_df, ["Level", "Slope", "Curvature", "residual_spread"],
        model_type=clustering_option, n_regimes=n_regimes,
    )

    st.subheader("Regime Timeline")
    fig_regime = go.Figure()
    train_sub = regime_df[regime_df["is_train"] == True]
    test_sub = regime_df[regime_df["is_train"] == False]
    fig_regime.add_trace(go.Scatter(x=train_sub["closeDate"], y=train_sub["regime"], mode="markers", marker=dict(size=7, color="blue"), name="Train"))
    fig_regime.add_trace(go.Scatter(x=test_sub["closeDate"], y=test_sub["regime"], mode="markers", marker=dict(size=7, color="red"), name="Test"))
    fig_regime.update_layout(title="Regime Classification Through Time", yaxis_title="Regime")
    st.plotly_chart(fig_regime)

    cur_reg = int(regime_df["regime"].dropna().iloc[-1])
    st.subheader(f"Current Regime: {cur_reg}")

    st.subheader("Regime Summary")
    summary = regime_df.groupby(["is_train", "regime"]).agg(
        Count=("regime", "size"),
        Avg_Residual=("residual_spread", "mean"),
        Std_Residual=("residual_spread", "std"),
    ).reset_index()
    st.dataframe(summary, hide_index=True)

    st.subheader("Regime Centers")
    st.dataframe(centers[["regime", "Level", "Slope", "Curvature", "residual_spread"]], hide_index=True)

    st.divider()

    # =================== Simple Residual Trading ===================

    st.subheader("Trading the residual")
    st.markdown("A simple trade idea: buy when the z-score is extremely low, sell when extremely high.")

    entry_threshold_simple, exit_band_simple, stop_loss_simple = 1.5, 0.25, 3
    z_window = int(stationarity.compute_half_life(residual_df["residual_spread"]))
    if np.isnan(z_window) or z_window < 5:
        z_window = 30

    st.markdown(f"Entry threshold (σ) = {entry_threshold_simple}, Exit band (σ) = {exit_band_simple}, Stop Loss (σ) = {stop_loss_simple}")

    signal_df_simple = compute_trading_signal(
        residual_df[["closeDate", "residual_spread"]].copy(),
        z_window=z_window,
        entry_long=-entry_threshold_simple,
        entry_short=entry_threshold_simple,
        exit_band=exit_band_simple,
        stop_loss=stop_loss_simple,
    )

    fig_signal = go.Figure()
    fig_signal.add_trace(go.Scatter(x=signal_df_simple["closeDate"], y=signal_df_simple["residual_z"], mode="lines", name="Residual Z-Score"))
    fig_signal.add_hline(y=entry_threshold_simple, line_dash="dash", line_color="red")
    fig_signal.add_hline(y=-entry_threshold_simple, line_dash="dash", line_color="green")
    fig_signal.add_hline(y=exit_band_simple, line_dash="dot", line_color="gray")
    fig_signal.add_hline(y=-exit_band_simple, line_dash="dot", line_color="gray")
    fig_signal.add_hline(y=stop_loss_simple, line_dash="dashdot", line_color="black")
    fig_signal.add_hline(y=-stop_loss_simple, line_dash="dashdot", line_color="black")
    fig_signal.add_hline(y=0, line_color="lightgray", line_width=0.5)
    fig_signal.update_layout(title="Residual Z-Score with Trading Bands", yaxis_title="Z-Score")
    st.plotly_chart(fig_signal)

    fig_pos_simple = go.Figure()
    colors = signal_df_simple["position"].map({1.0: "green", -1.0: "red", 0.0: "gray"})
    fig_pos_simple.add_trace(go.Bar(x=signal_df_simple["closeDate"], y=signal_df_simple["position"], marker_color=colors, name="Position"))
    fig_pos_simple.update_layout(title="Position Over Time", yaxis_title="Position", height=300, bargap=0)
    st.plotly_chart(fig_pos_simple)

    st.divider()

    # =================== Regime-Conditioned Trading ===================

    st.subheader("Trading the residual with Regimes")
    st.markdown("Now, lets trade the residual using different z-score of each regime. This is important so we can analyse each period characteristics.")

    trade_input = regime_df[["closeDate", "regime", "spread", "residual_spread"]].copy()
    trade_input, regime_stats = clustering.compute_regime_zscore(trade_input, train_end="2017-12-31")

    st.markdown("Residual distribution per regime (training sample)")
    display_stats = regime_stats.copy()
    display_stats["regime_mean_bps"] = (display_stats["regime_mean"] * 10000).round(2)
    display_stats["regime_std_bps"] = (display_stats["regime_std"] * 10000).round(2)
    st.dataframe(display_stats[["regime", "regime_mean_bps", "regime_std_bps"]], hide_index=True)

    all_regimes = sorted(trade_input["regime"].dropna().unique().astype(int).tolist())

    # Controls
    ctrl1, ctrl2, ctrl3 = st.columns(3)
    with ctrl1:
        entry_threshold = st.slider("Entry (σ)", 0.5, 3.0, 2.0, 0.1, key="pca_entry")
    with ctrl2:
        exit_band = st.slider("Exit Band (σ)", 0.0, 1.5, 0.4, 0.05, key="pca_exit")

    ctrl4, ctrl5, ctrl6, _ = st.columns(4)
    with ctrl4:
        roll_freq = st.slider("Roll Frequency (days)", 30, 180, 30, 10, key="pca_roll_freq")
    with ctrl5:
        roll_cost_bps = st.slider("Roll Cost (bps)", 0.0, 10.0, 2.0, 0.5, key="pca_roll_cost")
    with ctrl6:
        slippage_bps = st.slider("Slippage per trade (bps)", 0.0, 20.0, 1.0, 1.0, key="pca_slippage")

    ctrl7, ctrl8 = st.columns(2)
    with ctrl7:
        use_stop = st.checkbox("Enable Stop-Loss", value=True, key="pca_stop_check")
        if use_stop:
            stop_loss = st.slider("Stop-Loss (σ)", 2.0, 5.0, 4.0, 0.25, key="pca_stop")
        else:
            stop_loss = None

    with ctrl8:
        st.markdown("Tradeable Regimes")
        tradeable = []
        cols = st.columns(len(all_regimes))
        for i, reg in enumerate(all_regimes):
            with cols[i]:
                if st.checkbox(f"Regime {reg}", value=True, key=f"pca_regime_{reg}"):
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

    fig_signal = go.Figure()
    fig_signal.add_trace(go.Scatter(x=signal_df["closeDate"], y=signal_df["regime_z"], mode="lines", name="Regime Z-Score"))
    fig_signal.add_hline(y=entry_threshold, line_dash="dash", line_color="red")
    fig_signal.add_hline(y=-entry_threshold, line_dash="dash", line_color="green")
    fig_signal.add_hline(y=exit_band, line_dash="dot", line_color="gray")
    fig_signal.add_hline(y=-exit_band, line_dash="dot", line_color="gray")
    fig_signal.add_hline(y=0, line_color="lightgray", line_width=0.5)
    if stop_loss is not None:
        fig_signal.add_hline(y=stop_loss, line_dash="dashdot", line_color="black")
        fig_signal.add_hline(y=-stop_loss, line_dash="dashdot", line_color="black")

    tick_vals = [entry_threshold, -entry_threshold, exit_band, -exit_band]
    tick_labels = [f"Short +{entry_threshold}σ", f"Long −{entry_threshold}σ", f"Exit +{exit_band}σ", f"Exit −{exit_band}σ"]
    if stop_loss is not None:
        tick_vals += [stop_loss, -stop_loss]
        tick_labels += [f"Stop +{stop_loss}σ", f"Stop −{stop_loss}σ"]

    fig_signal.update_layout(
        title="Regime-Conditioned Z-Score with Trading Bands",
        yaxis_title="Z-Score",
        yaxis2=dict(overlaying="y", side="right", tickmode="array", tickvals=tick_vals, ticktext=tick_labels, showgrid=False),
        height=420,
    )
    fig_signal.add_trace(go.Scatter(x=[signal_df["closeDate"].iloc[0]], y=[0], yaxis="y2", mode="markers", marker=dict(opacity=0), showlegend=False))
    st.plotly_chart(fig_signal)

    # Backtest
    backtest_df = run_backtest(signal_df, slippage_bps=slippage_bps, roll_freq=roll_freq, roll_cost_bps=roll_cost_bps, train_end="2017-12-31")

    # Position + Regime chart
    fig_pos = go.Figure()
    for reg in all_regimes:
        mask = signal_df["regime"] == reg
        fig_pos.add_trace(go.Bar(
            x=signal_df.loc[mask, "closeDate"], y=[1] * mask.sum(),
            marker_color=REGIME_COLORS.get(reg, "rgba(200,200,200,0.15)"),
            name=f"Regime {reg}", yaxis="y2",
        ))
    pos_colors = backtest_df["position"].map({1.0: "green", -1.0: "red", 0.0: "lightgray"}).fillna("lightgray")
    fig_pos.add_trace(go.Bar(x=backtest_df["closeDate"], y=backtest_df["position"], marker_color=pos_colors, name="Position", showlegend=False))
    fig_pos.update_layout(
        title="Position & Regime (Long +1 / Flat 0 / Short −1)",
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
    test_bt = backtest_df[~backtest_df["is_train"]]
    fig_pnl.add_trace(go.Scatter(x=train_bt["closeDate"], y=train_bt["cumulative_pnl"] * 10000, mode="lines", name="Train", line=dict(color="blue", width=1.5)))
    fig_pnl.add_trace(go.Scatter(x=test_bt["closeDate"], y=test_bt["cumulative_pnl"] * 10000, mode="lines", name="Test", line=dict(color="green", width=1.5)))
    fig_pnl.add_hline(y=0, line_color="lightgray", line_width=0.5)
    fig_pnl.add_vline(x=TRAINING_DATE, line_dash="dash", line_color="orange")
    fig_pnl.update_layout(title="Cumulative Net P&L", yaxis_title="P&L (bps)", height=400)
    st.plotly_chart(fig_pnl)

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=backtest_df["closeDate"], y=backtest_df["drawdown"] * 10000,
        mode="lines", name="Drawdown",
        fill="tozeroy", fillcolor="rgba(239,68,68,0.15)",
        line=dict(color="red", width=1),
    ))
    fig_dd.add_vline(x=TRAINING_DATE, line_dash="dash", line_color="orange")
    fig_dd.update_layout(title="Drawdown", yaxis_title="Drawdown (bps)", height=300)
    st.plotly_chart(fig_dd)

    # Performance Metrics
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
    st.dataframe(pd.DataFrame(perf_data), hide_index=True)

    st.subheader("Monthly P&L Heatmap")
    monthly = backtest_df.copy()
    monthly["year"] = monthly["closeDate"].dt.year
    monthly["month"] = monthly["closeDate"].dt.month
    monthly_pnl = monthly.groupby(["year", "month"])["net_pnl"].sum().reset_index()
    monthly_pivot = monthly_pnl.pivot(index="year", columns="month", values="net_pnl")
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly_pivot.columns = [month_names[m - 1] for m in monthly_pivot.columns]

    fig_heatmap = go.Figure(data=go.Heatmap(
        z=monthly_pivot.values * 10000,
        x=monthly_pivot.columns,
        y=monthly_pivot.index.astype(str),
        colorscale="RdYlGn", zmid=0,
        text=np.round(monthly_pivot.values * 10000, 1),
        texttemplate="%{text}",
        textfont={"size": 10},
    ))
    fig_heatmap.update_layout(title="Monthly Net P&L (bps)", height=max(300, len(monthly_pivot) * 30 + 80))
    st.plotly_chart(fig_heatmap)

    st.subheader("Trade Log")
    if len(trade_log) > 0:
        display_log = trade_log.copy()
        display_log["entry_date"] = display_log["entry_date"].dt.date
        display_log["exit_date"] = display_log["exit_date"].dt.date
        display_log["pnl_bps"] = (display_log["pnl"] * 10000).round(2)
        display_log["entry_z"] = display_log["entry_z"].round(3)
        display_log["exit_z"] = display_log["exit_z"].round(3)
        show_cols = ["entry_date", "exit_date", "direction", "entry_z", "exit_z", "pnl_bps", "holding_days", "exit_type"]
        if "entry_regime" in display_log.columns:
            show_cols.insert(3, "entry_regime")
        st.dataframe(display_log[show_cols], hide_index=True)
    else:
        st.info("No trades generated with current parameters.")
