from macroAnalysis import (
    build_macro_features,
    compute_vif,
    run_granger_tests,
    run_engle_granger,
    compute_macro_residual_ols,
    compute_macro_residual_rolling_ridge,
    compare_regularization_models,
)
from validation import (
    walk_forward_backtest,
    run_zivot_andrews,
    run_chow_test,
    compute_rolling_ols_stability,
    time_series_cv,
)
from stationarity import stationarity_table
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


def _dual_axis_changes_plot(dates, spread_change, macro_change, spread_label, macro_label, title):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(x=dates, y=spread_change, mode="lines", name=spread_label, line=dict(color="blue", width=1)),secondary_y=False)
    fig.add_trace(go.Scatter(x=dates, y=macro_change, mode="lines", name=macro_label, line=dict(color="orange", width=1, dash="dot")),secondary_y=True)

    fig.update_yaxes(title_text=spread_label, secondary_y=False)
    fig.update_yaxes(title_text=macro_label, secondary_y=True)
    fig.update_layout(title=title, height=350, showlegend=True)
    return fig


def render(macro_df, spread_df, inflation_curve, nominal_curve):
    st.subheader("Macro Drivers")
    st.markdown("""The 1yr1yr is heavily influenced by near term inflation surprises, short term news and the current government’s economic and fiscal stance, as these factors directly influence the Central Bank’s reaction function over the next few policy meetings. As a result, this part of the curve tends to react quickly to changes in inflation expectations and short term risk sentiment.""")
    st.markdown("""In contrast, the 5y5y is more influenced by longer term inflation anchoring, fiscal sustainability, and institutional and political stability. Movements in this segment are typically associated with shifts in long run credibility rather than transitory shocks.""")

    # =================== Inflation Expectations ===================

    st.markdown("#### Inflation Expectations")
    infl_cols = [c for c in inflation_curve.columns if c.startswith("Inflation_")]

    col_l, col_r = st.columns(2)

    with col_l:
        fig = go.Figure()
        for col in infl_cols:
            label = col.replace("Inflation_", "").upper()
            fig.add_trace(go.Scatter(x=inflation_curve["closeDate"], y=inflation_curve[col], mode="lines", name=label))
        fig.update_layout(title="Inflation Expectations by Tenor", yaxis_title="Inflation Rate", yaxis_tickformat=".2%", height=350)
        st.plotly_chart(fig)

    with col_r:
        merged = spread_df.merge(inflation_curve[["closeDate", "Inflation_1y"]], on="closeDate", how="inner")
        merged["spread_chg"] = merged["spread"].diff()
        merged["infl_chg"] = merged["Inflation_1y"].diff()
        merged = merged.dropna()

        fig = _dual_axis_changes_plot(
            merged["closeDate"], merged["spread_chg"], merged["infl_chg"],
            "Δ Spread (bps)", "Δ Inflation 1Y",
            "Daily Changes: Spread vs Inflation 1Y",
        )
        st.plotly_chart(fig)

    # =================== CDS ===================

    cds_cols = sorted([c for c in macro_df.columns if c.lower().startswith("cds")])
    if cds_cols:
        st.markdown("#### Brazil CDS Spreads (Sovereign Risk)")
        st.caption("CDS are denominated in USD - note inherent correlation with USDBRL.")

        col_l, col_r = st.columns(2)

        with col_l:
            fig = go.Figure()
            for col in cds_cols:
                fig.add_trace(go.Scatter(x=macro_df["closeDate"], y=macro_df[col], mode="lines", name=col.upper()))
            fig.update_layout(title="CDS by Tenor", yaxis_title="CDS Spread (USD)", height=350)
            st.plotly_chart(fig)

        with col_r:
            cds5_col = next((c for c in cds_cols if "5" in c), cds_cols[0])
            merged_cds = spread_df.merge(macro_df[["closeDate", cds5_col]], on="closeDate", how="inner").dropna()
            merged_cds["spread_chg"] = merged_cds["spread"].diff()
            merged_cds["cds_chg"] = merged_cds[cds5_col].diff()
            merged_cds = merged_cds.dropna()

            fig = _dual_axis_changes_plot(
                merged_cds["closeDate"], merged_cds["spread_chg"], merged_cds["cds_chg"],
                "Δ Spread (bps)", f"Δ {cds5_col.upper()}",
                f"Daily Changes: Spread vs {cds5_col.upper()}",
            )
            st.plotly_chart(fig)

    # =================== US 10Y ===================

    if "us10y" in macro_df.columns:
        st.markdown("#### US 10Y Treasury")

        col_l, col_r = st.columns(2)

        with col_l:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=macro_df["closeDate"], y=macro_df["us10y"], mode="lines", name="US 10Y", line=dict(color="green")))
            fig.update_layout(title="US 10Y Yield", yaxis_title="Yield", height=350)
            st.plotly_chart(fig)

        with col_r:
            merged_us = spread_df.merge(macro_df[["closeDate", "us10y"]], on="closeDate", how="inner").dropna()
            merged_us["spread_chg"] = merged_us["spread"].diff()
            merged_us["us10y_chg"] = merged_us["us10y"].diff()
            merged_us = merged_us.dropna()

            fig = _dual_axis_changes_plot(
                merged_us["closeDate"], merged_us["spread_chg"], merged_us["us10y_chg"],
                "Δ Spread (bps)", "Δ US 10Y",
                "Daily Changes: Spread vs US 10Y",
            )
            st.plotly_chart(fig)

    # =================== USDBRL ===================

    if "usdbrl" in macro_df.columns:
        st.markdown("#### USDBRL")

        col_l, col_r = st.columns(2)

        with col_l:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=macro_df["closeDate"], y=macro_df["usdbrl"], mode="lines", name="USDBRL", line=dict(color="purple")))
            fig.update_layout(title="USDBRL", yaxis_title="BRL per USD", height=350)
            st.plotly_chart(fig)

        with col_r:
            merged_fx = spread_df.merge(macro_df[["closeDate", "usdbrl"]], on="closeDate", how="inner").dropna()
            merged_fx["spread_chg"] = merged_fx["spread"].diff()
            merged_fx["usdbrl_chg"] = merged_fx["usdbrl"].diff()
            merged_fx = merged_fx.dropna()

            fig = _dual_axis_changes_plot(
                merged_fx["closeDate"], merged_fx["spread_chg"], merged_fx["usdbrl_chg"],
                "Δ Spread (bps)", "Δ USDBRL",
                "Daily Changes: Spread vs USDBRL",
            )
            st.plotly_chart(fig)

    # =================== Correlation Matrix ===================

    st.divider()
    st.subheader("Correlation Matrix (Daily Changes)")
    st.markdown("Note that, as said previously, this Brazil CDS is in USD and has an inherent correlation with USDBRL.")

    corr_df = spread_df[["closeDate", "spread"]].copy()
    corr_cols = []

    for col in infl_cols:
        corr_df = corr_df.merge(inflation_curve[["closeDate", col]], on="closeDate", how="inner")
        corr_cols.append(col)

    for col in cds_cols:
        corr_df = corr_df.merge(macro_df[["closeDate", col]], on="closeDate", how="inner")
        corr_cols.append(col)

    if "us10y" in macro_df.columns:
        corr_df = corr_df.merge(macro_df[["closeDate", "us10y"]], on="closeDate", how="inner")
        corr_cols.append("us10y")

    if "usdbrl" in macro_df.columns:
        corr_df = corr_df.merge(macro_df[["closeDate", "usdbrl"]], on="closeDate", how="inner")
        corr_cols.append("usdbrl")

    if corr_cols:
        changes = corr_df[["spread"] + corr_cols].diff().dropna()
        corr_matrix = changes.corr()

        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale="RdBu_r",
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 11},
        ))
        fig_corr.update_layout(height=450, title="Correlation Matrix (Daily Changes)")
        st.plotly_chart(fig_corr)





# =================== MODEL SECTION ===================

def render_model(macro_df, spread_df, inflation_curve, train_end="2017-12-31"):
    st.divider()
    st.subheader("Macro Fair Value Model")
    st.markdown("""
    Lets model **Spread = f(macro features)** and trade the residual.
    Unlike the PCA approach (R² ≈ 0.98 — near-identity), macro drivers provide an economically meaningful fair value.
    """)

    infl_cols = [c for c in inflation_curve.columns if c.startswith("Inflation_")]
    cds_cols = sorted([c for c in macro_df.columns if c.lower().startswith("cds")])
    other_cols = [c for c in macro_df.columns if c not in ["closeDate"] + cds_cols]
    full_macro = macro_df.copy()
    for col in infl_cols:
        if col not in full_macro.columns:
            full_macro = full_macro.merge(inflation_curve[["closeDate", col]], on="closeDate", how="left")
    all_available = [c for c in full_macro.columns if c != "closeDate"]
    default_picks = []
    for candidate in ["Inflation_1y", "cds5y", "us10y", "usdbrl"]:
        if candidate in all_available:
            default_picks.append(candidate)

    st.markdown("#### Feature Selection")
    macro1, macro2, macro3, macro4 = st.columns(4)
    selected_levels = []

    with macro1:
        st.markdown("##### Inflation")
        if "Inflation_1y" in all_available and st.checkbox("Inflation 1Y", value=True, key="m_infl1y"):
            selected_levels.append("Inflation_1y")
        if "Inflation_2y" in all_available and st.checkbox("Inflation 2Y", key="m_infl2y"):
            selected_levels.append("Inflation_2y")
        if "Inflation_5y" in all_available and st.checkbox("Inflation 5Y", key="m_infl5y"):
            selected_levels.append("Inflation_5y")
        if "Inflation_10y" in all_available and st.checkbox("Inflation 10Y", key="m_infl10y"):
            selected_levels.append("Inflation_10y")

    with macro2:
        st.markdown("##### CDS Brazil")
        if "cds1y" in all_available and st.checkbox("CDS 1Y", key="m_cds1y"):
            selected_levels.append("cds1y")
        if "cds2y" in all_available and st.checkbox("CDS 2Y", key="m_cds2y"):
            selected_levels.append("cds2y")
        if "cds5y" in all_available and st.checkbox("CDS 5Y", key="m_cds5y"):
            selected_levels.append("cds5y")
        if "cds10y" in all_available and st.checkbox("CDS 10Y", value=True, key="m_cds10y"):
            selected_levels.append("cds10y")

    with macro3:
        st.markdown("##### US Treasury")
        if "us10y" in all_available and st.checkbox("US 10Y", value=True, key="m_us10y"):
            selected_levels.append("us10y")

    with macro4:
        st.markdown("##### USDBRL")
        if "usdbrl" in all_available and st.checkbox("USDBRL", value=True, key="m_usdbrl"):
            selected_levels.append("usdbrl")


    if len(selected_levels) < 1:
        st.warning("Select at least one macro variable.")
        return None

    # ---- Feature Engineering ----
    st.markdown("#### Feature Engineering")
    st.markdown("For each variable: **level z-score** (rolling 252d) + **momentum** (21d, 63d changes).")

    features_df = build_macro_features(full_macro, selected_levels)
    feature_cols = [c for c in features_df.columns if c != "closeDate"]

    st.caption(f"Generated {len(feature_cols)} features from {len(selected_levels)} variables")

    vif, granger = st.columns(2)
    with vif:
        # ---- VIF ----
        st.markdown("#### Multicollinearity Check (VIF)")
        st.markdown("VIF > 10 indicates severe multicollinearity. Those features are auto-dropped.")

        merged_for_vif = spread_df.merge(features_df, on="closeDate", how="inner").dropna()
        train_mask = merged_for_vif["closeDate"] <= pd.to_datetime(train_end)
        train_features = merged_for_vif.loc[train_mask, feature_cols]

        vif_table = compute_vif(train_features)
        
        st.dataframe(vif_table, hide_index=True)

        good_features = vif_table[vif_table["VIF"] <= 10]["Feature"].tolist()
        dropped = set(feature_cols) - set(good_features)
        if dropped:
            st.markdown(f"Dropped {len(dropped)} features: {', '.join(sorted(dropped))}")
        else:
            st.markdown("All features pass VIF < 10")
        
        feature_cols = good_features

        if len(feature_cols) < 1:
            st.error("No features remaining after VIF filter. Adjust variable selection.")
            return None
        
    with granger:
        # ---- Granger Causality ----
        st.markdown("#### Granger Causality Tests")
        st.markdown("Does each macro feature **Granger-cause** the spread?")

        granger_merged = spread_df.merge(features_df, on="closeDate", how="inner").dropna()
        granger_train = granger_merged[granger_merged["closeDate"] <= pd.to_datetime(train_end)]

        granger_results = run_granger_tests(
            granger_train["spread"],
            granger_train,
            feature_cols,
            max_lag=10,
        )
        st.dataframe(granger_results, hide_index=True)

        n_significant = (granger_results["Significant (5%)"] == "✓").sum()
        st.caption(f"{n_significant} of {len(granger_results)} features are significant at 5%.")

    st.divider()

    # ---- Regression Model ----
    st.markdown("#### Fair Value Regression")

    model_type = st.radio(
        "Model",
        ["Static OLS (train/test split)", "Rolling Ridge (walk-forward)", "Walk-Forward Expanding Window"],
        horizontal=True,
        key="macro_model_type",
    )

    if model_type == "Static OLS (train/test split)":
        result_df, betas, stats = compute_macro_residual_ols(
            spread_df, features_df, feature_cols, train_end=train_end,
        )

        col_l, col_r = st.columns([0.35, 0.65])
        with col_l:
            st.markdown("**Coefficients**")
            display_betas = betas.copy()
            display_betas["Beta"] = display_betas["Beta"].map(lambda x: f"{x:.6f}")
            display_betas["p-value"] = display_betas["p-value"].map(lambda x: f"{x:.4f}")
            st.dataframe(display_betas, hide_index=True)

            st.markdown("**Fit**")
            fit_rows = [
                ("Train R²", f"{stats['r2_train']:.4f}"),
                ("Train Adj. R²", f"{stats['adj_r2_train']:.4f}"),
                ("Test R²", f"{stats['r2_test']:.4f}" if not np.isnan(stats['r2_test']) else "N/A"),
            ]
            st.dataframe(pd.DataFrame(fit_rows, columns=["Metric", "Value"]), hide_index=True)

        with col_r:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=result_df["closeDate"], y=result_df["spread"], mode="lines", name="Actual Spread"))
            fig.add_trace(go.Scatter(x=result_df["closeDate"], y=result_df["fair_value"], mode="lines", name="Macro Fair Value"))
            fig.add_vline(x=str(pd.to_datetime(train_end).date()), line_dash="dash", line_color="orange")
            fig.update_layout(title="Actual Spread vs Macro Fair Value", yaxis_tickformat=".2%", height=400)
            st.plotly_chart(fig)

    elif model_type == "Rolling Ridge (walk-forward)":
        ctrl1, ctrl2 = st.columns(2)
        with ctrl1:
            window = st.slider("Rolling window (business days)", 252, 756, 504, 63, key="macro_ridge_window")
        with ctrl2:
            alpha = st.slider("Ridge α (regularization)", 0.1, 10.0, 1.0, 0.1, key="macro_ridge_alpha")

        with st.spinner("Running Rolling Ridge..."):
            result_df, rolling_betas = compute_macro_residual_rolling_ridge(
                spread_df, features_df, feature_cols,
                window=window, alpha=alpha,
            )

        col_l, col_r = st.columns([0.4, 0.6])
        with col_l:
            fig_r2 = go.Figure()
            fig_r2.add_trace(go.Scatter(x=result_df["closeDate"], y=result_df["rolling_r2"], mode="lines", name="Rolling R²"))
            fig_r2.update_layout(title="In-Sample Rolling R²", height=300)
            st.plotly_chart(fig_r2)

        with col_r:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=result_df["closeDate"], y=result_df["spread"], mode="lines", name="Actual Spread"))
            fig.add_trace(go.Scatter(x=result_df["closeDate"], y=result_df["fair_value"], mode="lines", name="Rolling Ridge Fair Value"))
            fig.update_layout(title="Actual Spread vs Rolling Ridge Fair Value", yaxis_tickformat=".2%", height=400)
            st.plotly_chart(fig)

        # Rolling betas
        if len(rolling_betas) > 0:
            st.markdown("**Coefficient Stability (Rolling Betas)**")
            fig_betas = go.Figure()
            beta_cols = [c for c in rolling_betas.columns if c != "closeDate"]
            for col in beta_cols:
                fig_betas.add_trace(go.Scatter(x=rolling_betas["closeDate"], y=rolling_betas[col], mode="lines", name=col))
            fig_betas.update_layout(title="Rolling Ridge Betas Over Time", height=350)
            st.plotly_chart(fig_betas)

    else:  # Walk-Forward Expanding Window
        ctrl1, ctrl2, ctrl3 = st.columns(3)
        with ctrl1:
            wf_initial = st.slider("Initial training (days)", 252, 756, 504, 63, key="wf_initial")
        with ctrl2:
            wf_step = st.slider("Refit step (days)", 21, 126, 63, 21, key="wf_step")
        with ctrl3:
            wf_model = st.selectbox("Model", ["ridge", "lasso", "elasticnet", "ols"], key="wf_model")

        with st.spinner("Running Walk-Forward backtest..."):
            result_df, fold_df = walk_forward_backtest(
                spread_df, features_df, feature_cols,
                initial_train=wf_initial, step=wf_step,
                model_type=wf_model,
            )

        # Rename for downstream compatibility
        result_df["fair_value"] = result_df["wf_fair_value"]
        result_df["macro_residual"] = result_df["wf_residual"]

        col_l, col_r = st.columns([0.35, 0.65])
        with col_l:
            st.markdown("**Walk-Forward Fold Metrics**")
            display_folds = fold_df.copy()
            display_folds["oos_start"] = display_folds["oos_start"].dt.date
            display_folds["oos_end"] = display_folds["oos_end"].dt.date
            display_folds["r2_train"] = display_folds["r2_train"].map(lambda x: f"{x:.4f}")
            display_folds["r2_oos"] = display_folds["r2_oos"].map(lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A")
            st.dataframe(
                display_folds[["fold", "oos_start", "oos_end", "train_size", "r2_train", "r2_oos"]],
                hide_index=True, height=300,
            )

            avg_oos_r2 = fold_df["r2_oos"].dropna().mean()
            st.metric("Avg OOS R²", f"{avg_oos_r2:.4f}")

        with col_r:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=result_df["closeDate"], y=result_df["spread"], mode="lines", name="Actual Spread"))
            fig.add_trace(go.Scatter(x=result_df["closeDate"], y=result_df["fair_value"], mode="lines", name="Walk-Forward Fair Value"))
            fig.update_layout(title="Actual Spread vs Walk-Forward Fair Value (True OOS)", yaxis_tickformat=".2%", height=400)
            st.plotly_chart(fig)

    # ---- Macro Residual ----
    st.markdown("#### Macro Residual")

    fig_res = go.Figure()
    fig_res.add_trace(go.Scatter(x=result_df["closeDate"], y=result_df["macro_residual"], mode="lines", name="Macro Residual"))
    fig_res.add_hline(y=0, line_color="lightgray", line_width=0.5)
    if model_type == "Static OLS (train/test split)":
        fig_res.add_vline(x=str(pd.to_datetime(train_end).date()), line_dash="dash", line_color="orange")
    fig_res.update_layout(title="Macro Residual (Spread − Fair Value)", yaxis_tickformat=".2%", height=350)
    st.plotly_chart(fig_res)

    st.markdown("#### Cointegration Test (Engle-Granger)")
    st.markdown("If the macro residual is stationary, the spread and macro drivers are **cointegrated** - mean reversion is structurally justified.")

    resid_stats = stationarity_table(result_df["macro_residual"].dropna(), name="Macro Residual")
    st.dataframe(resid_stats, hide_index=True)

    eg_merged = spread_df.merge(features_df, on="closeDate", how="inner").dropna()
    eg_train = eg_merged[eg_merged["closeDate"] <= pd.to_datetime(train_end)]

    if len(eg_train) > 100:
        eg_result = run_engle_granger(eg_train["spread"], eg_train[feature_cols])
        if eg_result.get("fallback"):
            st.caption("Fallback mode: using ADF on OLS residuals. Interpret as an approximate residual-stationarity check, not a formal MacKinnon cointegration p-value.")
        col_stat, col_verdict = st.columns([0.5, 0.5])
        with col_stat:
            c1, c2 = st.columns(2)
            with c1:
                st.metric("EG Test Statistic", f"{eg_result['stat']:.4f}")
            with c2:
                st.metric("p-value", f"{eg_result['pvalue']:.4f}")
        with col_verdict:
            if eg_result["pvalue"] < 0.05:
                st.markdown("Evidence of cointegration at 5% - mean reversion of the macro residual is statistically supported.")
            elif eg_result["pvalue"] < 0.10:
                st.markdown("Weak evidence of cointegration (10% level). Proceed with caution.")
            else:
                st.markdown("No evidence of cointegration. Mean reversion may be spurious.")

    # =================== Advanced Validation (Expandable) ===================
    st.divider()
    st.subheader("Advanced Validation")

    with st.expander("📊 Regularization Comparison (OLS vs Ridge vs Lasso vs ElasticNet)", expanded=False):
        st.markdown("Compare model performance and feature selection across regularization methods.")
        with st.spinner("Comparing models..."):
            reg_comparison = compare_regularization_models(
                spread_df, features_df, feature_cols, train_end=train_end,
            )
        display_reg = reg_comparison.copy()
        display_reg["R² Train"] = display_reg["R² Train"].map(lambda x: f"{x:.4f}")
        display_reg["R² Test (OOS)"] = display_reg["R² Test (OOS)"].map(lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A")
        st.dataframe(display_reg, hide_index=True)
        st.caption("Lasso performs automatic feature selection (Non-zero Coefs < Total). Ridge and ElasticNet shrink but retain all features.")

    with st.expander("🔄 Rolling Parameter Stability (HAC Standard Errors)", expanded=False):
        st.markdown("""
        Rolling OLS with **Newey-West (HAC) standard errors** shows whether macro betas are stable over time.
        Unstable betas suggest regime-dependent relationships — a key concern for any macro quant model.
        """)
        with st.spinner("Computing rolling OLS stability..."):
            stability_df = compute_rolling_ols_stability(
                spread_df, features_df, feature_cols, window=504, step=21,
            )
        if len(stability_df) > 0:
            beta_cols_stab = [c for c in stability_df.columns if c.startswith("beta_")]
            tstat_cols_stab = [c for c in stability_df.columns if c.startswith("tstat_")]

            fig_stab = go.Figure()
            for col in beta_cols_stab:
                fig_stab.add_trace(go.Scatter(
                    x=stability_df["closeDate"], y=stability_df[col],
                    mode="lines", name=col.replace("beta_", ""),
                ))
            fig_stab.add_hline(y=0, line_color="lightgray", line_width=0.5)
            fig_stab.update_layout(title="Rolling OLS Betas (HAC, 504d window)", height=400)
            st.plotly_chart(fig_stab)

            fig_tstat = go.Figure()
            for col in tstat_cols_stab:
                fig_tstat.add_trace(go.Scatter(
                    x=stability_df["closeDate"], y=stability_df[col],
                    mode="lines", name=col.replace("tstat_", ""),
                ))
            fig_tstat.add_hline(y=1.96, line_dash="dash", line_color="red", annotation_text="t=1.96")
            fig_tstat.add_hline(y=-1.96, line_dash="dash", line_color="red", annotation_text="t=-1.96")
            fig_tstat.add_hline(y=0, line_color="lightgray", line_width=0.5)
            fig_tstat.update_layout(title="Rolling t-Statistics (significance bands at ±1.96)", height=400)
            st.plotly_chart(fig_tstat)

            fig_r2_roll = go.Figure()
            fig_r2_roll.add_trace(go.Scatter(x=stability_df["closeDate"], y=stability_df["r2"], mode="lines", name="Rolling R²"))
            fig_r2_roll.update_layout(title="Rolling R² (OLS, 504d window)", height=300)
            st.plotly_chart(fig_r2_roll)

    with st.expander("🔬 Structural Break Test (Zivot-Andrews)", expanded=False):
        st.markdown("""
        Tests whether the spread series contains a **structural break** that changes its statistical properties.
        This is critical because forward rate spreads in Brazil are known to undergo regime changes
        (e.g., the 2016 easing cycle, 2020 pandemic shock).
        """)
        spread_series = spread_df.set_index("closeDate")["spread"].dropna()
        with st.spinner("Running Zivot-Andrews test..."):
            za_result = run_zivot_andrews(spread_series)

        if not np.isnan(za_result.get("statistic", np.nan)):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("ZA Statistic", f"{za_result['statistic']:.4f}")
            with c2:
                st.metric("p-value", f"{za_result['p_value']:.4f}")
            with c3:
                if za_result.get("break_index") is not None:
                    break_date = spread_series.index[za_result["break_index"]]
                    st.metric("Break Date", str(break_date.date()))

            if za_result["p_value"] < 0.05:
                st.success(f"Structural break detected at {break_date.date()}. Consider sub-period calibration or regime conditioning.")
            else:
                st.info("No significant structural break detected at 5% level.")
        else:
            st.warning(f"Zivot-Andrews test failed: {za_result.get('error', 'unknown')}")

    with st.expander("📈 Time-Series Cross-Validation (K-Fold)", expanded=False):
        st.markdown("""
        Expanding-window time-series CV with 5 folds. Unlike simple train/test, this shows
        how model performance evolves as training data grows — the gold standard for temporal validation.
        """)
        with st.spinner("Running 5-fold time-series CV..."):
            cv_results = time_series_cv(
                spread_df, features_df, feature_cols, n_splits=5, alpha=1.0,
            )

        if len(cv_results) > 0:
            display_cv = cv_results.copy()
            display_cv["train_end"] = display_cv["train_end"].dt.date
            display_cv["test_start"] = display_cv["test_start"].dt.date
            display_cv["test_end"] = display_cv["test_end"].dt.date
            display_cv["r2_train"] = display_cv["r2_train"].map(lambda x: f"{x:.4f}")
            display_cv["r2_test"] = display_cv["r2_test"].map(lambda x: f"{x:.4f}")
            st.dataframe(display_cv, hide_index=True)

            avg_cv_r2 = cv_results["r2_test"].mean()
            std_cv_r2 = cv_results["r2_test"].std()
            st.metric("Mean OOS R²", f"{avg_cv_r2:.4f} ± {std_cv_r2:.4f}")

    return result_df