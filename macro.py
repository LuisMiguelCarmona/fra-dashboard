from macroAnalysis import (
    build_macro_features,
    compute_vif,
    run_granger_tests,
    run_engle_granger,
    run_johansen_test,
    compute_macro_residual_ols,
    compute_macro_residual_rolling_ridge,
    compare_regularization_models,
)
from validation import (
    walk_forward_backtest,
    stability_selection,
    run_zivot_andrews,
    run_chow_test,
    run_bai_perron,
    run_ljung_box,
    compute_cusum,
    compute_cusumsq,
    diebold_mariano_test,
    compute_rolling_ols_stability,
)
from stationarity import stationarity_table
from config import TRAIN_END
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

    if "vix" in macro_df.columns:
        corr_df = corr_df.merge(macro_df[["closeDate", "vix"]], on="closeDate", how="inner")
        corr_cols.append("vix")

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

def render_model(macro_df, spread_df, inflation_curve, train_end=TRAIN_END):
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
    macro1, macro2, macro3, macro4, macro5 = st.columns(5)
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

    with macro5:
        st.markdown("##### Vix")
        if "vix" in all_available and st.checkbox("Vix", value=False, key="m_vix"):
            selected_levels.append("vix")

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

    # ---- Johansen Multivariate Cointegration ----
    st.markdown("#### Johansen Cointegration Test (Multivariate)")
    st.markdown("""
    Unlike Engle-Granger (bivariate), **Johansen** tests for cointegration among *all* variables simultaneously.
    It determines how many cointegrating vectors exist — i.e., how many independent long-run equilibria
    link the spread to macro factors.
    """)

    if len(eg_train) > 100:
        with st.spinner("Running Johansen test..."):
            joh_result = run_johansen_test(
                eg_train.set_index(eg_train.index)["spread"],
                eg_train.set_index(eg_train.index),
                feature_cols,
            )

        if "error" in joh_result:
            st.warning(f"Johansen test failed: {joh_result['error']}")
        else:
            col_j1, col_j2 = st.columns([0.6, 0.4])
            with col_j1:
                st.dataframe(joh_result["table"], hide_index=True)
            with col_j2:
                st.metric("Cointegrating Vectors (Trace)", joh_result["n_coint_trace"])
                st.metric("Cointegrating Vectors (Eigenvalue)", joh_result["n_coint_eigen"])

                if joh_result["n_coint_trace"] > 0:
                    st.success(f"Johansen confirms {joh_result['n_coint_trace']} cointegrating relationship(s). Long-run macro equilibrium exists.")
                else:
                    st.warning("Johansen finds no cointegrating vectors. Exercise caution with mean-reversion assumptions.")

    # =================== Advanced Validation (Expandable) ===================
    st.divider()
    st.subheader("Advanced Validation")

    st.subheader("Stability Selection — Robust Feature Selection (Meinshausen & Bühlmann, 2010)")
    st.markdown("""
    Runs **LASSO on 100 random subsamples** across a grid of regularization strengths.
    Features selected in ≥60% of runs are considered **stable** — robust to data perturbation.
    This controls the expected number of false positives and is far more reliable than single-shot LASSO.
    """)
    with st.spinner("Running Stability Selection (100 subsamples × 20 alphas)..."):
        ss_result = stability_selection(
            spread_df, features_df, feature_cols, train_end=train_end,
        )

    col_ss1, col_ss2 = st.columns([0.5, 0.5])
    with col_ss1:
        display_ss = ss_result.copy()
        display_ss["Selection Probability"] = display_ss["Selection Probability"].map(lambda x: f"{x:.3f}")
        display_ss["Selected"] = display_ss["Selected"].map(lambda x: "✓" if x else "")
        st.dataframe(display_ss, hide_index=True)
    with col_ss2:
        fig_ss = go.Figure()
        fig_ss.add_trace(go.Bar(
            x=ss_result["Feature"], y=ss_result["Selection Probability"].astype(float),
            marker_color=["green" if s else "lightgray" for s in ss_result["Selected"]],
        ))
        fig_ss.add_hline(y=0.6, line_dash="dash", line_color="red", annotation_text="Threshold (0.6)")
        fig_ss.update_layout(title="Feature Selection Probability", yaxis_title="P(selected)", height=350)
        st.plotly_chart(fig_ss)

    n_stable = ss_result["Selected"].sum()
    stable_features = ss_result[ss_result["Selected"]]["Feature"].tolist()
    st.caption(f"{n_stable} of {len(feature_cols)} features are stable: {', '.join(stable_features) if stable_features else 'none'}")

    st.subheader("Regularization Comparison — Rolling Walk-Forward (OLS vs Ridge vs Lasso vs ElasticNet)")
    st.markdown("""
    All models compared in the **same rolling walk-forward framework** (504d window).
    At each day t, every model trains on [t-504 : t] and predicts t — true out-of-sample.
    A static train/test split is meaningless for macro relationships that drift.
    """)
    with st.spinner("Running rolling comparison across 5 models..."):
        reg_comparison, reg_errors_df = compare_regularization_models(
            spread_df, features_df, feature_cols,
        )
    display_reg = reg_comparison.copy()
    for col in ["OOS R²", "OOS RMSE", "OOS MAE"]:
        display_reg[col] = display_reg[col].map(lambda x: f"{x:.6f}" if not np.isnan(x) else "N/A")
    st.dataframe(display_reg, hide_index=True)

    # Plot rolling RMSE per model
    error_cols = [c for c in reg_errors_df.columns if c.startswith("error_")]
    if error_cols:
        fig_rmse = go.Figure()
        for col in error_cols:
            model_name = col.replace("error_", "")
            rolling_rmse = reg_errors_df[col].pow(2).rolling(252).mean().pow(0.5)
            fig_rmse.add_trace(go.Scatter(
                x=reg_errors_df["closeDate"], y=rolling_rmse,
                mode="lines", name=model_name,
            ))
        fig_rmse.update_layout(title="Rolling 252d RMSE by Model (lower = better)", yaxis_title="RMSE", height=350)
        st.plotly_chart(fig_rmse)

    st.subheader("Rolling Parameter Stability (HAC Standard Errors)")
    st.markdown("Rolling OLS with **Newey-West (HAC)** standard errors. Unstable betas suggest regime-dependent relationships.")
    with st.spinner("Computing rolling OLS stability..."):
        stability_df = compute_rolling_ols_stability(
            spread_df, features_df, feature_cols, window=504, step=21,
        )
    if len(stability_df) > 0:
        beta_cols_stab = [c for c in stability_df.columns if c.startswith("beta_")]
        tstat_cols_stab = [c for c in stability_df.columns if c.startswith("tstat_")]

        fig_stab = go.Figure()
        for col in beta_cols_stab:
            fig_stab.add_trace(go.Scatter(x=stability_df["closeDate"], y=stability_df[col], mode="lines", name=col.replace("beta_", "")))
        fig_stab.add_hline(y=0, line_color="lightgray", line_width=0.5)
        fig_stab.update_layout(title="Rolling OLS Betas (HAC, 504d window)", height=400)
        st.plotly_chart(fig_stab)

        fig_tstat = go.Figure()
        for col in tstat_cols_stab:
            fig_tstat.add_trace(go.Scatter(x=stability_df["closeDate"], y=stability_df[col], mode="lines", name=col.replace("tstat_", "")))
        fig_tstat.add_hline(y=1.96, line_dash="dash", line_color="red", annotation_text="t=1.96")
        fig_tstat.add_hline(y=-1.96, line_dash="dash", line_color="red", annotation_text="t=-1.96")
        fig_tstat.add_hline(y=0, line_color="lightgray", line_width=0.5)
        fig_tstat.update_layout(title="Rolling t-Statistics (significance bands ±1.96)", height=400)
        st.plotly_chart(fig_tstat)

        fig_r2_roll = go.Figure()
        fig_r2_roll.add_trace(go.Scatter(x=stability_df["closeDate"], y=stability_df["r2"], mode="lines", name="Rolling R²"))
        fig_r2_roll.update_layout(title="Rolling R² (OLS, 504d window)", height=300)
        st.plotly_chart(fig_r2_roll)

    st.subheader("Structural Break Tests (Zivot-Andrews + Chow + Bai-Perron)")
    spread_series = spread_df.set_index("closeDate")["spread"].dropna()

    st.markdown("##### Zivot-Andrews (single endogenous break)")
    with st.spinner("Running Zivot-Andrews..."):
        za_result = run_zivot_andrews(spread_series)

    za_break_date = None
    if not np.isnan(za_result.get("statistic", np.nan)):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("ZA Statistic", f"{za_result['statistic']:.4f}")
        with c2:
            st.metric("p-value", f"{za_result['p_value']:.4f}")
        with c3:
            if za_result.get("break_index") is not None:
                za_break_date = spread_series.index[za_result["break_index"]]
                st.metric("Break Date", str(za_break_date.date()))
        if za_result["p_value"] < 0.05:
            st.success(f"Structural break detected at {za_break_date.date()}.")
        else:
            st.info("No significant structural break at 5% level.")
    else:
        st.warning(f"Zivot-Andrews failed: {za_result.get('error', 'unknown')}")

    st.divider()
    st.markdown("##### Chow Test (known break date)")
    st.markdown("Tests whether the **macro regression coefficients** are structurally different before vs after a given date. Uses the Zivot-Andrews break date by default, or pick your own.")

    chow_merged = spread_df.merge(features_df, on="closeDate", how="inner").dropna()
    chow_merged = chow_merged[chow_merged["closeDate"] <= pd.to_datetime(train_end)]

    if len(chow_merged) > 50:
        default_date = za_break_date.date() if za_break_date is not None else pd.to_datetime(train_end).date()
        chow_date = st.date_input(
            "Break date for Chow test",
            value=default_date,
            min_value=chow_merged["closeDate"].min().date(),
            max_value=chow_merged["closeDate"].max().date(),
            key="chow_date_input",
        )

        chow_result = run_chow_test(
            chow_merged["spread"].values,
            chow_merged[feature_cols].values,
            break_date=chow_date,
            dates=chow_merged["closeDate"],
        )

        if "error" in chow_result:
            st.warning(chow_result["error"])
        else:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("F-statistic", f"{chow_result['f_stat']:.4f}")
            with c2:
                st.metric("p-value", f"{chow_result['p_value']:.4f}")
            with c3:
                st.metric("Pre-break obs", chow_result["n_pre"])
            with c4:
                st.metric("Post-break obs", chow_result["n_post"])

            if chow_result["p_value"] < 0.05:
                st.warning(f"Chow test rejects stability at 5% — macro betas are structurally different before and after {chow_date}.")
            else:
                st.success(f"Chow test does not reject stability — no evidence that coefficients changed at {chow_date}.")

    st.divider()
    st.markdown("##### Bai-Perron (multiple structural breaks, BIC selection)")
    st.markdown("Finds **optimal breakpoints** via dynamic programming, uses BIC to select how many breaks are statistically justified.")
    with st.spinner("Running Bai-Perron..."):
        bp_result = run_bai_perron(spread_series.values, max_breaks=5, min_segment=60)

    if "error" not in bp_result:
        col_bp1, col_bp2 = st.columns([0.4, 0.6])
        with col_bp1:
            st.metric("Optimal Breaks (BIC)", bp_result["n_breaks"])
            st.markdown("**BIC Model Selection**")
            display_bic = bp_result["bic_table"].copy()
            display_bic["ssr"] = display_bic["ssr"].map(lambda x: f"{x:.6f}")
            display_bic["bic"] = display_bic["bic"].map(lambda x: f"{x:.2f}")
            st.dataframe(display_bic, hide_index=True)

        with col_bp2:
            if bp_result["n_breaks"] > 0:
                st.markdown("**Segment Statistics**")
                seg_display = bp_result["segments"].copy()
                # Map indices to dates
                for _, row in seg_display.iterrows():
                    start_idx = int(row["Start Idx"])
                    end_idx = int(row["End Idx"])
                    if start_idx < len(spread_series) and end_idx < len(spread_series):
                        row_start_date = spread_series.index[start_idx]
                        row_end_date = spread_series.index[end_idx]
                        seg_display.loc[seg_display["Segment"] == row["Segment"], "Start"] = str(row_start_date.date())
                        seg_display.loc[seg_display["Segment"] == row["Segment"], "End"] = str(row_end_date.date())
                seg_display["Mean"] = seg_display["Mean"].map(lambda x: f"{x:.4f}")
                seg_display["Std"] = seg_display["Std"].map(lambda x: f"{x:.4f}")
                st.dataframe(seg_display[["Segment", "Start", "End", "Obs", "Mean", "Std"]], hide_index=True)

                # Plot with break lines
                fig_bp = go.Figure()
                fig_bp.add_trace(go.Scatter(x=spread_series.index, y=spread_series.values, mode="lines", name="Spread"))
                for bi in bp_result["break_indices"]:
                    if bi < len(spread_series):
                        fig_bp.add_vline(x=spread_series.index[bi], line_dash="dash", line_color="red")
                fig_bp.update_layout(title="Spread with Bai-Perron Break Points", yaxis_tickformat=".2%", height=350)
                st.plotly_chart(fig_bp)
            else:
                st.info("No structural breaks detected — the series is best described by a single regime.")
    else:
        st.warning(f"Bai-Perron failed: {bp_result.get('error', 'unknown')}")

    st.subheader("Residual Diagnostics (Ljung-Box + CUSUM + CUSUMSQ)")
    resid_series = result_df["macro_residual"].dropna()

    st.markdown("##### Ljung-Box Test — Residual Autocorrelation")
    st.markdown("H₀: no autocorrelation in residuals up to lag *k*. Rejection means the model leaves exploitable serial structure.")
    lb_result = run_ljung_box(resid_series, lags=[5, 10, 20, 40])
    lb_display = lb_result.copy()
    lb_display["LB Statistic"] = lb_display["LB Statistic"].map(lambda x: f"{x:.2f}")
    lb_display["p-value"] = lb_display["p-value"].map(lambda x: f"{x:.4f}")
    st.dataframe(lb_display, hide_index=True)

    n_sig = (lb_result["Significant (5%)"] == "Y").sum()
    if n_sig > 0:
        st.warning(f"Autocorrelation detected at {n_sig} lag(s). Residuals still contain serial structure — consider adding lagged terms or AR components.")
    else:
        st.success("No significant residual autocorrelation. The model captures the predictable component well.")

    st.divider()
    st.markdown("##### CUSUM — Parameter Stability")
    st.markdown("Tracks cumulative recursive residuals. Breach of 5% bands signals **parameter instability** (structural change in the macro relationship).")

    resid_merged = spread_df.merge(features_df, on="closeDate", how="inner").dropna()
    resid_train = resid_merged[resid_merged["closeDate"] <= pd.to_datetime(train_end)]

    if len(resid_train) > 50:
        cusum_dates = resid_train["closeDate"].values
        cusum_result = compute_cusum(
            resid_train["spread"].values,
            resid_train[feature_cols].values,
            dates=cusum_dates,
        )

        fig_cusum = go.Figure()
        fig_cusum.add_trace(go.Scatter(x=cusum_result["dates"], y=cusum_result["cusum"], mode="lines", name="CUSUM", line=dict(color="blue")))
        fig_cusum.add_trace(go.Scatter(x=cusum_result["dates"], y=cusum_result["upper_band"], mode="lines", name="Upper 5%", line=dict(color="red", dash="dash")))
        fig_cusum.add_trace(go.Scatter(x=cusum_result["dates"], y=cusum_result["lower_band"], mode="lines", name="Lower 5%", line=dict(color="red", dash="dash")))
        fig_cusum.update_layout(title="CUSUM Test (Brown, Durbin & Evans, 1975)", height=350)
        st.plotly_chart(fig_cusum)

        if cusum_result["breach"]:
            st.warning("CUSUM breaches 5% significance bands — evidence of parameter instability.")
        else:
            st.success("CUSUM within 5% bands — parameters appear stable over the training period.")

        st.divider()
        st.markdown("##### CUSUMSQ — Variance Stability")
        st.markdown("Tracks cumulative squared recursive residuals. Deviation from the expected linear path indicates **heteroskedasticity or variance breaks**.")

        cusumsq_result = compute_cusumsq(
            resid_train["spread"].values,
            resid_train[feature_cols].values,
            dates=cusum_dates,
        )

        fig_csq = go.Figure()
        fig_csq.add_trace(go.Scatter(x=cusumsq_result["dates"], y=cusumsq_result["cusumsq"], mode="lines", name="CUSUMSQ", line=dict(color="blue")))
        fig_csq.add_trace(go.Scatter(x=cusumsq_result["dates"], y=cusumsq_result["expected"], mode="lines", name="Expected", line=dict(color="gray", dash="dot")))
        fig_csq.add_trace(go.Scatter(x=cusumsq_result["dates"], y=cusumsq_result["upper_band"], mode="lines", name="Upper 5%", line=dict(color="red", dash="dash")))
        fig_csq.add_trace(go.Scatter(x=cusumsq_result["dates"], y=cusumsq_result["lower_band"], mode="lines", name="Lower 5%", line=dict(color="red", dash="dash")))
        fig_csq.update_layout(title="CUSUMSQ Test — Variance Stability", height=350)
        st.plotly_chart(fig_csq)

        if cusumsq_result["breach"]:
            st.warning("CUSUMSQ breaches 5% bands — evidence of variance instability (heteroskedasticity).")
        else:
            st.success("CUSUMSQ within 5% bands — residual variance appears stable.")

    st.subheader("Diebold-Mariano Test — Model Comparison")
    st.markdown("""
    **Diebold-Mariano (1995)** tests whether two models have equal predictive accuracy.
    Both models are evaluated in the **same rolling walk-forward framework** — no static train/test.
    Uses **Harvey correction** for small samples.
    """)

    col_dm1, col_dm2 = st.columns(2)
    with col_dm1:
        dm_model_1 = st.selectbox("Model 1", ["OLS", "Ridge (α=1)"], index=0, key="dm_m1")
    with col_dm2:
        dm_model_2 = st.selectbox("Model 2", ["OLS", "Ridge (α=1)", "Ridge (α=10)", "Lasso (α=0.001)"], index=1, key="dm_m2")

    if st.button("Run Diebold-Mariano", key="dm_run"):
        with st.spinner("Computing rolling walk-forward errors for both models..."):
            _, dm_error_df = compare_regularization_models(
                spread_df, features_df, feature_cols,
            )

        col_e1 = f"error_{dm_model_1}"
        col_e2 = f"error_{dm_model_2}"

        if col_e1 in dm_error_df.columns and col_e2 in dm_error_df.columns:
            valid_mask = dm_error_df[col_e1].notna() & dm_error_df[col_e2].notna()
            e1 = dm_error_df.loc[valid_mask, col_e1].values
            e2 = dm_error_df.loc[valid_mask, col_e2].values

            if len(e1) > 50:
                dm_result = diebold_mariano_test(e1, e2, horizon=1, loss="mse")

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("DM Statistic", f"{dm_result['dm_stat']:.4f}")
                with c2:
                    st.metric("p-value", f"{dm_result['p_value']:.4f}")
                with c3:
                    better_label = dm_result["better_model"].replace("Model 1", dm_model_1).replace("Model 2", dm_model_2)
                    st.metric("Better Model", better_label)

                if dm_result["p_value"] < 0.05:
                    st.success(f"**{better_label}** is statistically more accurate at 5% (rolling walk-forward, {len(e1)} OOS obs).")
                else:
                    st.info(f"No significant difference between {dm_model_1} and {dm_model_2} ({len(e1)} OOS obs).")
            else:
                st.warning("Insufficient overlapping observations.")
        else:
            st.warning(f"Model names not found in error columns. Available: {[c for c in dm_error_df.columns if c.startswith('error_')]}")

    # st.subheader("Time-Series Cross-Validation (K-Fold)")
    # st.markdown("Expanding-window time-series CV with 5 folds — the gold standard for temporal validation.")
    # with st.spinner("Running 5-fold time-series CV..."):
    #     cv_results = time_series_cv(spread_df, features_df, feature_cols, n_splits=5, alpha=1.0)

    # if len(cv_results) > 0:
    #     display_cv = cv_results.copy()
    #     display_cv["train_end"] = display_cv["train_end"].dt.date
    #     display_cv["test_start"] = display_cv["test_start"].dt.date
    #     display_cv["test_end"] = display_cv["test_end"].dt.date
    #     display_cv["r2_train"] = display_cv["r2_train"].map(lambda x: f"{x:.4f}")
    #     display_cv["r2_test"] = display_cv["r2_test"].map(lambda x: f"{x:.4f}")
    #     st.dataframe(display_cv, hide_index=True)

    #     avg_cv_r2 = cv_results["r2_test"].mean()
    #     std_cv_r2 = cv_results["r2_test"].std()
    #     st.metric("Mean OOS R²", f"{avg_cv_r2:.4f} ± {std_cv_r2:.4f}")

    return result_df