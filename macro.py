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
        st.caption("CDS denominated in USD — note inherent correlation with USDBRL.")

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
        st.markdown("#### US 10Y Treasury (Global Duration Driver)")

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
    st.markdown("CDS in USD has inherent correlation with USDBRL.")

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