import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from data_loader import load_json, build_curve_df, spread_fra
from functions import run_pca


# ------------------- Basic Titles -------------------
st.set_page_config(page_title="FRA Spreads Dashboard", layout="wide")
st.title("FRA Spreads Dashboard - Case Study")

# ------------------- Loading Data -------------------
spot1y      = load_json(r"data/Spot1y.json")
spot2y      = load_json(r"data/Spot2y.json")
spot5y      = load_json(r"data/Spot5y.json")
spot10y     = load_json(r"data/Spot10y.json")
spread1y1y  = load_json(r"data/1y1y.json")
spread5y5y  = load_json(r"data/5y5y.json")

# ------------------- Dashboard Structure -------------------

# ---------- Basic Plots ----------
control_left, control_right = st.columns(2)

with control_left:
    c1, c2, c3, c4 = st.columns(4)

    show_1y  = c1.checkbox("1Y", value=True)
    show_2y  = c2.checkbox("2Y", value=True)
    show_5y  = c3.checkbox("5Y", value=True)
    show_10y = c4.checkbox("10Y", value=True)

with control_right:
    c5, c6, c7 = st.columns(3)

    show_1y1y  = c5.checkbox("1Y1Y", value=True)
    show_5y5y  = c6.checkbox("5Y5Y", value=True)
    show_spread = c7.checkbox("Spread", value=True)

col1, col2 = st.columns(2)

with col1:
    fig_left = go.Figure()
    if show_1y: fig_left.add_trace(go.Scatter(x=spot1y["closeDate"], y=spot1y["nominalRateValue"], mode="lines", name="Spot 1Y"))
    if show_2y: fig_left.add_trace(go.Scatter(x=spot2y["closeDate"], y=spot2y["nominalRateValue"], mode="lines", name="Spot 2Y"))
    if show_5y: fig_left.add_trace(go.Scatter(x=spot5y["closeDate"], y=spot5y["nominalRateValue"], mode="lines", name="Spot 5Y"))
    if show_10y: fig_left.add_trace(go.Scatter(x=spot10y["closeDate"], y=spot10y["nominalRateValue"], mode="lines", name="Spot 10Y"))

    fig_left.update_layout(title="Spot Curve Evolution", yaxis_tickformat=".2%")
    st.plotly_chart(fig_left, width="stretch")

spread = spread_fra(spread1y1y,spread5y5y)

with col2:
    fig_right = go.Figure()
    if show_1y1y: fig_right.add_trace(go.Scatter(x=spread1y1y["closeDate"], y=spread1y1y["nominalRateValue"], mode="lines", name="1Y1Y"))
    if show_5y5y: fig_right.add_trace(go.Scatter(x=spread5y5y["closeDate"], y=spread5y5y["nominalRateValue"], mode="lines", name="5Y5Y"))
    if show_spread: fig_right.add_trace(go.Scatter(x=spread["closeDate"], y=spread["spread"], mode="lines", name="Spread"))

    fig_right.update_layout(title="FRA Curve and Spread", yaxis_tickformat=".2%")
    st.plotly_chart(fig_right, width="stretch")


# ---------- PCA ----------

curve = build_curve_df(spot1y, spot2y, spot5y, spot10y)

st.subheader("PCA Analysis")

left_panel, middle_panel, right_panel = st.columns([0.4, 0.3, 0.3])
with left_panel:
    mode = st.radio("Select analysis period",["User defined Dates", "Predefined Periods"],horizontal=True)
with middle_panel:
    min_date = curve["closeDate"].min().date()
    max_date = curve["closeDate"].max().date()
    if mode == "User defined Dates":
        d1, d2 = st.columns(2)

        with d1:
            start_date = st.date_input("Start date",value=min_date,min_value=min_date,max_value=max_date)

        with d2:
            end_date = st.date_input("End date",value=max_date,min_value=min_date,max_value=max_date)

    else:
        period = st.selectbox("Predefined period", ["3Y", "5Y", "10Y", "15Y", "Max Period"])

        end_date = max_date
        if period == "3Y": start_date = (pd.Timestamp(end_date) - pd.DateOffset(years=3)).date()
        elif period == "5Y": start_date = (pd.Timestamp(end_date) - pd.DateOffset(years=5)).date()
        elif period == "10Y": start_date = (pd.Timestamp(end_date) - pd.DateOffset(years=10)).date()
        elif period == "15Y": start_date = (pd.Timestamp(end_date) - pd.DateOffset(years=15)).date()
        else: start_date = min_date

        if start_date < min_date:
            start_date = min_date

curve_pca = curve[(curve["closeDate"].dt.date >= start_date) & (curve["closeDate"].dt.date <= end_date)].copy()


left_panel, right_panel = st.columns([0.3, 0.7])
pca_df, loadings, explained = run_pca(curve_pca)

with left_panel:
    st.markdown("#### PCA Loadings")
    st.dataframe(loadings, width="stretch")

    st.markdown("#### Explained Variance")
    explained_display = explained.copy()
    explained_display["Explained Variance"] = explained_display["Explained Variance"].map(lambda x: f"{x:.2%}")
    st.dataframe(explained_display, width="stretch")

with right_panel:
    fig_loadings = go.Figure()

    fig_loadings.add_trace(go.Scatter(x=loadings.index, y=loadings["Level"], name="Level"))
    fig_loadings.add_trace(go.Scatter(x=loadings.index, y=loadings["Slope"], name="Slope"))
    fig_loadings.add_trace(go.Scatter(x=loadings.index, y=loadings["Curvature"], name="Curvature"))

    fig_loadings.update_layout(title="PCA Weights by Vertice",xaxis_title="Vertice",yaxis_title="Loading",barmode="group"
    )

    st.plotly_chart(fig_loadings, width="stretch")



row_pc1, row_pc2, row_pc3 = st.columns(3)

with row_pc1:
    fig_pc1 = go.Figure()
    fig_pc1.add_trace(go.Scatter(x=pca_df["closeDate"], y=pca_df["Level_z"], mode="lines", name="Level"))
    fig_pc1.update_layout(title="Level Time Series", xaxis_title="Date", yaxis_title="Level")
    st.plotly_chart(fig_pc1, width="stretch")

with row_pc2:
    fig_pc2 = go.Figure()
    fig_pc2.add_trace(go.Scatter(x=pca_df["closeDate"], y=pca_df["Slope_z"], mode="lines", name="Slope"))
    fig_pc2.update_layout(title="Slope Time Series", xaxis_title="Date", yaxis_title="Slope")
    st.plotly_chart(fig_pc2, width="stretch")

with row_pc3:
    fig_pc3 = go.Figure()
    fig_pc3.add_trace(go.Scatter(x=pca_df["closeDate"], y=pca_df["Curvature_z"], mode="lines", name="Curvature"))
    fig_pc3.update_layout(title="Curvature Time Series", xaxis_title="Date", yaxis_title="Curvature")
    st.plotly_chart(fig_pc3, width="stretch")