import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from data_loader import load_json, build_curve_df, spread_fra
from functions import run_pca, add_rolling_zscore   #PCA
from functions import stationarity_table            #Stationarity
from functions import compute_residual_spread       #residual
#from functions import run_regime_model              #regime clustering

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

# ---------- Stationarity ----------

stats_table = stationarity_table(spread["spread"], name="1Y1Y - 5Y5Y Spread")
startdate_spread = spread["closeDate"].min().date()
enddate_spread = spread["closeDate"].max().date()
st.subheader(f"Spread Stationarity Analysis ({startdate_spread} to {enddate_spread})")
st.dataframe(stats_table, width="stretch")
st.caption("The raw spread shows mixed evidence of stationarity. ADF suggests stationarity, while KPSS rejects level stationarity. The spread may be mean-reverting, but not cleanly stationary over the full sample.")


# ---------- PCA ----------

curve = build_curve_df(spot1y, spot2y, spot5y, spot10y)

st.subheader("PCA Analysis")

left_panel, middle_panel, right_panel = st.columns([0.4, 0.3, 0.3])
with left_panel:
    mode = st.radio("Select analysis period",["Predefined Periods", "User defined Dates"],horizontal=True)
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

curve_pca = curve[(curve["closeDate"].dt.date >= start_date) & (curve["closeDate"].dt.date <= end_date)].copy()


left_panel, right_panel = st.columns([0.3, 0.7])

if len(curve_pca.dropna()) < 60:
    st.warning("Not enough data in selected window for PCA. Select a bigger period.")
    st.stop()

pca_df, loadings, explained = run_pca(curve_pca)
pca_df = add_rolling_zscore(pca_df, cols=["Level", "Slope", "Curvature"], windows=[60, 120, 252])

with left_panel:
    st.dataframe(loadings, width="stretch")

    explained_display = explained.copy()
    explained_display["Explained Variance"] = explained_display["Explained Variance"].map(lambda x: f"{x:.2%}")
    st.dataframe(explained_display, width="stretch")

with right_panel:
    fig_loadings = go.Figure()

    fig_loadings.add_trace(go.Scatter(x=loadings.index, y=loadings["Level"], name="Level"))
    fig_loadings.add_trace(go.Scatter(x=loadings.index, y=loadings["Slope"], name="Slope"))
    fig_loadings.add_trace(go.Scatter(x=loadings.index, y=loadings["Curvature"], name="Curvature"))

    fig_loadings.update_layout(title="PCA Weights by Vertex",xaxis_title="Vertex",yaxis_title="Loading",barmode="group")

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


# ---------- Stationarity ----------

filtered_spread = spread[(spread["closeDate"].dt.date >= start_date) & (spread["closeDate"].dt.date <= end_date)].copy()

stats_table = stationarity_table(filtered_spread["spread"], name="1Y1Y - 5Y5Y Spread")

st.subheader(f"Spread Stationarity Analysis ({start_date} to {end_date})")
st.dataframe(stats_table, width="stretch")

# ---------- Residual ----------

residual_df, residual_betas, residual_stats = compute_residual_spread(filtered_spread, pca_df)
st.subheader(f"Factor-Adjusted Spread (Residual) - {start_date} to {end_date}")
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
    st.dataframe(residual_betas, width="stretch")
    fit_table = pd.DataFrame({"Metric": ["R-squared", "Adj. R-squared"],"Value": [residual_stats["r2"], residual_stats["adj_r2"]]})
    st.dataframe(fit_table, width="stretch")

with right_res:
    fig_res = go.Figure()
    fig_res.add_trace(go.Scatter(x=residual_df["closeDate"],y=residual_df["spread"],mode="lines",name="Actual Spread"))
    fig_res.add_trace(go.Scatter(x=residual_df["closeDate"],y=residual_df["fair_spread"],mode="lines",name="Fair Spread"))
    fig_res.update_layout(title="Actual vs Fair Spread",xaxis_title="Date",yaxis_title="Spread", yaxis_tickformat=".2%")
    st.plotly_chart(fig_res, width="stretch")


fig_residual = go.Figure()
fig_residual.add_trace(go.Scatter(x=residual_df["closeDate"],y=residual_df["residual_spread"],mode="lines",name="Residual Spread"))
fig_residual.update_layout(title="Residual Spread Time Series",xaxis_title="Date",yaxis_title="Residual Spread")
st.plotly_chart(fig_residual, width="stretch")

residual_stats_table = stationarity_table(residual_df["residual_spread"],name="1Y1Y - 5Y5Y Residual Spread")

st.subheader(f"Residual Spread Stationarity Analysis ({start_date} to {end_date})")
st.dataframe(residual_stats_table, width="stretch")

st.markdown("The residual helps us answer if the 1y1y-5y5y is high or low given today’s level, slope, and curvature. To support this analysis we will add a regime clustering.")


# ---------- Regime Clustering ----------

regime_df = pca_df[["closeDate", "Level", "Slope", "Curvature"]].copy()
regime_df = regime_df.merge(residual_df[["closeDate", "residual_spread"]],on="closeDate",how="inner")
regime_df = regime_df.dropna().reset_index(drop=True)

st.subheader(f"Regime Analysis - ({start_date} to {end_date})")
left_regime, right_regime = st.columns([0.5, 0.5])

with left_regime:
    clustering_option = st.radio("Choose the model:",options=["gmm","kmeans"],horizontal=True)

with right_regime:
    n_regimes = st.selectbox("Number of regimes", [2, 3, 4, 5], index=1)

# regime_df, centers, model = run_regime_model(regime_df,["Level", "Slope", "Curvature", "residual_spread"],model_type=clustering_option,n_regimes=n_regimes)

# st.subheader("Regime Timeline")
# fig_regime = go.Figure()
# fig_regime.add_trace(go.Scatter(x=regime_df["closeDate"],y=regime_df["regime"],mode="markers",marker=dict(size=6),name="Regime"))
# fig_regime.update_layout(title="Regime Classification Through Time",xaxis_title="Date",yaxis_title="Regime")
# st.plotly_chart(fig_regime, width="stretch")

# cur_reg = int(regime_df["regime"].dropna().iloc[-1])
# st.subheader(f"Current Regime: {cur_reg}")

# st.subheader("Regime Summary")
# summary = regime_df.groupby(["is_train", "regime"]).agg(Count=("regime", "size"),Avg_Residual=("residual_spread", "mean"),Std_Residual=("residual_spread", "std")).reset_index()
# st.dataframe(summary, width="stretch")

# st.subheader("Regime Centers")
# st.dataframe(centers, width="stretch")