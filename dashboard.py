# ------------------- IMPORTS -------------------
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import json

# ------------------- Basic Titles -------------------
st.set_page_config(page_title="FRA Spreads Dashboard", layout="wide")

st.title("FRA Spreads Dashboard - Case Study")

# ------------------- FUNCTIONS -------------------
def read_json(raw):
    rows = []
    for bucket in raw:
        bucket_name = bucket.get("bucketName")
        for point in bucket.get("curve", []):
            rows.append({
                "bucketName": bucket_name,
                "closeDate": point.get("closeDate"),
                "nominalRateValue": point.get("nominalRateValue")})
    df = pd.DataFrame(rows)
    df["closeDate"] = pd.to_datetime(df["closeDate"])
    return df

def load_json(path):
    with open(path, "r") as f:
        raw = json.load(f)
    return read_json(raw)

# ------------------- Loading Data -------------------

spot1y      = load_json(r"data/Spot1y.json")
spot2y      = load_json(r"data/Spot2y.json")
spot5y      = load_json(r"data/Spot5y.json")
spot10y     = load_json(r"data/Spot10y.json")
spread1y1y  = load_json(r"data/1y1y.json")
spread5y5y  = load_json(r"data/5y5y.json")

# ------------------- Dashboard Structure -------------------

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


spread = spread1y1y.copy()
spread["spread"] = spread1y1y["nominalRateValue"] - spread5y5y["nominalRateValue"]

with col2:
    fig_right = go.Figure()
    if show_1y1y: fig_right.add_trace(go.Scatter(x=spread1y1y["closeDate"], y=spread1y1y["nominalRateValue"], mode="lines", name="1Y1Y"))
    if show_5y5y: fig_right.add_trace(go.Scatter(x=spread5y5y["closeDate"], y=spread5y5y["nominalRateValue"], mode="lines", name="5Y5Y"))
    if show_spread: fig_right.add_trace(go.Scatter(x=spread["closeDate"], y=spread["spread"], mode="lines", name="Spread"))

    fig_right.update_layout(title="FRA Curve and Spread", yaxis_tickformat=".2%")
    st.plotly_chart(fig_right, width="stretch")


