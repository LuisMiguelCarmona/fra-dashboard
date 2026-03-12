import streamlit as st
import pandas as pd
import os

from data_loader import load_json, build_nominal_curve_df, build_inflation_curve_df, spread_fra, build_macro_df, load_copom_sentiment
from config import TRAIN_END
import curveEvolution
import stationarity
import pca
import macro
import macroTrading


st.set_page_config(page_title="FRA Spreads Dashboard", layout="wide")

MACRO_FACTORS = ['cds1y','cds2y','cds5y','cds10y','us10y','usdbrl','vix']


def load_all_data():
    spot1y  = load_json(r"data/Spot1y.json")
    spot2y  = load_json(r"data/Spot2y.json")
    spot5y  = load_json(r"data/Spot5y.json")
    spot10y = load_json(r"data/Spot10y.json")
    fra1y1y = load_json(r"data/1y1y.json")
    fra5y5y = load_json(r"data/5y5y.json")

    nominal_curve  = build_nominal_curve_df(spot1y, spot2y, spot5y, spot10y)
    inflation_curve = build_inflation_curve_df(spot1y, spot2y, spot5y, spot10y)
    spread_df      = spread_fra(fra1y1y, fra5y5y)

    macro_df = build_macro_df(MACRO_FACTORS)

    macro_df = macro_df.merge(inflation_curve,on="closeDate", how="left")

    try:
        copom_df = load_copom_sentiment(r"data/copom_regime_classification.json")
    except Exception as e:
        copom_df = None
        st.warning(f"Could not load COPOM sentiment file: {e}")



    return {
        "spot1y": spot1y,
        "spot2y": spot2y,
        "spot5y": spot5y,
        "spot10y": spot10y,
        "fra1y1y": fra1y1y,
        "fra5y5y": fra5y5y,
        "nominal_curve": nominal_curve,
        "inflation_curve": inflation_curve,
        "spread_df": spread_df,
        "macro_df": macro_df,
        "copom_df": copom_df
    }



def main():
    st.title("FRA Spreads Dashboard - Case Study")

    data = load_all_data()
    max_date = data['spot1y']['closeDate'].max().date()
    min_date = data['spot1y']['closeDate'].min().date()
    macro_cols = [c for c in data["macro_df"].columns if c != "closeDate"]
    with st.sidebar:
        st.markdown("### Data Summary")
        spread = data["spread_df"]
        st.caption(
            f"Period: {spread['closeDate'].min().date()} to {spread['closeDate'].max().date()}"
        )
        st.caption(f"Trading days: {len(spread):,}")

        if macro_cols:
            st.markdown("**Macro columns:**")
            for col in macro_cols:
                st.caption(f"  • {col}")

    curveEvolution.curvePlots(data['spot1y'],data['spot2y'],data['spot5y'],data['spot10y'],data['fra1y1y'],data['fra5y5y'],data['spread_df'])

    st.subheader(f'Spread Stationarity Analysis ({min_date} to {max_date})')
    st.dataframe(stationarity.stationarity_table(data['spread_df']['spread']),hide_index=True)
    st.markdown('The raw spread shows mixed evidence of stationarity. ADF suggests stationarity, while KPSS rejects level stationarity. The spread may be mean-reverting, but not cleanly stationary over the full sample.')
    st.markdown("""
    <style>
    /* Tabs row */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.75rem;
        justify-content: center;
        padding: 0.25rem 0 1rem 0;
    }

    /* Individual tab */
    .stTabs [data-baseweb="tab"] {
        flex: 1 1 0;
        max-width: 360px;
        justify-content: center;
        border-radius: 999px;
        padding: 0.7rem 1.1rem;
        font-weight: 600;
        border: 1px solid rgba(127, 127, 127, 0.22);
        background: color-mix(in srgb, currentColor 6%, transparent);
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.10);
        transition: all 0.18s ease;
    }

    /* Hover */
    .stTabs [data-baseweb="tab"]:hover {
        background: color-mix(in srgb, currentColor 10%, transparent);
        transform: translateY(-1px);
    }

    /* Selected tab */
    .stTabs [aria-selected="true"] {
        background: color-mix(in srgb, currentColor 14%, transparent);
        border: 1px solid rgba(127, 127, 127, 0.35);
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.14);
    }

    /* Hide default underline */
    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }

    /* Tab text */
    .stTabs [data-baseweb="tab"] p {
        font-size: 0.98rem;
    }
    </style>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Factor-Macro Trading", "PCA Analysis"])

    with tab1:
        macro.render(data['macro_df'], data['spread_df'], data['inflation_curve'], data['nominal_curve'])        
        result_df = macro.render_model(data['macro_df'], data['spread_df'], data['inflation_curve'], TRAIN_END)
        macroTrading.render(result_df, data['inflation_curve'], data['spread_df'], data['copom_df'], TRAIN_END)

    with tab2:
        pca_df, start_date,end_date = pca.plot_pca(data['nominal_curve'])
        
        filtered_spread = data['spread_df'][(data['spread_df']["closeDate"].dt.date >= start_date) & (data['spread_df']["closeDate"].dt.date <= end_date)].copy()
    
        st.subheader(f"Spread Stationarity Analysis ({start_date} to {end_date})")
        st.dataframe(stationarity.stationarity_table(filtered_spread['spread'], name="1Y1Y - 5Y5Y Spread"),hide_index=True)
        
        st.divider()
        pca.render(data['nominal_curve'], data['spread_df'], TRAIN_END)

        






if __name__ == "__main__":
    main()