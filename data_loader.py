import os
import json
import pandas as pd

def read_json(raw):
    rows = []
    for bucket in raw:
        bucket_name = bucket.get("bucketName")
        for point in bucket.get("curve", []):
            rows.append({
                "bucketName": bucket_name,
                "closeDate": point.get("closeDate"),
                "nominalRateValue": point.get("nominalRateValue"),
                "inflationRateValue": point.get("inflationRateValue")})
    df = pd.DataFrame(rows)
    df["closeDate"] = pd.to_datetime(df["closeDate"])
    return df

def load_json(path):
    with open(path, "r") as f:
        raw = json.load(f)
    return read_json(raw)

def build_nominal_curve_df(spot1y, spot2y, spot5y, spot10y):
    names = ["Nominal_1y", "Nominal_2y", "Nominal_5y", "Nominal_10y"]
    spots = [spot1y, spot2y, spot5y, spot10y]

    df = spots[0][["closeDate", "nominalRateValue"]].rename(columns={"nominalRateValue": names[0]})
    for spot, name in zip(spots[1:], names[1:]):
        temp = spot[["closeDate", "nominalRateValue"]].rename(columns={"nominalRateValue": name})
        df = df.merge(temp, on="closeDate", how="inner")
    return df.sort_values("closeDate").reset_index(drop=True)

def build_inflation_curve_df(spot1y, spot2y, spot5y, spot10y):
    names = ["Inflation_1y", "Inflation_2y", "Inflation_5y", "Inflation_10y"]
    spots = [spot1y, spot2y, spot5y, spot10y]

    df = spots[0][["closeDate", "inflationRateValue"]].rename(columns={"inflationRateValue": names[0]})
    for spot, name in zip(spots[1:], names[1:]):
        temp = spot[["closeDate", "inflationRateValue"]].rename(columns={"inflationRateValue": name})
        df = df.merge(temp, on="closeDate", how="inner")
    return df.sort_values("closeDate").reset_index(drop=True)

def spread_fra(curve1, curve2):
    merged = curve1[["closeDate", "nominalRateValue"]].merge(curve2[["closeDate", "nominalRateValue"]],on="closeDate", how="inner", suffixes=("_1", "_2"))
    merged["spread"] = merged["nominalRateValue_1"] - merged["nominalRateValue_2"]
    return merged[["closeDate", "spread"]]


def load_macro_excel(filepath):
    df = pd.read_excel(filepath)
    df["closeDate"] = pd.to_datetime(df["date"])
    df = df.sort_values("closeDate").reset_index(drop=True)
    return df[['closeDate','value']]


def build_macro_df(macro_factors):
    series_dict = {}
    for macro in sorted(macro_factors):
        filename = f"{macro}.xlsx"
        filepath = os.path.join("data", filename)
        df = load_macro_excel(filepath)
        series_dict[macro] = df.rename(columns={"value": macro})
    first_name = list(series_dict.keys())[0]
    result = series_dict[first_name][["closeDate"]].copy()
    for name, df in series_dict.items():
        result = result.merge(df[["closeDate", name]], on="closeDate", how="left")
    result = result.sort_values("closeDate").reset_index(drop=True)
    return result