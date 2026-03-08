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
                "nominalRateValue": point.get("nominalRateValue")})
    df = pd.DataFrame(rows)
    df["closeDate"] = pd.to_datetime(df["closeDate"])
    return df

def load_json(path):
    with open(path, "r") as f:
        raw = json.load(f)
    return read_json(raw)

def build_curve_df(spot1y, spot2y, spot5y, spot10y):
    dfs = [spot1y.rename(columns={"nominalRateValue": "1y"}),spot2y.rename(columns={"nominalRateValue": "2y"}),
        spot5y.rename(columns={"nominalRateValue": "5y"}),spot10y.rename(columns={"nominalRateValue": "10y"})]
    df = dfs[0][["closeDate", "1y"]]
    for d in dfs[1:]:
        df = df.merge(d[["closeDate", d.columns[-1]]], on="closeDate", how="inner")
    return df.sort_values("closeDate").reset_index(drop=True)

def spread_fra(curve1, curve2):
    spread = curve1.copy()
    spread["spread"] = curve1["nominalRateValue"] - curve2["nominalRateValue"]
    return spread[['closeDate','spread']]


