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
    return pd.DataFrame({"closeDate": spot1y["closeDate"],"1y": spot1y["nominalRateValue"],"2y": spot2y["nominalRateValue"],"5y": spot5y["nominalRateValue"],"10y": spot10y["nominalRateValue"]})

def spread_fra(curve1, curve2):
    spread = curve1.copy()
    spread["spread"] = curve1["nominalRateValue"] - curve2["nominalRateValue"]
    return spread[['closeDate','spread']]


