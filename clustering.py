from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import pandas as pd

# =================== K-means & GMM ===================

def run_regime_model(df, feature_cols, model_type="gmm", n_regimes=3, train_end="2017-12-31", random_state=42):
    out = df.copy()
    train_end = pd.to_datetime(train_end)

    X = out[["closeDate"] + feature_cols].dropna().copy()
    idx = X.index
    train_mask = X["closeDate"] <= train_end
    X_train = X.loc[train_mask, feature_cols]
    X_full = X[feature_cols]

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_full_scaled = scaler.transform(X_full)

    if model_type == "kmeans":
        model = KMeans(n_clusters=n_regimes, n_init=20, random_state=random_state)
        model.fit(X_train_scaled)
        labels = model.predict(X_full_scaled)
        out.loc[idx, "regime"] = labels
        out.loc[idx, "is_train"] = train_mask.values
        centers = pd.DataFrame(
            scaler.inverse_transform(model.cluster_centers_),
            columns=feature_cols,
        )
        centers["regime"] = range(n_regimes)

    elif model_type == "gmm":
        model = GaussianMixture(n_components=n_regimes, random_state=random_state)
        model.fit(X_train_scaled)
        labels = model.predict(X_full_scaled)
        probs = model.predict_proba(X_full_scaled)
        out.loc[idx, "regime"] = labels
        out.loc[idx, "is_train"] = train_mask.values
        for i in range(n_regimes):
            out.loc[idx, f"prob_regime_{i}"] = probs[:, i]
        centers = pd.DataFrame(
            scaler.inverse_transform(model.means_),
            columns=feature_cols,
        )
        centers["regime"] = range(n_regimes)
    else:
        raise ValueError("model_type must be 'kmeans' or 'gmm'")

    return out, centers, model


# =================== Regime Z-Score ===================

def compute_regime_zscore(regime_df, train_end="2017-12-31"):
    df = regime_df.copy()
    train_end = pd.to_datetime(train_end)
    train = df[df["closeDate"] <= train_end]

    regime_stats = (
        train.groupby("regime")["residual_spread"]
        .agg(["mean", "std"])
        .reset_index()
    )
    regime_stats.columns = ["regime", "regime_mean", "regime_std"]

    df = df.merge(regime_stats, on="regime", how="left")
    df["regime_z"] = (df["residual_spread"] - df["regime_mean"]) / df["regime_std"]

    return df, regime_stats