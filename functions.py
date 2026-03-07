import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


def run_pca(curve_df):
    X = curve_df[["1y", "2y", "5y", "10y"]].dropna()

    pca = PCA(n_components=3)
    pcs = pca.fit_transform(X)

    pca_df = curve_df.loc[X.index, ["closeDate"]].copy()
    pca_df["Level"] = pcs[:, 0]
    pca_df["Slope"] = pcs[:, 1]
    pca_df["Curvature"] = pcs[:, 2]

    loadings = pd.DataFrame(pca.components_.T,index=["1y", "2y", "5y", "10y"],columns=["Level", "Slope", "Curvature"])

    explained = pd.DataFrame({"Component": ["Level", "Slope", "Curvature"],"Explained Variance": pca.explained_variance_ratio_})

    for pc in ["Level", "Slope", "Curvature"]:
        mean = pca_df[pc].mean()
        std = pca_df[pc].std()
        pca_df[f"{pc}_z"] = (pca_df[pc] - mean) / std

    return pca_df, loadings, explained