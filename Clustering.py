import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

os.environ["OMP_NUM_THREADS"] = "2"

def load_features():
    # Replace with actual data loading
    df = pd.read_csv("features.csv")  # Assumes 'segment_id' column is present
    return df

def run_final_clustering(df, n_clusters=3):
    feature_cols = [col for col in df.columns if col not in ["segment_id", "anomaly_label"]]
    X = df[feature_cols]

    # Fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    df["cluster_id"] = kmeans.fit_predict(X)

    # Map numeric clusters to interpretable labels
    cluster_label_map = {0: "Segment A", 1: "Segment B", 2: "Segment C"}
    df["cluster_label"] = df["cluster_id"].map(cluster_label_map)

    # Analyze group-wise feature means
    print("\n=== Average feature values by cluster ===")
    print(df.groupby("cluster_label")[feature_cols].mean())

    # Optional: Compare with original anomaly labels
    if "anomaly_label" in df.columns:
        print("\n=== Cross-tab of cluster vs anomaly ===")
        print(pd.crosstab(df["cluster_label"], df["anomaly_label"]))

    # Export labeled dataset
    df.to_csv("clustered_segments.csv", index=False)
    print("\nClustered data saved to clustered_segments.csv")

def main():
    df = load_features()
    run_final_clustering(df, n_clusters=3)

if __name__ == "__main__":
    main()
