'''
utils.py
Autor: Bryson Sanders
Creation Date: 06/21/2025
Last modified: 07/14/2025
Purpose: Assists the evaluation of a model/algorithm's performance
'''

#Import Libraries
#general libraries
import numpy as np
import pandas as pd

#evaluation libraries
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, average_precision_score, silhouette_score
from pyod.utils.data import precision_n_scores
from sklearn.decomposition import PCA

#visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

#custom libraries
from config import features

#General Evaluation Tool
def evaluate_metrics(y_test, y_pred, y_proba=None, digits=3):
    res = {"Accuracy": round(accuracy_score(y_test, y_pred), digits),
           "Precision": precision_score(y_test, y_pred).round(digits),
           "Recall": recall_score(y_test, y_pred).round(digits),
           "F1": f1_score(y_test, y_pred).round(digits),
           "MCC": round(matthews_corrcoef(y_test, y_pred), ndigits=digits)}
    if y_proba is not None:
        res["AUC_PR"] = average_precision_score(y_test, y_proba).round(digits)
        res["AUC_ROC"] = roc_auc_score(y_test, y_proba).round(digits)
        res["PREC_N_SCORES"] = precision_n_scores(y_test, y_proba).round(digits)
    return res

def visualize_clusters(ddf, k_predicted, n_clusters):

    X_test_df = ddf.X_anomalies_test.copy()
    X_test_df["cluster"] = k_predicted
    X_test_df = X_test_df.reset_index()
    
    X = ddf.X2_anomalies_test
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X)
    pca_df = pd.DataFrame(reduced, columns=["PC1", "PC2"])
    pca_df["cluster"] = k_predicted

    sns.scatterplot(data=pca_df, x="PC1", y="PC2", style="cluster", hue="cluster", palette="deep")
    plt.title(f"KMeans Test Prediction with {n_clusters} Clusters")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()

    clustered_df = X_test_df.groupby('cluster')[['mean', 'var', 'kurtosis']].mean()
    sns.heatmap(clustered_df, annot=True, cmap="YlGnBu")
    plt.title("Average Feature Values by Cluster")
    plt.show()

    score = silhouette_score(ddf.X_anomalies_test, k_predicted)
    print(f"Silhouette Score: {score: 3f}")
    return score
