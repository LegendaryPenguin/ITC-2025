'''
models.py
Autor: Bryson Sanders
Creation Date: 06/21/2025
Last modified: 07/14/2025
Purpose: Contains all models used for separating anomalies and uncorrupted data and clustering the anomalies
'''
#tools
from pyod.models.iforest import IForest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

#custom libraries
from preprocessing import DatasetDataFrame as DDF
from utils import evaluate_metrics, visualize_clusters, silhouette_score
from segment import graph_segments_in_cluster as gsic

def prep_models(dataset = "dataset.csv"):
    ddf = DDF(dataset)
    ddf.normalize()
    ddf.export_dfs_to_csv()
    return ddf

def knn(k, ddf):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(ddf.X2_train, ddf.y_train)
    y_predicted = model.predict(ddf.X2_test)
    y_predicted_score = model.predict_proba(ddf.X2_test)[:,1]
    evaluate_metrics(ddf.X2_test, y_predicted)

def k_means(n_clusters, ddf):
    
    model = KMeans(n_clusters, random_state=1)
    model.fit_predict(ddf.X2_anomalies_train)
    k_predicted = model.predict(ddf.X2_anomalies_test)
    k_means_silhouette_score = visualize_clusters(ddf, k_predicted, n_clusters)
    return k_means_silhouette_score