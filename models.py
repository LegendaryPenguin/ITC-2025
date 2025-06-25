'''
models.py
Autor: Bryson Sanders
Creation Date: 06/21/2025
Last modified: 06/25/2025
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
    
    model = KMeans(n_clusters)
    model.fit_predict(ddf.X2_anomalies_train)
    print("Test shape before KMeans in main", ddf.X2_anomalies_test.shape)
    k_predicted = model.predict(ddf.X2_anomalies_test)
    print(k_predicted)
    print(type(k_predicted))
    k_means_silhouette_score = visualize_clusters(ddf, k_predicted)
    return k_means_silhouette_score