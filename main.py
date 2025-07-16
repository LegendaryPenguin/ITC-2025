'''
main.py
Autor: Bryson Sanders
Creation Date: 05/30/2025
Last modified: 07/14/2025
Purpose: Impliment machine learning models with telemetry data to identify and categorize annomolies
'''
import os #for the KMeans model
os.environ["OMP_NUM_THREADS"] = "2"

import matplotlib.pyplot as plt
from models import prep_models, knn, k_means

def main():
    
    ddf = prep_models()
    silhouette_scores = dict()
    for clusters in [2,3,4,5,6,7,8,9,10]:
        silhouette_scores[clusters] = k_means(clusters, ddf)
    x = list(silhouette_scores.keys())
    y = list(silhouette_scores.values())
    
    plt.plot(x, y, marker='o')
    plt.title("Silhouette Score by Number of Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.show()
    


main()


