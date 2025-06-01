'''
Autor: Bryson Sanders
Creation Date: 05/30/2025
Last modified: 06/01/2025
Purpose: Impliment machine learning models with telemetry data to identify and categorize annomolies
'''
# Import Libraries
import os
os.environ["OMP_NUM_THREADS"] = "2" #for the KMeans model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, average_precision_score
from pyod.utils.data import precision_n_scores
from pyod.models.iforest import IForest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from segment import Segment

# Define Functions and Lists
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
features = [
    "mean", "var", "std", "len_weighted", "gaps_squared", "n_peaks",
    "smooth10_n_peaks", "smooth20_n_peaks", "var_div_duration", "var_div_len",
    "diff_peaks", "diff2_peaks", "diff_var", "diff2_var", "kurtosis", "skew",
] # removed len and dration as recomended by authors

# Load Data
df = pd.read_csv("dataset.csv", index_col="segment") 
'''
this code tells you column headers and quatity of null values for each column
df.info()
print(df.isnull().sum()) # tells you how many empty cells there are
'''



# K-Nearest Neighbor algorithm
def knn(k):
    # Categorizes Segments as Training or Testing
    X_train, y_train = df.loc[df.train==1, features], df.loc[df.train==1, "anomaly"]
    X_test, y_test = df.loc[df.train==0, features], df.loc[df.train==0, "anomaly"]
    X_train_nominal = df.loc[(df.anomaly==0)&(df.train==1), features]

    prep = StandardScaler()
    X_train_nominal2 = prep.fit_transform(X_train_nominal)
    X_train2 = prep.transform(X_train)
    X_test2 = prep.transform(X_test)
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train2, y_train)
    y_predicted = model.predict(X_test2)
    y_predicted_score = model.predict_proba(X_test2)[:,1]

    print(model, '\n', evaluate_metrics(y_test, y_predicted, y_predicted_score))
#attempting data clustering
def k_means(n_clusters):
    X_train = df.loc[(df.train==1)&(df.anomaly==1), features] #assuming we can decide between anomoly and non anomoly with first model
    X_test = df.loc[(df.train==0)&(df.anomaly==1), features]
    X_train_nominal = df.loc[(df.anomaly==1)&(df.train==1), features]

    prep = StandardScaler()
    X_train_nominal2 = prep.fit_transform(X_train_nominal)
    X_train2 = prep.transform(X_train)
    X_test2 = prep.transform(X_test)
    model = KMeans(n_clusters)
    model.fit_predict(X_train2)
    k_predicted = model.predict(X_test2)
k_means(6)
 