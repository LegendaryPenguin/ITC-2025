# Import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, average_precision_score
from pyod.utils.data import precision_n_scores
from pyod.models.iforest import IForest
from sklearn.neighbors import KNeighborsClassifier

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
    "mean", "var", "std", "len", "duration", "len_weighted", "gaps_squared", "n_peaks",
    "smooth10_n_peaks", "smooth20_n_peaks", "var_div_duration", "var_div_len",
    "diff_peaks", "diff2_peaks", "diff_var", "diff2_var", "kurtosis", "skew",
]

# Load in the data run tis before making the rest of the code
df = pd.read_csv(r"H:\python-projects\itc-research\dataset.csv", index_col="segment") 
'''
The following commented out code was ran and it was determined the the dataset is clean
This means that there were no empty values that we would need to decide what data to fill those gaps with
df.info()
print(df.isnull().sum()) # tells you how many empty cells there are
'''

X_train, y_train = df.loc[df.train==1, features], df.loc[df.train==1, "anomaly"]
X_test, y_test = df.loc[df.train==0, features], df.loc[df.train==0, "anomaly"]
X_train_nominal = df.loc[(df.anomaly==0)&(df.train==1), features]

prep = StandardScaler()
X_train_nominal2 = prep.fit_transform(X_train_nominal)
X_train2 = prep.transform(X_train)
X_test2 = prep.transform(X_test)

for neighbors in range(2,11):
    model = KNeighborsClassifier(n_neighbors=neighbors)
    model.fit(X_train2, y_train)

    y_predicted = model.predict(X_test2)
    y_predicted_score = model.predict_proba(X_test2)[:,1]

    print(model, '\n', evaluate_metrics(y_test, y_predicted, y_predicted_score))

