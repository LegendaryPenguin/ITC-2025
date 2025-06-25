'''
preprocessing.py
Autor: Bryson Sanders
Creation Date: 06/21/2025
Last modified: 06/25/2025
Purpose: Prepares data for use in training and testing
'''
#Import Libraries
#general libraries
import numpy as np
import pandas as pd

#tools
from sklearn.preprocessing import StandardScaler

#custom values and lists
from config import features


class DatasetDataFrame:
   def __init__(self, csv_filename): #csv file name must contain columns with test (determins train or test) and anomoly (is, or is not)
      #load full dataframe
      try:
         self.all = pd.read_csv(csv_filename, index_col="segment")
      except:
         print("Error loading data, ensure the csv meets the requirements for the DataFrame Class")
         print("Common problem: DataFrame column name \'segment\' expected for indexing.")
        
      #seperate csv into train or test with both all values and just the anomalies
      self.X_train, self.y_train = self.all.loc[self.all.train==1, features], self.all.loc[self.all.train==1, "anomaly"]
      self.X_train_nominal = self.all.loc[(self.all.anomaly==0)&(self.all.train==1), features]
      self.X_test, self.y_test = self.all.loc[self.all.train==0, features], self.all.loc[self.all.train==0, "anomaly"]
      self.X_anomalies_train = self.all.loc[(self.all.train==1) & (self.all.anomaly==1), features]
      self.X_anomalies_train_nominal = self.all.loc[(self.all.anomaly==1)&(self.all.train==1), features]
      self.X_anomalies_test = self.all.loc[(self.all.train==0) & (self.all.anomaly==1), features]

   #tool for ensuring valid datafram entry
   def is_clean(df):
      df.info()
      print(df.isnull().sum()) #tells you how many empty cells there are per column
   
   #Creates spererate csvs that can be accessed by multiple models and enable visual inspection
   def export_dfs_to_csv(self, base_filename=""):
      for df_name, df_data in vars(self).items():
         if "X2" in df_name:
            base_filename2 = "normalized_"
         elif base_filename == "":
            base_filename2 = f"{base_filename}"
         else:
            base_filename2 = f"{base_filename}_"
         if isinstance(df_data, pd.DataFrame):
            df_data.to_csv(f"seperate_dfs\\{base_filename2}{df_name}.csv")
         elif isinstance(df_data, np.ndarray):
            pd.DataFrame(df_data).to_csv(f"seperate_dfs\\{base_filename2}{df_name}.csv")

   #normalizes all features to between 0 and 1 and updates csv files
   def normalize(self):
      prep = StandardScaler()
      
      self.X2_train_nominal = prep.fit_transform(self.X_train_nominal)
      self.X2_train = prep.transform(self.X_train)
      self.X2_test = prep.transform(self.X_test)
      self.X2_anomalies_train_nominal = prep.fit_transform(self.X_anomalies_train_nominal)
      self.X2_anomalies_train = prep.transform(self.X_anomalies_train)
      self.X2_anomalies_test = prep.transform(self.X_anomalies_test)

      original_data = {
         "X2_train_nominal": self.X_train_nominal, 
         "X2_train": self.X_train, 
         "X2_test": self.X_test, 
         "X2_anomalies_train_nominal": self.X_anomalies_train_nominal,
         "X2_anomalies_train": self.X_anomalies_train,
         "X2_anomalies_test": self.X_anomalies_test
         } #helps to restore headers and index values for later
      

    

      

