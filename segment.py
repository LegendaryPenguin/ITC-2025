'''
segment.py
Autor: Bryson Sanders
Creation Date: 06/01/2025
Last modified: 06/01/2025
Purpose: simplify visualization
'''
# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

# Creating Class
class Segment:
    def __init__(self, segments_file_name, segment_id):
        self.id = segment_id #which segment are you looking for
        self.df = pd.read_csv(segments_file_name) #opens file
        self.df = self.df[self.df["segment"] == self.id] #issolates segment within file
    def __iter__(self):
        return Segment(self.id)
    def visual(self):
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        plt.figure(self.id)
        plt.title(f"Telemetry Segment {self.id}")
        plt.xlabel("Timestamp")
        plt.ylabel("Analog Value")
        plt.plot(self.df['timestamp'], self.df['value'])
        plt.show(block=False)

def graph_segments_in_cluster(k_predicted, cluster, test_csv=r"seperate_dfs\X_anomalies_test.csv"): #note, when called it will mess up the cluster visualization graphs
    k_predicted_2 = k_predicted
    dataframe = pd.read_csv(test_csv)
    segments_to_clusters = dict()
    segments_in_test = dataframe["segment"]
    segment_in_test = segments_in_test.tolist()
    for i in range(len(k_predicted_2)):
        segments_to_clusters[segment_in_test[i]] = k_predicted_2[i]
    graph_count = 0
    for segment_ids, cluster_ids in segments_to_clusters.items():
        if cluster_ids == cluster and graph_count <= 5:
            segment = Segment("segments.csv", segment_ids)
            segment.visual()
            print(segment_ids, cluster_ids)
            graph_count += 1

