'''
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
        plt.plot(self.df['timestamp'], self.df['value'])
        plt.show()