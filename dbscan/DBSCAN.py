# DBSCAN Clustering

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values


# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import DBSCAN
dbscan=DBSCAN(eps=3,min_samples=4)

# Fitting the model

model=dbscan.fit(X)

labels=model.labels_
#-1 is outiler,  0,1,2 clusteers

from sklearn import metrics

#identifying the points which makes up our core points
sample_cores=np.zeros_like(labels,dtype=bool)   #everything false

sample_cores[dbscan.core_sample_indices_]=True  #except -1 rem are true

#Calculating the number of clusters

n_clusters=len(set(labels))- (1 if -1 in labels else 0) #-1 is noise



print(metrics.silhouette_score(X,labels)) #score cal based avg mean of no of points imdicate noisey another point as group cluster



