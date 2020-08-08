# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 13:52:29 2020

@author: DELL
"""

"""centroid-based algorithm, which works by updating candidates for centroids to be the mean of the points within a given region. 
These candidates are then filtered in a post-processing stage to eliminate near-duplicates to form the final set of centroids."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.cluster import MeanShift
from matplotlib import pyplot
from numpy import unique
from numpy import where
# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values
model = MeanShift()
# fit model and predict clusters
yhat = model.fit_predict(X)
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
	# get row indexes for samples with this cluster
	row_ix = where(yhat == cluster)
	# create scatter of these samples
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
pyplot.show()