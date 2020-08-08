# Clustering

1.K-Means Clustering
randomly initialize their respective center points  but use kmeans++ get fastly accurately
Each data point is classified by computing the distance between that point and each group center, 
and then classifying the point to be in the group whose center is closest to it.
advantage that it’s pretty fast, as all we’re really doing is computing the distances between points and group center

2.Agglomerative Hierarchical Clustering
fall into 2 categories: top-down or bottom-up
Bottom-up algorithms treat each data point as a single cluster and then successively merge (or agglomerate) pairs of clusters until all clusters have been merged into a single cluster that contains all data points
This hierarchy of clusters is represented as a tree (or dendrogram).

3.Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
DBSCAN is a density based clustered algorithm.
The neighborhood of this point is extracted using a distance epsilon ε (All points which are within the ε distance are neighborhood points).

4.Mean-Shift Clustering
find dense areas of data points.
centroid-based algorithm meaning that the goal is to locate the center points of each class

5.Clustering using Gaussian Mixture Models (GMM)
drawbacks of K-Means is its naive use of the mean value for the cluster center.
K-Means can’t handle this because the mean values of the clusters are a very close together
Gaussian Mixture Models (GMMs) give us more flexibility than K-Means.
two parameters to describe the shape of the clusters: the mean and the standard deviation
clusters can take any kind of elliptical shape (since we have standard deviation in both the x and y directions
Thus, each Gaussian distribution is assigned to a single cluster
GMMs are a lot more flexible in terms of cluster covariance than K-Means; due to the standard deviation parameter
