# clustering
EIGEN FREQUENCY CLUSTERING USING [KMEANS] [KMEANS &amp; PCA ] [DBSCAN] [HDBSCAN]

Kmeans has been used over the years as a clustering algorithm and quite reasonably performed very well. 
Its approval however is not due to any complex structure but a rather subtle appraoch towards data clustering using K-value(this value 
is the number of cluster using their mean. ) This mean value then serves as a refernce point for the clusters sorrounding it.

As simple and nicely as KMeans may try to solve our clustering problem. It would howvere fall short of clustering data
containing very noisy observation. This is therefore the disadvantage of Kmeans.

To solve this problem therefor..a team of researchers have introduced an approach which ignores the mean value for clustering but
even better this time considers their density...Hmmn! Reasonable right!. Indeed it is.

This algorithm is able to cluster data based on their density..According to their wiki
#-------------------------------------------------------------------
### A cluster then satisfies two properties:

### All points within the cluster are mutually density-connected.
### If a point is density-reachable from any point of the cluster, it is part of the cluster as well.

