#import the required libraries
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import PCA

#import working directory
import os
#import working directory
os.chdir('C:/Users/ek-stud/Desktop/CLUSTER')

#load pandas and read data
import pandas as pd
data = pd.read_csv('sample.txt', sep= ' ', header = None)

#preprocess data
#
data = data.dropna(axis = 1, how = 'all')
del data[data.columns[0]]
data = data

#KMeans class
class clustering():
    """docstring for clustering."""
    def __init__(self):
        #init function plots/returns out data
        self.plot_optimum_cluster(data)
        self.plot_kmeans(data)

    def plot_optimum_cluster(self, data):
        #set a list to append the iter values
        iter_num = []
        for i in range(1, 15):
            #perform kmeans to get best cluster value using elbow method
            kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42, max_iter = 300)
            kmeans.fit(data)
            iter_num.append(kmeans.inertia_)
            #plot the optimum graph
        plt.plot(range(1, 15), iter_num)
        plt.title('The Elbow method of determining number of clusters')
        plt.xlabel('Number of clusters')
        plt.ylabel('iter_num')
        plt.show()
            #from here we could observe the best cluster is
            #around 5..further insight shows cluster of 7 is as well good
        pass

    def plot_kmeans(self, data):
        #instatitate our kmeans class
        kmeans = KMeans(n_clusters = 7, init = 'k-means++', random_state = 42)
        y_kmeans = kmeans.fit_predict(data)

        for i in range(0, 7):
            color = ['red', 'blue', 'green', 'cyan', 'yellow', 'black', 'purple']
            label = ['cluster 1', 'cluster 2', 'cluster 3', 'cluster 4', 'cluster 5', 'cluster 6', 'cluster 7']
            '''
                        summarizing all of this in one line of code below
            plt.scatter(np.where(y_kmeans == 0), data.iloc[y_kmeans == 0, 0], s = 25, c = 'red', label = 'Cluster 1')
            plt.scatter(np.where(y_kmeans == 1), data.iloc[y_kmeans == 1, 0], s = 25, c = 'blue', label = 'Cluster 2')
            plt.scatter(np.where(y_kmeans == 2), data.iloc[y_kmeans == 2, 0], s = 25, c = 'green', label = 'Cluster 3')
            plt.scatter(np.where(y_kmeans == 3), data.iloc[y_kmeans == 3, 0], s = 25, c = 'cyan', label = 'Cluster 4')
            plt.scatter(np.where(y_kmeans == 4), data.iloc[y_kmeans == 4, 0], s = 25, c = 'yellow', label = 'Cluster 5')
            plt.scatter(np.where(y_kmeans == 5), data.iloc[y_kmeans == 5, 0], s = 25, c = 'black', label = 'Cluster 6')
            plt.scatter(np.where(y_kmeans == 6), data.iloc[y_kmeans == 6, 0], s = 25, c = 'purple', label = 'Cluster 7')
            '''#here
            plt.scatter(np.where(y_kmeans == i), data.iloc[y_kmeans == i, 0], s = 25, c = color[i], label = label[i])
        plt.title('frequency clustering w.r.t damping and eigen modes[KMEANS]')
        plt.xlabel('sample index')
        plt.ylabel('frequencies')
        plt.legend()
        plt.show()
        pass

#we create another function for DBSCAN
#
class DB_SCAN():
    """docstring for DBSCAN."""
    def __init__(self):
        self.plot_dbscan(data)

    def plot_dbscan(self, data):
        #instatiate the DBSCAN class
        dbscan = DBSCAN(eps = 0.3, min_samples = 10).fit(data)
        core_samples_mask = np.zeros_like(dbscan.labels_, dtype = bool)
        core_samples_mask[dbscan.core_sample_indices_] = True
        labels = dbscan.labels_

        #now we check the number of clusters in labels and since -1 is a noise in our label we remove/set it to 1
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        #now we plot the dbscan
        unique_labels = set(labels)
        colors = [plt.cm.Set3(each) for each in np.linspace(0, 1, len(unique_labels))]
        for i in range(0, n_clusters):
            color = ['red', 'blue', 'green', 'cyan', 'yellow', 'black', 'purple']
            label = ['cluster 1', 'cluster 2', 'cluster 3', 'cluster 4', 'cluster 5', 'cluster 6', 'cluster 7']
            plt.scatter(np.where(labels == i), data.iloc[labels == i, 0], s = 25, c = color[i], label = label[i])

        plt.title('frequency clustering w.r.t damping and eigen modes[DBSCAN]')
        plt.xlabel('sample index')
        plt.ylabel('frequencies')
        plt.legend()
        plt.show()
        pass

#Clustering using PCA and Kmeans
##
#
class pca_kmeans():
    """docstring for pca_kmeans."""
    def __init__(self):
        self.plot_pca_kmeans(data)

    def plot_pca_kmeans(self, data):
        #instatitate the pca class
        pca = PCA(n_components='mle', whiten=True, svd_solver='full').fit(data)
        pca_transform = pca.transform(data)
        #instatiate our Kmeans
        kmeans = KMeans(n_clusters = 7, init = 'k-means++', random_state = 42)
        #fit our transform dimension into kmeans
        y_kmeans = kmeans.fit(pca_transform)
        labels = y_kmeans.labels_
        #then plot our 2D graph
        for i in range(0, 7):
            color = ['red', 'blue', 'green', 'cyan', 'yellow', 'black', 'purple']
            label = ['cluster 1', 'cluster 2', 'cluster 3', 'cluster 4', 'cluster 5', 'cluster 6', 'cluster 7']
            plt.scatter(np.where(labels == i), data.iloc[labels == i, 0], s = 25, c = color[i], label = label[i])

        plt.title('frequency clustering w.r.t damping and eigen modes[PCA_KMEANS]')
        plt.xlabel('sample index')
        plt.ylabel('frequencies')
        plt.legend()
        plt.show()
        pass

#HDBSCAN...eliminates epsilon value and introduced a new approach
#Condensing dendogram...
#import required libraries
'''
HDBSCAN is a recent algorithm developed by some of the
same people who write the original DBSCAN paper.
Their goal was to allow varying density clusters.
The algorithm starts off much the same as DBSCAN:
we transform the space according to density,
exactly as DBSCAN does, and perform single linkage
clustering on the transformed space. Instead of taking
an epsilon value as a cut level for the dendrogram however,
a different approach is taken: the dendrogram is
condensed by viewing splits that result in a small number
of points splitting off as points falling out of a cluster.
This results in a smaller tree with fewer clusters that ulose points.

#-------------AS SEEN ON -----------------------------#
http://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html
#-----------------------------------------------------#
'''
import hdbscan

#write a class to implement HDBSCAN
class h_dbscan():
    """docstring for h_dbscan."""
    def __init__(self):
        return self.plot_hdb_scan(data)

    def plot_hdb_scan(self, data):
        #instantiate the HDBSCAN
        min_cluster_size = 10
        hd_scan = hdbscan.HDBSCAN(min_cluster_size)
        #see labels after prediction using fit_predict
        cluster_labels = hd_scan.fit_predict(data)
        #print len(np.unique(cluster_labels)) From here 8 clusters were identified
        #plot using a for loop
        for i in range(0, len(np.unique(cluster_labels))):
            color = ['red', 'blue', 'green', 'cyan', 'yellow', 'black', 'purple', 'brown']
            label = ['cluster 1', 'cluster 2', 'cluster 3', 'cluster 4', 'cluster 5', 'cluster 6', 'cluster 7', 'cluster 8']
            plt.scatter(np.where(cluster_labels == i), data.iloc[cluster_labels == i, 0], s = 25, c = color[i], label = label[i])

        plt.title('frequency clustering w.r.t damping and eigen modes[HDBSCAN]')
        plt.xlabel('sample index')
        plt.ylabel('frequencies')
        plt.legend()
        plt.show()
        pass

#call our cluster class
if __name__ == '__main__':
    #KMeans
    kmeans = clustering()
    #DBSCAN
    dbscan = DB_SCAN()
    #PCA_KMEANS
    pca_kmean = pca_kmeans()
    #HDBSCAN
    hd_scan = h_dbscan()


##################################################################################
##########################   CONCLUSION #########################################
############# KMEANS || DBSCAN || 'PCA+KMEANS' CLUSTERING #########################
#################################################################################
#We observed that the DBSCAN algorithm is a little better than the KMeans clustering algorithm
#in the clustering of our eigne frequency data.
#To this end it is therefore imperative that choose DBSCAN over KMeans
#Another approach would be to use PCA together with KMeans..perhaps we could get a better
#result...at this point--> DBSCAN is the obviously cluster winner
#.....................................
#It turns out combining PCA with Kmeans doesnt actually improve the Clustering
#rather it changes the cluster approach which doesnt solve our problem.
#Hence conclude DBSCAN has the productive clustering for our data
#Flowwoed next to HDBSCAN
