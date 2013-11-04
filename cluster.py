from numpy import array
from scipy.cluster.vq import kmeans2, whiten
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
import hmm

class StylometricCluster:

	def kmeans(self, stylo_vectors, k):
		'''
		Given a list of stylo_vectors, where each element is itself a list ('vector'),
		return the assigned clusters of each stylo_vector to clusters 0, 2, ..., k - 1
		where the i_th element in the returned list represents the cluster assigned
		to the i_th input vector

		Uses k-means clustering to do so
		TODO allow for more parameters to be passed to the <kmeans2> function
		'''
		feature_mat = array(stylo_vectors)
		# Normalizes by column
		normalized_features = whiten(feature_mat)
		
		# Initialize <k> clusters to be points in input vectors
		centroids, assigned_clusters = kmeans2(normalized_features, k, minit = 'points')
		return list(assigned_clusters)

	def agglom(self, stylo_vectors, k):
		'''
		Given a list of stylo_vectors, where each element is itself a list ('vector'),
		return the assigned clusters of each stylo_vector to clusters 0, 2, ..., k - 1
		where the i_th element in the returned list represents the cluster assigned
		to the i_th input vector

		Performs agglomerative clustering on <stylo_vectors> and returns the level in
		the dendogram with at most <k> clusters
		TODO allow for more parameters to be passed to the <kmeans2> function

		'''
		feature_mat = array(stylo_vectors)

		pairwise_dists = pdist(feature_mat)
		# Linkage matrix has somewhat complicated structure, but stores
		# all levels of the agglomerative clustering process, and can
		# be passed into other functions to extract info about the clusters. See:
		# http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
		linkage_mat = linkage(pairwise_dists)
		assigned_clusters = fcluster(linkage_mat, k, criterion = 'maxclust')

		return list(assigned_clusters)
		
	def hmm(self, stylo_vectors, k):
		'''
		Given a list of stylo_vectors, where each element is itself a feature vector,
		return the assigned clusters of each stylo_vector to clusters 0, 1, ..., k - 1
		where the i_th element in the returned list represents the cluster assigned
		to the i_th input vector
		
		Uses a hidden markov model to assign states to the observed feature vector outputs.
		Observed outputs assigned to identical states are assigned to the same cluster.
		'''
		return hmm.hmm_cluster(stylo_vectors, k)
			
		
		