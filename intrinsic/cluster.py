import hmm
from plagcomps.intrinsic import outlier_detection

from numpy import array, matrix, random
from scipy.cluster.vq import kmeans2, whiten
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from collections import Counter

from kmedians import kmedians

def cluster(method, k, items):
    if method == "kmeans":
        return _kmeans(items, k)
    elif method == "agglom":
        return _agglom(items, k)
    elif method == "hmm":
        return _hmm(items, k)
    elif method == "outlier":
        return outlier_detection.density_based(items)
    else:
        raise ValueError("Invalid cluster method. Acceptable values are 'kmeans', 'agglom', or 'hmm'.")

def _kmedians(stylo_vectors, k):
    features = array(stylo_vectors)

    clusterer = kmedians(k)
    clusterer.fit(features)

    return clusterer.get_confidences()

def _kmeans(stylo_vectors, k):
    '''
    Given a list of stylo_vectors, where each element is itself a list ('vector'),
    return our confidence that each vector is in the plagiarism cluster 
    where the i_th element in the returned list represents the confidence assigned
    to the i_th input vector

    Uses k-means clustering to do so
    TODO allow for more parameters to be passed to the <kmeans2> function
    '''
    feature_mat = array(stylo_vectors)
    # Normalizes by column
    normalized_features = whiten(feature_mat)
    
    # Initialize <k> clusters to be points in input vectors
    centroids, assigned_clusters = kmeans2(normalized_features, k, minit = 'points')
    
    
    # Get confidences
    
    # Special case when there is only one atom
    if len(stylo_vectors) == 1:
        return [0] # We are 0% confident that the atom is plagiarized.
    
    if k == 2:
        confidences = []
        plag_cluster = Counter(assigned_clusters).most_common()[-1][0] #plag_cluster is the smallest cluster
        not_plag_cluster = 1 if plag_cluster == 0 else 0
        for feat_vec in normalized_features:
            distance_from_plag = float(pdist(matrix([centroids[plag_cluster], feat_vec])))
            distance_from_notplag = float(pdist(matrix([centroids[not_plag_cluster], feat_vec])))
            conf = distance_from_notplag / (distance_from_notplag + distance_from_plag)
            confidences.append(conf)
    else:
        # TODO: Develop a notion of confidence when k != 2
        plag_cluster = Counter(assigned_clusters).most_common()[-1][0]
        confidences = [1 if x == plag_cluster else 0 for x in assigned_clusters]
        
    return confidences


def _agglom(stylo_vectors, k):
    '''
    Given a list of stylo_vectors, where each element is itself a list ('vector'),
    return our confidence that each vector is in the plagiarism cluster 
    where the i_th element in the returned list represents the confidence assigned
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

    # Get confidences
    # TODO: Develop a real notion of confidence for agglom clustering
    plag_cluster = Counter(assigned_clusters).most_common()[-1][0]
    return [1 if x == plag_cluster else 0 for x in assigned_clusters]
    
def _hmm(stylo_vectors, k):
    '''
    Given a list of stylo_vectors, where each element is itself a feature vector,
    return our confidence that each vector is in the plagiarism cluster 
    where the i_th element in the returned list represents the confidence assigned
    to the i_th input vector
    
    Uses a hidden markov model to assign states to the observed feature vector outputs.
    Observed outputs assigned to identical states are assigned to the same cluster.
    '''
    centroids, assigned_clusters = hmm.hmm_cluster(stylo_vectors, k)
    
    # Get confidences
    # TODO: Develop a real notion of confidence for hmm clustering
    plag_cluster = Counter(assigned_clusters).most_common()[-1][0]
    return [1 if x == plag_cluster else 0 for x in assigned_clusters]

def two_normal_test(n, spacing):
    first = [[random.normal()] for x in range(n)]
    second = [[random.normal(spacing)] for x in range(n)]

    for c in ["kmeans", "agglom", "hmm"]:
        print c, cluster(c, 2, first + second)

def _test():
    fs = [
        [4.5, 5.2, 1.9],
        [1.1, 2.03, 2.45],
        [4.5, 5.2, 8.1]
    ]
    cluster_one = [
        [1.0, 1.0],
        [1.25, 1.25],
        [1.5, 1.5],
        [1.75, 1.75],
        [2.0, 2.0]
    ]
    cluster_two = [
        [10.0, 10.0],
        [10.25, 10.25],
        [10.5, 10.5],
        [10.75, 10.75],
        [11.0, 11.0]
    ]

    fs_obvious = cluster_one + cluster_two

    print cluster("kmeans", 2, fs_obvious)
    print cluster("agglom", 2, fs_obvious)
    print cluster("hmm", 2, fs_obvious)

    print cluster("kmeans", 2, fs)
    print cluster("agglom", 2, fs)
    print cluster("hmm", 2, fs)

if __name__ == "__main__":
    #_test()
    _two_normal_test(10, 2)
        
