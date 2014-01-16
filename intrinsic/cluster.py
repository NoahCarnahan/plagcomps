import hmm
import featureextraction

from numpy import array, matrix
from scipy.cluster.vq import kmeans2, whiten
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from collections import Counter

def cluster(method, k, items):
    if method == "kmeans":
        return _kmeans(items, k)
    elif method == "agglom":
        return _agglom(items, k)
    elif method == "hmm":
        return _hmm(items, k)
    else:
        raise ValueError("Invalid cluster method. Acceptable values are 'kmeans', 'agglom', or 'hmm'.")

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
    Given a list of stylo_vectors, where each element of the list is a feature vector,
    use a hidden Markov model to assign one of k preposed state values to the vectors, 
    Observed outputs assigned similar state values are of the same cluster.

    To Do:
    Return a list of confidence values for the vector assignments
    where the i_th element in the returned list represents the confidence assigned
    to the i_th input vector.
    '''
    
    #get centroids and cluster assignments
    centroids, assigned_clusters = hmm.hmm_cluster(stylo_vectors, k)
    #print 'centroids are: ', centroids
    #print assigned_clusters
    
    #

    # Get confidences
    # TODO: Develop a real notion of confidence for hmm clustering
    plag_cluster = Counter(assigned_clusters).most_common()[-1][0]
    cluster_assign = [1 if x == plag_cluster else 0 for x in assigned_clusters]
    confidences = hmm.get_confidences(stylo_vectors, centroids, cluster_assign)
    return cluster_assign , confidences

def _test():
    doc = open('/copyCats/pan-plagiarism-corpus-2009/intrinsic-detection-corpus/suspicious-documents/part1/suspicious-document00667.txt', 'r')
    text = doc.read()
    #print text
    f = featureextraction.FeatureExtractor(text)
    doc.close()
    fs = f.get_feature_vectors(["average_sentence_length","average_word_length"],"sentence")
    #fs = [[4.5,5.2,1.9],[1.1,2.03,2.45],[4.5,5.2,8.1],[1.1,2.03,2.45],[4.5,5.2,8.1],[1.1,2.03,2.45],[4.5,5.2,8.3],[1.1,2.13,2.45],[4.6,5.1,8.1],[1.1,2.02,2.45],[4.5,5.2,8.2]]
    #print cluster("kmeans", 2, fs)
    #print cluster("agglom", 2, fs)
    print cluster("hmm", 2, fs)
    #print cluster("hmm", 2, fs)
    #print cluster("hmm", 2, fs)

if __name__ == "__main__":
    _test()
        