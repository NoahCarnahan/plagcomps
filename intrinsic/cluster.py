import hmm
import featureextraction
from kmedians import KMedians
import outlier_detection
import classify
#from plagcomps.intrinsic import outlier_detection

from numpy import array, matrix, random
from scipy.cluster.vq import kmeans2, whiten
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from collections import Counter


def cluster(method, k, items, **kwargs):
    if method == "kmeans":
        return _kmeans(items, k)
    elif method == "agglom":
        return _agglom(items, k)
    elif method == "hmm":
        return _hmm(items, k)
    elif method == "median_simple":
        return _median_simple(items)
    elif method == "median_kmeans":
        return _median_kmeans(items, k)
    elif method == "outlier":
        return outlier_detection.density_based(items, **kwargs)
    elif method == "kmedians":
        return _kmedians(items, k)
    elif method == "nn_confidences":
        return _nn_confidences(items)
    elif method == "combine_confidences":
        return _combine_confidences(items, **kwargs)
    else:
        raise ValueError("Invalid cluster method. Acceptable values are 'kmeans', 'agglom', or 'hmm'.")

def _kmedians(stylo_vectors, k):
    features = array(stylo_vectors)

    clusterer = KMedians(k)
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
            if plag_cluster < len(centroids) and plag_cluster >= 0:
                distance_from_plag = float(pdist(matrix([centroids[plag_cluster], feat_vec])))
                distance_from_notplag = float(pdist(matrix([centroids[not_plag_cluster], feat_vec])))
                if distance_from_notplag + distance_from_plag > 0:
                    conf = distance_from_notplag / (distance_from_notplag + distance_from_plag)
                else:
                    conf = 0
                confidences.append(conf)
            else:
                confidences.append(0)
    else:
        # TODO: Develop a notion of confidence when k != 2
        non_plag_cluster = Counter(assigned_clusters).most_common()[0][0]
        max_dist = 0
        max_dist_vec = normalized_features[0]

        for i in xrange(len(assigned_clusters)):
            if assigned_clusters[i] == non_plag_cluster:
                dist = float(pdist(matrix([centroids[non_plag_cluster], normalized_features[i]]))) 
                if dist > max_dist:
                    max_dist_vec = normalized_features[i]
                    max_dist = dist

        avg_centroid_dist = 0
        sum_dist = 0
        centroid_distance = {}
        for center in centroids:
            sum_dist += float(pdist(matrix([center, centroids[non_plag_cluster]])))
            avg_centroid_dist = float(sum_dist/len(centroids))
        
        confidences = []
        for i in xrange(len(assigned_clusters)):
            try:
                if assigned_clusters[i] != non_plag_cluster:
                    distance_from_max = float(pdist(matrix([max_dist_vec, normalized_features[i]])))
                    centroid_distance = float(pdist(matrix([centroids[assigned_clusters[i]], centroids[non_plag_cluster]])))
                    conf = (1-(float(1/(max_dist/distance_from_max))))*float(centroid_distance/avg_centroid_dist)
                    confidences.append(conf)
                else:
                    confidences.append(0)
            except:
                confidences.append(0)
        
    return confidences

def _median_kmeans(stylo_vectors, k):
    '''
    Run kmeans.  Find the median in the largest cluster.  Calculate confidences of all data points
    as the (normalized) distance from the median.
    '''
    if not (len(stylo_vectors) and len(stylo_vectors[0]) == 1):
        raise ValueError("Cluster method '_median_kmeans' can only handle 1-dimensional stylometric vectors.")
        return
    feature_mat = array(stylo_vectors)
    normalized_features = whiten(feature_mat)
    centroids, assigned_clusters = kmeans2(normalized_features, k, minit = 'points')

    # find largest cluster, and call it the "non-plagiarized" cluster
    non_plag_cluster = Counter(assigned_clusters).most_common()[0][0] # non-plag is largest cluster
    non_plag_vectors = [x for i, x in enumerate(stylo_vectors, 0) if assigned_clusters[i] == non_plag_cluster]
    # get median of non-plagiarized cluster
    non_plag_vectors_copy = non_plag_vectors[:]
    non_plag_vectors_copy.sort()
    median = _get_list_median(non_plag_vectors_copy)
    # find max dist from median and build confidences
    max_dist = float(max([abs(vec[0] - median) for vec in stylo_vectors]))
    if max_dist > 0:
        confidences = [abs(vec[0] - median) / max_dist for vec in stylo_vectors]
    else:
        confidences = [0 for vec in stylo_vectors]
    return confidences

def _median_simple(stylo_vectors):
    '''
    Given a list of 1-dimensional stylo_vectors, generated confidences in the following way:
    1. Find median of stylo_vectors
    2. Find max distance from median
    3. Construct confidences for points by taking their percentage of the max distance from median
    '''
    length = len(stylo_vectors)
    if length > 0 and len(stylo_vectors[0]) == 1:
        stylo_vectors_copy = stylo_vectors[:]
        stylo_vectors_copy.sort()  
        median = _get_list_median(stylo_vectors_copy)
        max_dist = float(max(median - stylo_vectors_copy[0][0], stylo_vectors_copy[-1][0] - median))
        if max_dist > 0:
            confidences = [abs(x[0] - median) / max_dist for x in stylo_vectors]
        else:
            confidences = [0 for x in stylo_vectors]
        return confidences
    else:
        raise ValueError("Cluster method 'median_simple' can only handle 1-dimensional stylometric vectors.")

def _get_list_median(vectors):
    '''
    Helper function that returns the median of the given SORTED vector list.
    '''
    length = len(vectors)
    if length % 2 == 0:
        return (vectors[length / 2][0] + vectors[(length / 2) - 1][0]) / 2.0
    else:
        return float(vectors[length / 2][0])

def _nn_confidences(items):
    from classify import NeuralNetworkConfidencesClassifier
    classifier = NeuralNetworkConfidencesClassifier()
    return classifier.nn_confidences(items)

def _combine_confidences(feature_vectors, feature_confidence_weights=None):
    weights_sum = float(sum(feature_confidence_weights))
    # "normalize" (I don't know if that's the right word) the weights, and make sure none are equal to 0
    feature_confidence_weights = [max(0.00001, x/weights_sum) for x in feature_confidence_weights]
    confidence_vectors = []
    for fi in xrange(len(feature_vectors[0])):
        single_feature_vecs = [[x[fi]] for x in feature_vectors]
        confs = _kmeans(single_feature_vecs, 2)
        for i, confidence in enumerate(confs, 0):
            if len(confidence_vectors) <= i:
                confidence_vectors.append([])
            confidence_vectors[i].append(confidence * feature_confidence_weights[fi])
            
    confidences = []
    for vec in confidence_vectors:
        confidences.append(min(sum(vec), 1))
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
    #print 'assigned_clusters are: (before knowing which group is nonplagiarized)', assigned_clusters    
    '''feature_mat = array(stylo_vectors)
    # Normalizes by column
    normalized_features = whiten(feature_mat)
    #Noah's version
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
    '''
    # Get confidences
    # TODO: Develop a real notion of confidence for hmm clustering
    plag_cluster = Counter(assigned_clusters).most_common()[-1][0]
    
    cluster_assign = [1 if x == plag_cluster else 0 for x in assigned_clusters]
    print cluster_assign
    confidences = hmm.get_confidences1(stylo_vectors, centroids, cluster_assign)
    #return cluster_assign , confidences
    return confidences

def two_normal_test(n, spacing):
    first = [[random.normal()] for x in range(n)]
    second = [[random.normal(spacing)] for x in range(n)]

    for c in ["kmeans", "agglom", "hmm"]:
        print c, cluster(c, 2, first + second)

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

    for method in ["kmeans", "agglom", "hmm", "kmedians"]:
        print method, cluster(method, 2, fs_obvious)
#    print cluster("kmeans", 2, fs_obvious)
#    print cluster("agglom", 2, fs_obvious)
#    print cluster("hmm", 2, fs_obvious)
#    print "hi", cluster("kmedians", 2, fs_obvious)

    print cluster("kmeans", 2, fs)
    print cluster("agglom", 2, fs)
    print cluster("hmm", 2, fs)
    print cluster("kmedians", 2, fs)

def _all_clusters_all_features():
    import plagcomps.evaluation.intrinsic as ev
    clusterings = [
        (ev.cluster, "simple_median", ("median_simple", None)),
        (ev.cluster, "kmeans_median", ("median_kmeans", 2)),
        (ev.cluster, "kmedians", ("kmedians", 2)),
        (ev.cluster, "kmeans", ("kmeans", 2)),
        (ev.cluster, "agglom", ("agglom", 2)),
        (ev.cluster, "hmm", ("hmm", 2)),
        #(ev.cluster, "outlier", ("outlier", 2))  was crashing
        (ev.cluster, "kmeans", ("kmeans", 3)),
        (ev.cluster, "kmeans", ("kmeans", 4)),
        (ev.cluster, "kmeans", ("kmeans", 5))
    ]

    unique_features = []
    
    for char_feature in [
        "punctuation_percentage",
    ]:
        for char_modifier in [
            "", "avg(", "std(", "avg(avg(", "avg(std(", "avg(avg(avg(", "avg(avg(std("
        ]:
            unique_features.append(char_modifier + char_feature + ")" * char_modifier.count("("))

    for word_feature in [
        "num_chars",
        "average_syllables_per_word",
        "stopword_percentage",
        "syntactic_complexity", 
        "avg_external_word_freq_class", 
        "avg_internal_word_freq_class", 
    ]:
        for word_modifier in [
            "", "avg(", "std(", "avg(avg(", "avg(std("
        ]:
            unique_features.append(word_modifier + word_feature + ")" * word_modifier.count("("))

    for sentence_feature in [
        "flesch_reading_ease",
        "yule_k_characteristic",
        "honore_r_measure",
        "gunning_fog_index",
    ]:
        for sentence_modifier in [
            "", "avg(", "std("
        ]:
            unique_features.append(sentence_modifier + sentence_feature + ")" * sentence_modifier.count("("))

    print "running on features:", unique_features

    for feature in unique_features:
        if feature != "yule_k_characteristic": 
            print "running clusterings on", feature
            ev.compare_cluster_methods(feature, 200, clusterings)

if __name__ == "__main__":
    #_test()
    _all_clusters_all_features() 
