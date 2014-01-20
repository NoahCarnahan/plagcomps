# Copyright Mathieu Blondel December 2011
# License: BSD 3 clause
# Edited by plagcomps January 2014

import numpy as np
import pylab as pl

from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans as KMeansGood
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from sklearn.datasets.samples_generator import make_blobs

from operator import itemgetter

##############################################################################
# Generate sample data


class KMedians(BaseEstimator):

    def __init__(self, k, max_iter=100, random_state=0, tol=1e-4):
        self.k = k
        self.max_iter = max_iter
        self.random_state = random_state
        self.tol = tol

    def _e_step(self):
        self.labels_ = manhattan_distances(self.vectors, self.cluster_centers_).argmin(axis=1)

    def _average(self, vectors):
        return np.median(vectors, axis=0)

    def _m_step(self):
        vectors_center = None
        for center_id in range(self.k):
            center_mask = self.labels_ == center_id
            if not np.any(center_mask):
                # The centroid of empty clusters is set to the center of
                # everything
                if vectors_center is None:
                    vectors_center = self._average(self.vectors)
                self.cluster_centers_[center_id] = vectors_center
            else:
                self.cluster_centers_[center_id] = \
                    self._average(self.vectors[center_mask])

    def fit(self, vectors, y=None):
        self.vectors = vectors
        n_samples = self.vectors.shape[0]
        vdata = np.mean(np.var(self.vectors, 0))

        random_state = check_random_state(self.random_state)
        self.labels_ = random_state.permutation(n_samples)[:self.k]
        self.cluster_centers_ = self.vectors[self.labels_]

        for i in xrange(self.max_iter):
            centers_old = self.cluster_centers_.copy()

            self._e_step()
            self._m_step()

            if np.sum((centers_old - self.cluster_centers_) ** 2) < self.tol * vdata:
                break

        return self
    
    def get_confidences(self):
        sizes = {}
        for i in range(len(self.cluster_centers_)):
            condition = np.equal(self.labels_, i)
            sizes[i] = len(np.extract(condition, self.labels_))

        no_plag_cluster = sorted(sizes.iteritems(), key = itemgetter(1), reverse=True)[0]

        confidences = []
        distances = euclidean_distances(self.vectors, self.cluster_centers_[no_plag_cluster[0]])
        
        max_distance = np.amax(distances)
        for distance in distances:
            confidences.append(distance[0] / max_distance)

        return confidences

if __name__ == "__main__":
    np.random.seed(0)
    
    centers = [[10, 10], [-10, -10], [10, -10]]
    n_clusters = len(centers)
    vectors, labels_true = make_blobs(n_samples=100, centers=centers, cluster_std=12.3)
    kmedians = KMedians(k=2)
    kmedians.fit(vectors)
    print kmedians.get_confidences()
    
