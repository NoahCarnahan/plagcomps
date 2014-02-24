from os import path as ospath
import time

from ..intrinsic.featureextraction import FeatureExtractor

from pyevolve import G1DList
from pyevolve import GSimpleGA
import pyevolve.Consts
from pyevolve import Initializators, Mutators

from plagcomps.evaluation.intrinsic import evaluate_n_documents
from plagcomps.shared.util import IntrinsicUtility
from plagcomps.evaluation.intrinsic import _get_reduced_docs
from cluster import cluster

from ..dbconstants import username
from ..dbconstants import password
from ..dbconstants import dbname

import sqlalchemy
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

import scipy
import sklearn
import sklearn.metrics
import matplotlib
import matplotlib.pyplot as pyplot

Base = declarative_base()

class FeatureVectorWeightsEvolver:

    def __init__(self, features, n, raw_features, first_doc_num, atom_type, cluster_type):
        self.num_documents = n
        self.features = features
        self.raw_features = raw_features
        self.first_doc_num = first_doc_num
        self.atom_type = atom_type
        self.cluster_type = cluster_type


    def evolve_weights(self):
        genome = G1DList.G1DList(len(self.features))
        if self.raw_features:
            genome.setParams(rangemin=0, rangemax=100.0, gauss_mu=0, gauss_sigma=5.0)
            genome.evaluator.set(self.eval_func_raw_features)
        else:
            genome.setParams(rangemin=0, rangemax=1.0, gauss_mu=0, gauss_sigma=1.0/len(self.features))
            genome.evaluator.set(self.eval_func_confidences)
        genome.initializator.set(Initializators.G1DListInitializatorReal)
        genome.mutator.set(Mutators.G1DListMutatorRealGaussian)

        ga = GSimpleGA.GSimpleGA(genome)
        ga.setElitism(False)
        ga.setPopulationSize(12)
        ga.setMinimax(pyevolve.Consts.minimaxType["maximize"])
        ga.setGenerations(100)
        ga.setMutationRate(0.7)
        ga.evolve(freq_stats=1)
        path = ospath.join(ospath.dirname(__file__), "evolved_weights/"+str(time.time())+".txt")
        f = open(path, 'w')
        f.write(str(ga.bestIndividual()))
        f.write(str(self.features) + "\n")
        f.write("atom_type: " + str(self.atom_type) + "\n")
        f.write("cluster type: " + str(self.cluster_type) + "\n")
        f.write("num_documents: " + str(self.num_documents) + "\n")
        f.close()
        print ga.bestIndividual()


    def eval_func_confidences(self, feature_weights):
        weights_sum = float(sum(feature_weights))
        # "normalize" (I don't know if that's the right word) the weights, and make sure none are equal to 0
        feature_weights = [max(0.00001, x/weights_sum) for x in feature_weights]
        IU = IntrinsicUtility()
        all_test_files = IU.get_n_training_files(n=self.num_documents, first_doc_num=self.first_doc_num, min_len=35000, pct_plag=1)
        reduced_docs = _get_reduced_docs(self.atom_type, all_test_files, session)

        actuals = []
        confidences = []

        confidence_vectors = []
        for feature, weight in zip(self.features, feature_weights):
            vi = 0
            for doc in reduced_docs:
                feature_vectors = doc.get_feature_vectors([feature], session)
                confs = cluster(self.cluster_type, 2, feature_vectors)
                for i, confidence in enumerate(confs, 0):
                    if len(confidence_vectors) <= vi:
                        confidence_vectors.append([])
                    confidence_vectors[vi].append(confidence * weight)
                    vi += 1
                    
        for doc in reduced_docs:
            for span in doc._spans:
                actual = 1 if doc.span_is_plagiarized(span) else 0
                actuals.append(actual)

        for vec in confidence_vectors:
            confidences.append(min(1, sum(vec)))

        fpr, tpr, thresholds = sklearn.metrics.roc_curve(actuals, confidences, pos_label=1)
        roc_auc = sklearn.metrics.auc(fpr, tpr)
        print 'evaluated:', roc_auc, [w for w in feature_weights]
        return roc_auc


    def eval_func_raw_features(self, feature_weights):
        feature_weights = [max(0.00001, x) for x in feature_weights]
        roc_path, roc_auc, _, _, _, _, _ = evaluate_n_documents(self.features, self.cluster_type, 2, self.atom_type, self.num_documents, save_roc_figure=False, feature_weights=feature_weights, first_doc_num=self.first_doc_num)
        print 'evaluated:', roc_auc, [w for w in feature_weights]
        return roc_auc


# an Engine, which the Session will use for connection resources
url = "postgresql://%s:%s@%s" % (username, password, dbname)
engine = sqlalchemy.create_engine(url)
# create tables if they don't already exist
Base.metadata.create_all(engine)
# create a configured "Session" class
Session = sessionmaker(bind=engine)


if __name__ == '__main__':
    session = Session()

    features = FeatureExtractor.get_all_feature_function_names()

    n=100
    first_doc_num=100
    train_raw_weights = False
    atom_type = "nchars"
    cluster_type = "kmeans"
    evolver = FeatureVectorWeightsEvolver(features, n, train_raw_weights, first_doc_num, atom_type, cluster_type)
    evolver.evolve_weights()
