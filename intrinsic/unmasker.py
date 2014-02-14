from itertools import combinations
from ..evaluation.intrinsic import _get_reduced_docs
from ..shared.util import IntrinsicUtility, ExtrinsicUtility
from ..shared.util import BaseUtility
from ..dbconstants import username
from ..dbconstants import password
from ..dbconstants import dbname
from ..intrinsic.cluster import cluster

import sqlalchemy
from sqlalchemy import Table, Column, Sequence, Integer, String, Float, DateTime, ForeignKey, and_
from sqlalchemy.orm import relationship, backref, sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm.collections import attribute_mapped_collection
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.associationproxy import association_proxy

import matplotlib.pyplot as plt


Base = declarative_base()
plt.ion()

# an Engine, which the Session will use for connection resources
url = "postgresql://%s:%s@%s" % (username, password, dbname)
engine = sqlalchemy.create_engine(url)
# create tables if they don't already exist
Base.metadata.create_all(engine)
# create a configured "Session" class
Session = sessionmaker(bind=engine)

dataPoints = {}


def Unmask(features, cluster_type, k, atom_type, n, corpus='intrinsic', save_roc_figure=True, min_len=None, first_doc_num=0, feature_weights=None, feature_confidence_weights=None):
    first_training_files = IntrinsicUtility().get_n_training_files(n, min_len=min_len, first_doc_num=first_doc_num)

    metadata = {
        'min_len' : min_len,
        'first_doc_num' : first_doc_num
    }

    dataPoints["control"] = evaluate_confidences(features, cluster_type, k, atom_type, first_training_files)

    for i in xrange(len(features)):
    	subSet = features[:i]+features[i+1:]

    	print "REMOVED: " + features[i]

    	dataPoints[features[i]] = evaluate_confidences(subSet, cluster_type, k, atom_type, first_training_files)
    	# evaluate_confidences([f], cluster_type, k, atom_type, first_training_files)


    # for subFeatures in combinations(features, 7):
    #     print list(subFeatures)
    #     evaluate_confidences(list(subFeatures), cluster_type, k, atom_type, first_training_files)



def evaluate_confidences(features, cluster_type, k, atom_type, docs, corpus='intrinsic', save_roc_figure=True, min_len=None, first_doc_num=0, feature_weights=None, feature_confidence_weights=None, **clusterargs):
    session = Session()

    reduced_docs = _get_reduced_docs(atom_type, docs, session, corpus=corpus)

    plag_likelihoods = []
    doc_plag_assignments = {}


    count = 0
    for d in reduced_docs:
        feature_vecs = d.get_feature_vectors(features, session)
        likelihood = cluster(cluster_type, k, feature_vecs, **clusterargs)
        # print likelihood
        doc_plag_assignments[d] = likelihood
        point = classify(likelihood)

    # dataPoints[features[0]] = point
    # dataPoints[features] = float(point[0])/float(point[1])

    session.close()

    return float(point[0])/float(point[1])

def classify(confidences):
	plag = 0
	non_plag = 0
	for confidence in confidences:
		if confidence > .50:
			plag += 1
		else:
			non_plag += 1
	return (plag, non_plag)


def visualize(dataset):
	x = []
	y = []
	i = 0
	for key in dataset.keys():
		x.append(i)
		y.append(dataset[key])
		i+=1
	plt.plot(y)
	plt.axis([0, 25, 0, 1])
	plt.show(block=True)


if __name__ == "__main__":

    features = ['average_sentence_length', 'average_syllables_per_word', 'avg_external_word_freq_class', 
        'avg_internal_word_freq_class', 'evolved_feature_one', 'evolved_feature_two', 'flesch_kincaid_grade', 
        'flesch_reading_ease', 'gunning_fog_index', 'honore_r_measure', 'num_chars', 'punctuation_percentage', 
        'stopword_percentage', 'syntactic_complexity', 'syntactic_complexity_average', 'yule_k_characteristic', 
        'vowelness_trigram,C,V,C', 'vowelness_trigram,C,V,V', 'vowelness_trigram,V,V,C', 
        'vowelness_trigram,V,V,V', 'word_unigram,is', 'word_unigram,of', 'word_unigram,been', 'word_unigram,the']

        # 'pos_trigram,NN,VB,NN', 'pos_trigram,NN,NN,VB', 'pos_trigram,VB,NN,NN', 'pos_trigram,NN,IN,NP', 
        # 'pos_trigram,NN,NN,CC', 'pos_trigram,NNS,IN,DT', 'pos_trigram,DT,NNS,IN', 'pos_trigram,VB,NN,VB', 
        # 'pos_trigram,DT,NN,IN', 'pos_trigram,NN,NN,NN', 'pos_trigram,NN,IN,DT', 'pos_trigram,NN,IN,NN', 
        # 'pos_trigram,VB,IN,DT',

    # Unmask(["average_sentence_length", "pos_trigram,NNS,IN,DT", "word_unigram,of"], "kmeans", 2, "paragraph", 10)
    Unmask(features, "kmeans", 2, "paragraph", 10)
    visualize(dataPoints)