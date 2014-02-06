# classify.py
# Alternative methods to clustering

import sys, os
from random import shuffle
import cPickle
from collections import Counter
sys.path.append('../pybrain/') # add the pybrain module to the path... TODO: actually install it.
from plagcomps.shared.util import IntrinsicUtility

from ..dbconstants import username
from ..dbconstants import password
from ..dbconstants import dbname
'''
from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection, TanhLayer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets            import SupervisedDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader
from pybrain.structure.modules import BiasUnit
'''
import scipy
import sklearn
import sklearn.metrics
import matplotlib
import matplotlib.pyplot as pyplot

from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal

import sqlalchemy
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class NeuralNetworkConfidencesClassifier:

    nn_filepath = os.path.join(os.path.dirname(__file__), "neural_networks/nn.xml")
    dataset_filepath = os.path.join(os.path.dirname(__file__), "neural_networks/dataset.pkl")

    def create_nn(self, features, num_hidden_layer_nodes):
        net = buildNetwork(len(features), num_hidden_layer_nodes, 1)
        return net

    def create_trainer(self, network, dataset):
        trainer = BackpropTrainer(network, dataset, learningrate=0.01, momentum=0.01, verbose=True)
        return trainer

    def roc(self, confidences, actuals):
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(actuals, confidences, pos_label=1)
        roc_auc = sklearn.metrics.auc(fpr, tpr)
        print 'ROC area under curve:', roc_auc
        
        # The following code is from http://scikit-learn.org/stable/auto_examples/plot_roc.html
        pyplot.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        pyplot.plot([0, 1], [0, 1], 'k--')
        pyplot.xlim([0.0, 1.0])
        pyplot.ylim([0.0, 1.0])
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        pyplot.title('Receiver operating characteristic')
        pyplot.legend(loc="lower right")
        
        #path = "figures/roc"+str(time.time())+".pdf"
        path = ospath.join(ospath.dirname(__file__), "neural_networks/roc"+str(time.time())+".pdf")
        pyplot.savefig(path)
        return path, roc_auc

    def construct_confidence_vectors_dataset(self, reduced_docs, features, session):
        from cluster import cluster
        conf_dataset = SupervisedDataSet(len(features), 1)

        confidence_vectors = []
        num_trues = 0
        for feature in features:
            vi = 0
            for doc in reduced_docs:
                feature_vectors = doc.get_feature_vectors([feature], session)
                confidences = cluster("outlier", 2, feature_vectors, center_at_mean=True, num_to_ignore=1, impurity=.2)
                for i, confidence in enumerate(confidences, 0):
                    if len(confidence_vectors) <= vi:
                        confidence_vectors.append([[], 0])
                    if doc.span_is_plagiarized(doc._spans[i]):
                        t = 1
                        num_trues += 1
                    else:
                        t = 0
                    confidence_vectors[vi][0].append(confidence)
                    confidence_vectors[vi][1] = t
                    vi += 1

        num_plagiarised = num_trues / len(features)
        print num_plagiarised

        shuffle(confidence_vectors)
        for vec in confidence_vectors:
            if vec[1] == 0:
                num_plagiarised -= 1
            if not (vec[1] == 0 and num_plagiarised <= 0):
                conf_dataset.addSample(vec[0], vec[1])

        f = open(self.dataset_filepath, 'wb')
        cPickle.dump(conf_dataset, f)
        print 'dumped dataset file'

        return conf_dataset

    def read_dataset(self):
        f = open(self.dataset_filepath, 'rb')
        return cPickle.load(f)

    def construct_and_train_nn(self, features, num_files, epochs, filepath, session):
        from plagcomps.evaluation.intrinsic import _get_reduced_docs

        IU = IntrinsicUtility()
        all_test_files = IU.get_n_training_files(n=num_files)
        reduced_docs = _get_reduced_docs("paragraph", all_test_files, session)
        
        print 'constructing datasets...'
        # dataset = self.construct_confidence_vectors_dataset(reduced_docs, features, session)
        dataset = self.read_dataset()
        training_dataset, testing_dataset = dataset.splitWithProportion(0.75)
        print 'dataset lengths:', len(dataset), len(training_dataset), len(testing_dataset)
        print

        print 'creating neural network...'
        net = self.create_nn(features, num_hidden_layer_nodes)

        print 'creating trainer...'
        trainer = self.create_trainer(net, training_dataset)

        print 'training neural network for', epochs, 'epochs...'
        trainer.trainEpochs(epochs)

        print 'writing neural network to ' + str(filepath) + '...'
        NetworkWriter.writeToFile(net, filepath)

        print 'testing neural network...'
        confidences = []
        actuals = []
        for point in testing_dataset:
            confidences.append(net.activate(point[0])[0])
            actuals.append(point[1][0])

        print 'confidences|actuals ', zip(confidences, actuals)

        print 'generating ROC curve...'
        matplotlib.use('pdf')
        path, auc = self.roc(confidences, actuals)
        print 'area under curve =', auc


    def nn_confidences(self, feature_vectors):
        '''
        Read the saved nn and run it.
        '''
        net = NetworkReader.readFrom(self.nn_filepath)
        confidences = []
        for feature_vector in feature_vectors:
            confidences.append(net.activate(feature_vector)[0])
        return confidences

# an Engine, which the Session will use for connection resources
url = "postgresql://%s:%s@%s" % (username, password, dbname)
engine = sqlalchemy.create_engine(url)
# create tables if they don't already exist
Base.metadata.create_all(engine)
# create a configured "Session" class
Session = sessionmaker(bind=engine)

if __name__ == '__main__':
    session = Session()
    features = ['average_sentence_length',
                 'average_syllables_per_word',
                 'avg_external_word_freq_class',
                 'avg_internal_word_freq_class',
                 'flesch_kincaid_grade',
                 'flesch_reading_ease',
                 'num_chars',
                 'punctuation_percentage',
                 'stopword_percentage',
                 'syntactic_complexity',
                 'syntactic_complexity_average']

    num_hidden_layer_nodes = 20
    num_files = 30
    epochs = 400
    filepath = os.path.join(os.path.dirname(__file__), "neural_networks/nn.xml")

    NN = NeuralNetworkConfidencesClassifier()
    NN.construct_and_train_nn(features, num_files, epochs, filepath, session)


