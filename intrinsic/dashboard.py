from plagcomps.intrinsic.featureextraction import FeatureExtractor
from plagcomps.evaluation.intrinsic import evaluate_n_documents
from plagcomps.shared.util import IntrinsicUtility
from plagcomps.shared.util import BaseUtility
from plagcomps.dbconstants import username
from plagcomps.dbconstants import password
from plagcomps.dbconstants import dbname


import datetime
import time
import itertools
import os.path

import sqlalchemy
from sqlalchemy import Table, Column, Sequence, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import sessionmaker

DASHBOARD_VERSION = 1
DASHBOARD_WEIGHTING_FILE = 'feature_weights.txt'

Base = declarative_base()

class IntrinsicTrial(Base):
    '''
    '''
    __tablename__ = 'intrinsic_trials'

    id = Column(Integer, Sequence("intrinsic_trials_id_seq"), primary_key=True)

    # Parameters
    atom_type = Column(String)
    cluster_type = Column(String)
    features = Column(ARRAY(String))
    # Only used when features are weighted
    feature_weights = Column(ARRAY(Float))
    feature_weights_file = Column(String)
    first_doc_num = Column(Integer)
    n = Column(Integer)
    min_len = Column(Integer)

    # Metadata
    # 'intrinsic' or 'extrinsic'
    figure_path = Column(String)
    timestamp = Column(DateTime)
    version_number = Column(Integer)
    corpus = Column(String)

    # Actual results
    time_elapsed = Column(Float)
    auc = Column(Float)
   
    def __init__(self, **args):
        '''
        All arguments are wrapped as keywords in the **args argument to avoid a HUGE parameter
        list.

        Note that required arguments are accessed using ['name'] accessors, which
        will raise an error if the key is not present in the dictionary. This is on
        purpose, since these arguments are REQUIRED!

        Arguments accessed with .get('name', default_val) are optional/have default values
        that are assumed unless specified otherwise
        '''
        self.atom_type = args['atom_type']
        self.cluster_type = args['cluster_type']
        self.features = args['features']
        # Only used when features are weighted
        self.feature_weights = args.get('feature_weights', []) # either raw weights on the features or weights for the individual confidences
        self.feature_weights_file = args.get('feature_weights_file', '')
        self.first_doc_num = args.get('first_doc_num', 0)
        self.n = args['n']
        self.min_len = args.get('min_len', 0)

        # Metadata
        # 'intrinsic' or 'extrinsic'
        self.figure_path = args['figure_path']
        self.timestamp = datetime.datetime.now()
        self.version_number = args['version_number']
        self.corpus = args.get('corpus', 'intrinsic')

        # Actual results
        self.time_elapsed = args['time_elapsed']
        self.auc = args['auc']
    

def get_feature_sets():
    '''
    Returns a list containing every set of features we want to test. Since we want to test
    each feature individually, for example, we will return something like:
    [['feat1'], ['feat2'], ..., ['feat1', 'feat2']] 
    '''
    all_features = FeatureExtractor.get_all_feature_function_names()
    individual_features = [[feat] for feat in all_features]

    # Test all features as a feature set, as well
    all_sets = individual_features + [all_features]

    return all_sets

def all_k_sets_of_features(k=2):
    all_features = FeatureExtractor.get_all_feature_function_names()
    k_sets = [list(combo) for combo in itertools.combinations(all_features, k)]

    return k_sets

def run_one_trial(feature_set, atom_type, cluster_type, k, first_doc_num, n, min_len):
    '''
    Runs <evaluate_n_documents> and saves trial to DB
    '''
    session = Session()

    start = time.time()
    path, auc = evaluate_n_documents(feature_set, cluster_type, k, atom_type, n, min_len=min_len)
    end = time.time()

    time_elapsed = end - start
    version_number = DASHBOARD_VERSION
    trial_results = {
        'atom_type' : atom_type,
        'cluster_type' : cluster_type,
        'features' : feature_set,
        'first_doc_num' : first_doc_num,
        'n' : n,
        'min_len' : min_len,
        'figure_path' : os.path.basename(path),
        'version_number' : version_number,
        'time_elapsed' : time_elapsed,
        'auc' : auc
    }
    trial = IntrinsicTrial(**trial_results)
    session.add(trial)
    print 'Made a trial!'

    session.commit()
    session.close()

    return trial


def run_one_trial_weighted(feature_set, feature_set_weights, feature_weights_filename, atom_type, cluster_type, k, first_doc_num, n, min_len):
    '''
    Runs <evaluate_n_documents> using the given raw feature weights or confidence
    weights, and saves trail to DB.
    '''
    session = Session()

    start = time.time()
    if cluster_type == "combine_confidences":
        path, auc = evaluate_n_documents(feature_set, cluster_type, k, atom_type, n, min_len=min_len, feature_confidence_weights=feature_set_weights)
    else:
        path, auc = evaluate_n_documents(feature_set, cluster_type, k, atom_type, n, min_len=min_len, feature_weights=feature_set_weights)
    end = time.time()

    time_elapsed = end - start
    version_number = DASHBOARD_VERSION
    trial_results = {
        'atom_type' : atom_type,
        'cluster_type' : cluster_type,
        'features' : feature_set,
        'feature_weights' : feature_set_weights,
        'feature_weights_file' : feature_weights_filename,
        'first_doc_num' : first_doc_num,
        'n' : n,
        'min_len' : min_len,
        'figure_path' : os.path.basename(path),
        'version_number' : version_number,
        'time_elapsed' : time_elapsed,
        'auc' : auc
    }
    trial = IntrinsicTrial(**trial_results)
    session.add(trial)
    print 'Made a weighted trial!'

    session.commit()
    session.close()

    return trial


def run_all_dashboard():
    '''
    Runs through all parameter options as listed below, writing results to DB as it goes 
    '''
    feature_set_options = get_feature_sets()
    atom_type_options = [
        'nchars',
        'paragraph'
    ]

    cluster_type_options = [
        'outlier',
        'kmeans'
    ]

    # For now, test on all documents (not just "long" ones)
    min_len_options = [0]

    for feature_set, atom_type, cluster_type, min_len in \
            itertools.product(feature_set_options, atom_type_options, cluster_type_options, min_len_options):
        print feature_set, atom_type, cluster_type, min_len
        params = {
            'atom_type' : atom_type,
            'cluster_type' : cluster_type,
            'feature_set' : feature_set,
            'first_doc_num' : 0,
            'n' : 500,
            'min_len' : min_len,
            'k' : 2,
        }

        trial = run_one_trial(**params)

    run_all_weighting_schemes(atom_type_options, cluster_type_options, min_len_options)


def run_all_weighting_schemes(atom_types, cluster_types, min_len_options):
    '''
    Reads the weighting schemes from 'feature_weights.txt' and write the results to DB.
    '''
    # weighting_schemes list contains entries like (weighting_type, [feature_set], [feature_weights])
    weighting_schemes = []

    f = open(os.path.join(os.path.dirname(__file__), DASHBOARD_WEIGHTING_FILE), 'r')
    for line in f:
        if line.startswith("#") or not len(line.strip()):
            continue
        weighting_type, feature_set, weights = line.split('\t')
        feature_set = [feature.strip() for feature in feature_set.split(";")]
        weights = [float(weight.strip()) for weight in weights.split(";")]
        weighting_schemes.append((weighting_type, feature_set, weights))
    f.close()

    for scheme, atom_type, min_len in itertools.product(weighting_schemes, atom_types, min_len_options):
        used_confidence_weights = False
        for cluster_type in cluster_types:
            if scheme[0] == "confidence_weights":
                if used_confidence_weights:
                    continue
                cluster_type = "combine_confidences"
                used_confidence_weights = True

            print scheme[1], atom_type, cluster_type, min_len
            params = {
                'atom_type' : atom_type,
                'cluster_type' : cluster_type,
                'feature_set' : scheme[1],
                'feature_set_weights' : scheme[2],
                'feature_weights_filename' : DASHBOARD_WEIGHTING_FILE,
                'first_doc_num' : 0,
                'n' : 500,
                'min_len' : min_len,
                'k' : 2
            }
            print

            trial = run_one_trial_weighted(**params)


# an Engine, which the Session will use for connection resources
url = "postgresql://%s:%s@%s" % (username, password, dbname)
engine = sqlalchemy.create_engine(url)
# create tables if they don't already exist
Base.metadata.create_all(engine)
# create a configured "Session" class
Session = sessionmaker(bind=engine)

if __name__ == '__main__':
    run_all_dashboard()
