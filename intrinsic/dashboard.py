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
import cPickle
import glob
import pprint

import numpy

import sqlalchemy
from sqlalchemy import Table, Column, Sequence, Integer, String, Float, DateTime, Boolean, and_, cast
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import sessionmaker

DASHBOARD_VERSION = 3
DASHBOARD_WEIGHTING_FILENAME = 'weighting_schemes/scheme*.pkl'
printer = pprint.PrettyPrinter(indent=3)

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
    cheating = Column(Boolean)

    # Actual results
    time_elapsed = Column(Float)
    auc = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    fmeasure = Column(Float)
    granularity = Column(Float)
    overall = Column(Float)
    threshold = Column(Float)
   
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
        self.timestamp = datetime.datetime.now()
        self.version_number = args['version_number']
        self.corpus = args.get('corpus', 'intrinsic')
        self.cheating = args.get('cheating', False)

        # Actual results
        self.time_elapsed = args['time_elapsed']
        self.auc = args.get('auc', None)
        self.figure_path = args.get('figure_path', None)
        
        # Benno definitions
        self.precision = args.get('precision', None)
        self.recall = args.get('recall', None)
        self.fmeasure = args.get('fmeasure', None)
        self.granularity = args.get('granularity', None)
        self.overall = args.get('overall', None)
        self.threshold = args.get('threshold', None)
    

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

def run_one_trial(feature_set, atom_type, cluster_type, k, first_doc_num, n, min_len, cheating, eval_method='roc'):
    '''
    Runs <evaluate_n_documents> and saves trial to DB
    '''
    session = Session()

    version_number = DASHBOARD_VERSION
    trial_results = {
        'atom_type' : atom_type,
        'cluster_type' : cluster_type,
        'features' : feature_set,
        'first_doc_num' : first_doc_num,
        'n' : n,
        'min_len' : min_len,
        'version_number' : version_number
    }
    if eval_method == 'roc':
        start = time.time()
        path, auc = evaluate_n_documents(feature_set, cluster_type, k, atom_type, n, min_len=min_len, cheating=cheating, eval_method=eval_method)
        end = time.time()
        time_elapsed = end - start

        further_params = {
            'time_elapsed' : time_elapsed,
            'auc' : auc,
            'figure_path' : os.path.basename(path),
            'cheating' : cheating
        }
        trial_results.update(further_params)
        trial = IntrinsicTrial(**trial_results)
        session.add(trial)

    elif eval_method == 'prec_recall':
        start = time.time()
        thresh_prec_avgs, thresh_recall_avgs, thresh_fmeasure_avgs, thresh_granularity_avgs, thresh_overall_avgs = \
            evaluate_n_documents(feature_set, cluster_type, k, atom_type, n, min_len=min_len, cheating=cheating, eval_method=eval_method)
        end = time.time()
        time_elapsed = end - start

        for thresh in thresh_prec_avgs.keys():
            precision = thresh_prec_avgs[thresh]
            recall = thresh_recall_avgs[thresh]
            fmeasure = thresh_fmeasure_avgs[thresh]
            granularity = thresh_granularity_avgs[thresh]
            overall = thresh_overall_avgs[thresh]

            further_params = {
                'threshold' : thresh,
                'time_elapsed' : time_elapsed,
                'precision' : precision,
                'recall' : recall,
                'fmeasure' : fmeasure,
                'granularity' : granularity,
                'overall' : overall
            }
            # Thanks to http://stackoverflow.com/questions/6005066/adding-dictionaries-together-python
            one_trial_params = dict(trial_results, **further_params)
            # print 'Would populate with:'
            # printer.pprint(one_trial_params)

            # print '-'*40
            trial = IntrinsicTrial(**one_trial_params)
            session.add(trial)
            print 'Made a trial!'

    session.commit()
    session.close()


def run_one_trial_weighted(feature_set, feature_set_weights, feature_weights_filename, atom_type, cluster_type, k, first_doc_num, n, min_len, cheating):
    '''
    Runs <evaluate_n_documents> using the given raw feature weights or confidence
    weights, and saves trail to DB.
    '''
    session = Session()

    start = time.time()
    if cluster_type == "combine_confidences":
        path, auc, _, _, _, _, _ = evaluate_n_documents(feature_set, cluster_type, k, atom_type, n, min_len=min_len, feature_confidence_weights=feature_set_weights, cheating=cheating)
    else:
        path, auc, _, _, _, _, _ = evaluate_n_documents(feature_set, cluster_type, k, atom_type, n, min_len=min_len, feature_weights=feature_set_weights, cheating=cheating)
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
        'auc' : auc,
        'cheating' : cheating
    }
    trial = IntrinsicTrial(**trial_results)
    session.add(trial)
    print 'Made a weighted trial!'

    session.commit()
    session.close()

    return trial


def run_all_dashboard(num_files, cheating=False, feature_set=None, eval_method='roc'):
    '''
    Runs through all parameter options as listed below, writing results to DB as it goes 
    '''
    if feature_set:
        feature_set_options = feature_set
    else:
        feature_set_options = get_feature_sets()

    atom_type_options = [
        'nchars',
        # 'paragraph'
    ]

    cluster_type_options = [
        'outlier',
        # 'kmeans'
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
            'n' : num_files,
            'min_len' : min_len,
            'k' : 2,
            'cheating' : cheating
        }

        trial = run_one_trial(eval_method=eval_method, **params)

    # run_all_weighting_schemes(num_files, atom_type_options, cluster_type_options, min_len_options, cheating)


def run_all_weighting_schemes(num_files, atom_types, cluster_types, min_len_options, cheating):
    '''
    Reads the weighting schemes from 'feature_weights.txt' and write the results to DB.
    '''
    # weighting_schemes list contains entries like (weighting_type, [feature_set], [feature_weights]) (weighting_type = {confidence_weights, raw_weights})
    weighting_schemes = []

    weighting_scheme_filenames = glob.glob(os.path.join(os.path.dirname(__file__), DASHBOARD_WEIGHTING_FILENAME))
    weighting_scheme_filenames.sort()
    for filepath in weighting_scheme_filenames:
        f = open(filepath, 'rb')
        scheme = cPickle.load(f)
        scheme = (scheme[0], scheme[1], scheme[2], filepath.rsplit("/", 1)[1])
        weighting_schemes.append(scheme)
        f.close()

    for scheme, atom_type, min_len in itertools.product(weighting_schemes, atom_types, min_len_options):
        used_confidence_weights = False
        for cluster_type in cluster_types:
            if scheme[0] == "confidence_weights":
                if used_confidence_weights:
                    continue
                cluster_type = "combine_confidences"
                used_confidence_weights = True

            print scheme[1], scheme[2], scheme[3], atom_type, cluster_type, min_len
            params = {
                'atom_type' : atom_type,
                'cluster_type' : cluster_type,
                'feature_set' : scheme[1],
                'feature_set_weights' : scheme[2],
                'feature_weights_filename' : scheme[3],
                'first_doc_num' : 0,
                'n' : num_files,
                'min_len' : min_len,
                'k' : 2,
                'cheating' : cheating
            }
            print

            trial = run_one_trial_weighted(**params)

def get_latest_dashboard():
    '''
    TODO finish this -- should grab/display latest dashboard runs broken
    down by various params. Perhaps like:
    |    PARAGRAPH     |       NCHARS     |
    | kmeans | outlier | kmeans | outlier |
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

    for feature_set, atom_type, cluster_type, min_len in \
            itertools.product(feature_set_options, atom_type_options, cluster_type_options, min_len_options):
        print feature_set, atom_type, cluster_type, min_len
        try:
            q = session.query(IntrinsicTrial).filter(
                and_(IntrinsicTrial.atom_type == atom_type,
                     IntrinsicTrial.cluster_type == cluster_type,
                     IntrinsicTrial.features == feature_set,
                     IntrinsicTrial.min_len == min_len)).order_by(IntrinsicTrial.timestamp)

            latest_matching_trial = q.first()
        except sqlalchemy.orm.exc.NoResultFound, e:
            print 'Didn\'t find a trial for %s, %s, min_len = %i' % (atom_type, cluster_type, )
            print 'Using' 

def get_pairwise_results(atom_type, cluster_type, n, min_len, feature_set=None, cheating=False, write_output=False):
    '''
    Generates a table for the results of all feature pairs.
    '''
    all_features = FeatureExtractor.get_all_feature_function_names()
    if not feature_set:
        feature_set = list(itertools.combinations(all_features, 2))
        feature_set += [(x,x) for x in all_features]
    session = Session()

    values = []
    results = {}
    for feature_pair in feature_set:
        if feature_pair[0] == feature_pair[1]:
            feature_pair = [feature_pair[0]]
        trial = _get_latest_trial(atom_type, cluster_type, n, min_len, list(feature_pair), cheating, session)
        if trial:
            results[tuple(feature_pair)] = round(trial.auc, 4)
            values.append(trial.auc)
        else:
            results[tuple(feature_pair)] = "n/a"

    mean = numpy.array(values).mean()
    stdev = numpy.array(values).std()

    columns = all_features
    rows = all_features

    cells = []
    for feature_a in rows:
        row = []
        for feature_b in columns:
            if feature_a == feature_b:
                row.append(results[tuple([feature_a])])
            else:
                if (feature_a, feature_b) in results:
                    row.append(results[(feature_a, feature_b)])
                elif (feature_b, feature_a) in results:
                    row.append(results[(feature_b, feature_a)])
                else:
                    row.append('???')
        cells.append(row)

    # Is html table the best way to view it?
    html = '<html><head></head><body>'
    html += '<h1>Pairwise Feature Results</h1>'
    html += '<p>DASHBOARD_VERSION = ' + str(DASHBOARD_VERSION) + '</p>'
    html += '<p>cheating = ' + str(cheating) + '</p>'
    html += '<p>atom_type = ' + str(atom_type) + '</p>'
    html += '<p>cluster_type = ' + str(cluster_type) + '</p>'
    html += '<p>n >= ' + str(n) + '</p>'
    html += '<p>min_len = ' + str(min_len) + '</p>'
    html += '<p>auc mean = ' + str(round(mean, 4)) + ', stdev = ' + str(round(stdev, 4)) + '</p>'
    html += '<table border="1">'
    html += '<tr>'
    html += '<td></td>'
    for feature in columns:
        html += '<td style="font-size: 0.7em">' + feature + '</td>'
    html += '</tr>'
    for i, feature_a in enumerate(rows, 0):
        html += '<tr>'
        html += '<td>' + feature_a + '</td>'
        for j, feature_b in enumerate(columns, 0):
            # set bg color of table cell to help visualize good features
            if type(cells[i][j]) == float:
                val = cells[i][j]
                z_score = (val - mean) / stdev
                if z_score > 3:
                    bgcolor = '#00FF00'
                elif z_score > 2:
                    bgcolor = '#AAFFAA'
                elif z_score > 1:
                    bgcolor = '#DDFFDD'
                elif z_score > -1:
                    bgcolor = '#FFFFFF'
                elif z_score > -2:
                    bgcolor = '#FFDDDD'
                elif z_score > -3:
                    bgcolor = '#FFAAAA'
                else:
                    bgcolor = '#FF0000'
            else:
                bgcolor = '#888888'

            html += '<td style="background-color: ' + bgcolor + '">' + str(cells[i][j]) + '</td>'
        html += '</tr>'

    html += '</table></body></html>'
    
    if write_output:
        html_path = os.path.join(os.path.dirname(__file__), "../figures/dashboard_pairwise_table_"+str(DASHBOARD_VERSION)+"_"+str(time.time())+".html")
        with open(html_path, 'w') as f:
            f.write(html)
        print 'Saved pairwise feature table to ' + html_path

    return html


def measure_cheating_improvement(n, feature_set_options, min_len=0):
    '''
    Compare each method using both cheating and non-cheating.
    '''
    session = Session()

    atom_types = ['nchars']
    cluster_types = ['outlier']

    differences = []
    for atom_type, cluster_type, feature_set in itertools.product(atom_types, cluster_types, feature_set_options):
        cheating_trial = _get_latest_trial(atom_type, cluster_type, n, min_len, feature_set, True, session)
        honest_trial = _get_latest_trial(atom_type, cluster_type, n, min_len, feature_set, False, session)
        print cheating_trial, honest_trial
        if not (cheating_trial and honest_trial):
            print 'At least one of the trials is not in the database:', atom_type, cluster_type, feature_set
            continue
        diff = cheating_trial.auc - honest_trial.auc
        differences.append(diff)

    print 'Average ROC auc gain:', sum(differences) / len(differences)


def _get_latest_trial(atom_type, cluster_type, n, min_len, feature_set, cheating, session):
    '''
    Helper that queries that database for the latest version of the trials with the
    given parameters.
    '''
    try:
        q = session.query(IntrinsicTrial).filter(
            and_(IntrinsicTrial.atom_type == atom_type,
                 IntrinsicTrial.cluster_type == cluster_type,
                 IntrinsicTrial.features == cast(feature_set, ARRAY(String)),
                 IntrinsicTrial.n >= n,
                 IntrinsicTrial.min_len == min_len,
                 IntrinsicTrial.cheating == cheating)).order_by(IntrinsicTrial.timestamp)
        latest_matching_trial = q.first()
    except sqlalchemy.orm.exc.NoResultFound, e:
        print 'Didn\'t find a trial for params:', atom_type, cluster_type, feature_set, n, min_len
        latest_matching_trial = None
    return latest_matching_trial


# an Engine, which the Session will use for connection resources
url = "postgresql://%s:%s@%s" % (username, password, dbname)
engine = sqlalchemy.create_engine(url)
# create tables if they don't already exist
Base.metadata.create_all(engine)
# create a configured "Session" class
Session = sessionmaker(bind=engine)

if __name__ == '__main__':
    # measure_cheating_improvement(500, all_k_sets_of_features(k=2))
    # get_pairwise_results('nchars', 'outlier', 500, 0, cheating=True)

    # good_features = ['average_sentence_length',
    #                 'average_syllables_per_word',
    #                 'avg_external_word_freq_class',
    #                 'avg_internal_word_freq_class',
    #                 'gunning_fog_index',
    #                 'honore_r_measure',
    #                 'num_chars',
    #                 'punctuation_percentage',
    #                 'syntactic_complexity',
    #                 'vowelness_trigram,C,V,C',
    #                 'vowelness_trigram,C,V,V',
    #                 'word_unigram,of',
    #                 'word_unigram,the']
    # feature_set_options = []
    # for i in xrange(3, len(good_features)+1):
    #     for x in itertools.combinations(good_features, i):
    #         feature_set_options.append(x)
    # feature_set_options = feature_set_options[50:]

    # All pairs of features
    feature_set_options = all_k_sets_of_features(k=2)
    n = 50
    run_all_dashboard(n, feature_set=feature_set_options, cheating=False, eval_method='prec_recall')
