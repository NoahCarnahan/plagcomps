from plagcomps.dbconstants import username
from plagcomps.dbconstants import password
from plagcomps.dbconstants import dbname
from plagcomps.shared.util import IntrinsicUtility, BaseUtility
from plagcomps.evaluation.intrinsic import ReducedDoc, _get_reduced_docs
from plagcomps.intrinsic.cluster import cluster
from plagcomps.corpus_partition.metadata import five_num_summary
from plagcomps.intrinsic.featureextraction import FeatureExtractor

import numpy as np
from sklearn.linear_model import LogisticRegression

import sqlalchemy
from sqlalchemy import and_
from sqlalchemy.orm import sessionmaker


# an Engine, which the Session will use for connection resources
url = "postgresql://%s:%s@%s" % (username, password, dbname)
engine = sqlalchemy.create_engine(url)
# create a configured "Session" class
Session = sessionmaker(bind=engine)


def train(features, cluster_type, atom_type, ntrain, start_doc=0, regularization='l2', class_weight=None, **cluster_args):
    '''
    Given a list of <features>, cluster each passage (parsed by <atom_type>)
    individually using <cluster_type> to obtain confidences that a given 
    passage was plag. according to an individual feature. 

    Train a logistic regression model (i.e. weights for each feature)
    on first <ntrain> documents, then test the model on next <ntest>
    documents

    Returns a LogisticRegression object
    '''    
    training_matrix, training_actuals = _get_feature_conf_and_actuals(features, cluster_type, atom_type, start_doc, ntrain, **cluster_args)
    
    print 'Training mat dimesions', training_matrix.shape
    print training_matrix[0:10, ]

    model = LogisticRegression(class_weight=class_weight, penalty=regularization)
    model.fit(training_matrix, training_actuals)
    
    return model

def predict(model, features, cluster_type, atom_type, start_doc, ntest, **cluster_args):
    '''
    Runs the process of clustering for each feature/calculating confidences and 
    plugs these confidences into <model>, returning the weighted confidences of plag.
    for all passages parsed from start_doc -> start_doc + ntest documents

    TODO do one document at a time
    '''
    test_matrix, actuals = _get_feature_conf_and_actuals(features, cluster_type, atom_type, start_doc, ntest)
    
    predicted = model.predict(test_matrix)
    # predict_proba returns a list of probabilities of being in class number i
    # Since we only have two classes (0 == nonplag, 1 == plag), just keep
    # the prob/confidence of plag. 
    confidences = [x[1] for x in model.predict_proba(test_matrix)]

    pos_predictions = [x for x, y in zip(confidences, actuals) if y == 1]
    neg_predictions = [x for x, y in zip(confidences, actuals) if y == 0]
    print 'for those which are pos'
    print five_num_summary(pos_predictions)
    print 'for those which are neg'
    print five_num_summary(neg_predictions)

    print 'pct. plag', sum(actuals) / float(len(actuals))
    print 'pct correct:'
    print sum([x == y for x, y in zip(predicted, actuals)]) / float(len(predicted))

    metadata = {
        'features' : features,
        'cluster_type' : cluster_type,
        'feature_selection' : True,
        'atom_type' : atom_type,
        'start_doc' : start_doc,
        'ntest' : ntest
    }

    path, auc = BaseUtility.draw_roc(actuals, confidences, **metadata)
    print path, auc

    return confidences

def compare_params():
    '''
    [('l1', 'auto', 0.59759576698869676, 'plagcomps/shared/../figures/roc1390881314.99.pdf'),
     ('l1', None, 0.60174204862821445, 'plagcomps/shared/../figures/roc1390881397.91.pdf'),
     ('l2', 'auto', 0.60095727893574291, 'plagcomps/shared/../figures/roc1390881480.62.pdf'),
     ('l2', None, 0.5977554082484301, 'plagcomps/shared/../figures/roc1390881563.36.pdf')
    ]

    '''
    features = FeatureExtractor.get_all_feature_function_names()
    features = [f for f in features if 'unigram' not in f and 'trigram' not in f]
    cluster_type = 'outlier'
    atom_type = 'paragraph' 
    start_doc = 0
    ntrain = 100
    ntest = 200

    # Process the test set once
    test_matrix, actuals = _get_feature_conf_and_actuals(features, cluster_type, atom_type, ntrain, ntest)

    # Options for Log regression
    regularization_options = ['l1', 'l2']
    class_weight_options = ['auto', None]

    results = []
    for regularization in regularization_options:
        for class_weight in class_weight_options:
            model = train(features, cluster_type, atom_type, ntrain, start_doc=start_doc, regularization=regularization, class_weight=class_weight)
            confidences = [x[1] for x in model.predict_proba(test_matrix)]
            path, auc = BaseUtility.draw_roc(actuals, confidences, combination='Using Combination')

            results.append((regularization, class_weight, auc, path))

            print results

    print results
    return results



def train_and_predict(features, cluster_type, atom_type, start_doc, ntrain, ntest, **cluster_args):
    '''
    Trains a LogisticRegression model on the documents [start_doc : start_doc + ntrain]
    Tests the model on documents [start_doc + ntrain : start_doc + ntrain + ntest] 

    Prints the trained coefficients and returns confidences
    '''
    train_start = start_doc 
    test_start = start_doc + ntrain
    model = train(features, cluster_type, atom_type, ntrain, train_start, **cluster_args)
    print 'model params', model.get_params()

    for feat_num in xrange(len(features)):
        print features[feat_num], model.coef_[0][feat_num]

    confidences = predict(model, features, cluster_type, atom_type, test_start, ntest, **cluster_args)

    return confidences

def _get_feature_conf_and_actuals(features, cluster_type, atom_type, start_doc, n, pct_plag=None, **cluster_args):
    '''
    Returns a matrix of dimension <num_passages> x <num_features> where each row holds 
    the confidence that that row was plagiarized according to each feature. In other
    words,
    mat[passage_num][feat_num] is the plag. confidence of <passage_num> according to <feat_num>

    Note that the transpose of this matrix is built below, and then transposed before returning
    '''

    first_training_files = IntrinsicUtility().get_n_training_files(n, first_doc_num=start_doc, pct_plag=pct_plag)
    session = Session()
    reduced_docs = _get_reduced_docs(atom_type, first_training_files, session)

    actuals = []
    
    # feature_conf_matrix[feat][span_index] == Conf. that <span_index>
    # was plag. according to <feat>
    # NOTE that we're ignoring document boundaries in the storage of this 
    # matrix. So <span_index> is not relative to any document
    feature_conf_matrix = [[] for i in xrange(len(features))]
    
    for doc_index in xrange(len(reduced_docs)):
        if doc_index % 10 == 0:
            print 'Working on doc number (in training corpus)', start_doc + doc_index
        doc = reduced_docs[doc_index]
        spans = doc.get_spans()

        for feat_num in xrange(len(features)):
            feat = features[feat_num]
            feature_vecs = doc.get_feature_vectors([feat], session)
            # One column, i.e. confidence values for <feat> over all passages 
            # in <doc>
            confidences = cluster(cluster_type, 2, feature_vecs, **cluster_args)
            # Use append if we care about document_num
            
            feature_conf_matrix[feat_num].extend(confidences)
            
        for span_index in xrange(len(spans)):
            span = spans[span_index]
            
            actuals.append(1 if doc.span_is_plagiarized(span) else 0)
            
    
    rotated = np.matrix(feature_conf_matrix).T

    return rotated, actuals

def _test():
    # features = [
    #     'average_syllables_per_word',
    #     'flesch_kincaid_grade',
    #     'gunning_fog_index',
    #     'honore_r_measure',
    #     'yule_k_characteristic',
    #     'avg_external_word_freq_class'
    # ]
    features = FeatureExtractor.get_all_feature_function_names()
    features = [f for f in features if 'unigram' not in f and 'trigram' not in f]
    print features
    start_doc = 0
    cluster_type = 'kmeans'
    atom_type = 'paragraph'
    ntrain = 10
    ntest = 25

    confs = train_and_predict(features, cluster_type, atom_type, start_doc, ntrain, ntest)
    
if __name__ == '__main__':
    _test()