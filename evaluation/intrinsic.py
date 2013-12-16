from ..intrinsic.featureextraction import FeatureExtractor
from ..shared.util import IntrinsicUtility
from ..dbconstants import username
from ..dbconstants import password
from ..dbconstants import dbname

import datetime
import xml.etree.ElementTree as ET
import time

import sklearn.metrics
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as pyplot



import sqlalchemy
from sqlalchemy import Table, Column, Sequence, Integer, String, Float, DateTime, ForeignKey, and_
from sqlalchemy.orm import relationship, backref, sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm.collections import attribute_mapped_collection
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.associationproxy import association_proxy

import scipy, random

Base = declarative_base()

DEBUG = True

def _populate_EVERYTHING():
    '''
    Populate the database with ReducedDocs for all documents, atom types, and features.
    This takes days.
    '''

    all_test_files = IntrinsicUtility().get_n_training_files()

    session = Session()

    for doc in all_test_files:
        for atom_type in ["word","sentence", "paragraph"]:
            for feature in ['averageSentenceLength', 'averageWordLength', 'get_avg_word_frequency_class','get_punctuation_percentage','get_stopword_percentage']:
                d = _get_reduced_docs(atom_type, [doc], session)[0]
                print "Calculating", feature, "for", str(d), str(datetime.datetime.now())
                d.get_feature_vectors([feature], session)
    session.close()

def populate_database(atom_type, num):
    '''
    Populate the database with the first num training files parsed with the given atom_type.
    Refer to the code to see which features it populates.
    '''
    
    session = Session()
    
    # Get the first num training files
    util = IntrinsicUtility()
    first_training_files = util.get_n_training_files(num)

    features = ['punctuation_percentage',
                'stopword_percentage',
                'average_sentence_length',
                'average_word_length',
            # Tests haven't been written for these:
                #'avg_internal_word_freq_class',
                #'avg_external_word_freq_class',
            # These are broken:
                #'pos_percentage_vector',
                #'syntactic_complexity',
                ]
    
    count = 0
    for doc in first_training_files:
        count += 1
        if DEBUG:
            print "On document", count
        d = _get_reduced_docs(atom_type, [doc], session)[0]
        d.get_feature_vectors(features, session)

    session.close()

def evaluate_n_documents(features, cluster_type, k, atom_type, n):
    '''
    Return the evaluation (roc curve path, area under the roc curve) of the first n training
    documents parsed by atom_type, using the given features, cluster_type, and number of clusters k.
    '''
    # Get the first n training files
    first_training_files = IntrinsicUtility().get_n_training_files(n)
    
    roc_path, roc_auc = evaluate(features, cluster_type, k, atom_type, first_training_files)
    
    # Store the figures in the database
    session = Session()
    f = _Figure(roc_path, "roc", roc_auc, sorted(features), cluster_type, k, atom_type, n)
    session.add(f)
    session.commit()
    session.close()
    
    return roc_path, roc_auc

def evaluate(features, cluster_type, k, atom_type, docs):
    '''
    Return the roc curve path and area under the roc curve for the given list of documents parsed
    by atom_type, using the given features, cluster_type, and number of clusters k.
    
    features is a list of strings where each string is the name of a StylometricFeatureEvaluator method.
    cluster_type is "kmeans", "hmm", or "agglom".
    k is an integer.
    atom_type is "word", "sentence", or "paragraph".
    docs should be a list of full path strings.
    '''
    # TODO: Return more statistics, not just roc curve things. 
    
    session = Session()
    
    reduced_docs = _get_reduced_docs(atom_type, docs, session)
    plag_likelihoods = []
    
    count = 0
    for d in reduced_docs:
        count += 1
        if DEBUG:
            print "On document", d, ". The", count, "th document."
        likelihood = cluster.cluster(d.get_feature_vectors(features, session), cluster_type, k)
        plag_likelihoods.append(likelihood)
    
    roc_path, roc_auc = _roc(reduced_docs, plag_likelihoods, features, cluster_type, k, atom_type)
    session.close()
    return roc_path, roc_auc



def _stats_evaluate_n_documents(features, atom_type, n):
    '''
    Does _feature_stats_evaluate(features, atom_type, doc) on the first n docuemtns of the training
    set. Returns None.
    '''
    first_training_files = IntrinsicUtility().get_n_training_files(n)
    return _feature_stats_evaluate(features, atom_type, first_training_files)

def _feature_stats_evaluate(features, atom_type, docs):
    '''
    This function writes data to two files. The data is used by an R script to generate box plots
    that show the significance of the given features within the given documents. Returns None.
    '''
    session = Session()
    
    document_dict = {}
    percent_paragraphs = .10
    same_doc_outfile = open('test_features_on_self.txt', 'w')
    different_doc_outfile = open('test_features_on_different.txt', 'w')

    reduced_docs = _get_reduced_docs(atom_type, docs, session)
    for d in reduced_docs:
        document_dict[d._short_name] = {}

        paragraph_feature_list = d.get_feature_vectors(features, session)
        spans = d.get_spans()
        
        # builds the dictionary for feature vectors for each non-plagiarized paragraph in each document        
        for paragraph_index in range(len(paragraph_feature_list)):
            if not d.span_is_plagiarized(spans[paragraph_index]):
                for feature_index in range(len(features)):
                    # ask zach/noah what this line does
                    document_dict[d._short_name][features[feature_index]] = document_dict[d._short_name].get(features[feature_index], [])+ [paragraph_feature_list[paragraph_index][feature_index]]

    for feature in features:
        for doc in document_dict:    
            # take two samples and compare them for our same_doc test, then pick one of the samples and save it for our later tests
            feature_vect = document_dict[doc][feature] 
            sample = random.sample(feature_vect, max(2, int(len(feature_vect)*percent_paragraphs*2)))
            sample_one = sample[:len(sample)/2]
            sample_two = sample[len(sample)/2:]
            print "SAMPLES:", sample_one, sample_two
            stats = scipy.stats.ttest_ind(sample_one, sample_two)
            same_doc_outfile.write(feature + "\t" + doc + "\t" + str(stats[1]) + "\r\n")
            document_dict[doc][feature] = sample_one

    for feature in features:
        for first in document_dict: 
            for second in document_dict:
                 if first != second:
                     sample_one = document_dict[first][feature]
                     sample_two = document_dict[second][feature]
                     stats = scipy.stats.ttest_ind(sample_one, sample_two)
                     different_doc_outfile.write(feature + "\t" + first + "," + second + "\t" + str(stats[1]) + "\r\n") 

    same_doc_outfile.close()
    different_doc_outfile.close()
    session.close() 



def _get_reduced_docs(atom_type, docs, session, create_new=True):
    '''
    Return ReducedDoc objects for the given atom_type for each of the documents in docs. docs is a
    list of strings like "/copyCats/pan-plagiarism-corpus-2009/intrinsic-detection-corpus/suspicious-documents/part2/suspicious-document02456.txt".
    They must be full paths. This function retrieves the corresponding ReducedDocs from
    the database if they exist. 

    Otherwise, if <create_new> is True (which it is by default), it creates new ReducedDoc objects.
    Set <create_new> to False if you don't want to wait for new ReducedDoc objects to be created.
    Useful if just messing around from the command line/doing sanity checks 
    '''
    reduced_docs = []
    for doc in docs:
        try:
            r = session.query(ReducedDoc).filter(and_(ReducedDoc.full_path == doc, ReducedDoc.atom_type == atom_type, ReducedDoc.version_number == 2)).one()
        except sqlalchemy.orm.exc.NoResultFound, e:
            if create_new:
                r = ReducedDoc(doc, atom_type)
                session.add(r)
                session.commit()
            else:
                continue
        reduced_docs.append(r)
        
    return reduced_docs

def _roc(reduced_docs, plag_likelihoods, features = None, cluster_type = None, k = None, atom_type = None):
    '''
    Generates a reciever operator characterstic (roc) curve and returns both the path to a pdf
    containing a plot of this curve and the area under the curve. reduced_docs is a list of
    ReducedDocs, plag_likelihoods is a list of lists whrere plag_likelihoods[i][j] corresponds
    to the likelihood that the jth span in the ith reduced_doc was plagiarized.
    
    The optional parameters allow for a more verbose title of the graph in the pdf document.
    
    Note that all passages have equal weight for this curve. So, if one document is considerably
    longer than the others and our tool does especially poorly on this document, it will look as
    if the tool is bad (which is not necessarily a bad thing).
    '''

    # This function was modeled in part from this example:
    # http://scikit-learn.org/0.13/auto_examples/plot_roc.html
    
    actuals = []
    confidences = []
    
    for doc_index in xrange(len(reduced_docs)):
        doc = reduced_docs[doc_index]
        spans = doc.get_spans()
        for span_index in xrange(len(spans)):
            span = spans[span_index]
            actuals.append(1 if doc.span_is_plagiarized(span) else 0)
            confidences.append(plag_likelihoods[doc_index][span_index])
    
    # actuals is a list of ground truth classifications for passages
    # confidences is a list of confidence scores for passages
    # So, if confidences[i] = .3 and actuals[i] = 1 then passage i is plagiarized and
    # we are .3 certain that it is plagiarism (So its in the non-plag cluster).
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(actuals, confidences, pos_label=1)
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    
    # The following code is from http://scikit-learn.org/stable/auto_examples/plot_roc.html
    pyplot.clf()
    pyplot.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pyplot.plot([0, 1], [0, 1], 'k--')
    pyplot.xlim([0.0, 1.0])
    pyplot.ylim([0.0, 1.0])
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    if features and cluster_type and k and atom_type:
        pyplot.title("ROC, %s, %s %s, %s" % (atom_type, cluster_type, k, features)) 
    else:
        pyplot.title('Receiver operating characteristic')
    pyplot.legend(loc="lower right")
    
    path = "figures/roc"+str(time.time())+".pdf"
    pyplot.savefig(path)
    return path, roc_auc

class ReducedDoc(Base):
    '''
    The class represents a suspect document that has been reduced to passages and feature
    vectors. Features are extracted lazily. 
    '''
    
    __tablename__ = "reduced_doc"
    
    id = Column(Integer, Sequence("reduced_doc_id_seq"), primary_key=True)
    _short_name = Column(String)
    full_path = Column(String)
    _full_xml_path = Column(String)
    atom_type = Column(String)
    _spans = Column(ARRAY(Integer))
    _plagiarized_spans = Column(ARRAY(Integer))
    

    # The miracle of features is explained here:
    # http://docs.sqlalchemy.org/en/rel_0_7/orm/extensions/associationproxy.html#proxying-dictionaries
    _features = association_proxy(
                    'reduced_doc_features',
                    'feature',
                    creator=lambda k, v:
                        _ReducedDocFeature(special_key=k, feature=v)
                )
                
    timestamp = Column(DateTime)
    version_number = Column(Integer)
    
    def __init__(self, path, atom_type):
        '''
        Initializes a ReducedDoc. No feature vectors will be calculated at instantiation time.
        get_feature_vectors triggers the lazy instantiation of these values.
        '''
        
       
        #base_path = "/copyCats/pan-plagiarism-corpus-2009/intrinsic-detection-corpus/suspicious-documents"
        
        self.full_path = path
        # _short_name example: '/part1/suspicious-document00536'
        self._short_name = "/"+self.full_path.split("/")[-2] +"/"+ self.full_path.split("/")[-1] 
        self._full_xml_path = path[:-3] + "xml"
        
        
        self.atom_type = atom_type
        self.timestamp = datetime.datetime.now()
        self.version_numer = 2
        
        # NOTE: I think the note below is outdated/different now.
        #       Now FeatureExtractor.get_feature_vectors() does return a feature for each
        #       span. So, we could set self._spans now. But, the initialization of that
        #       object does take same time, so we would want to save it probably rather
        #       than do it twice.
        # NOTE: because feature_evaluator.get_specific_features doesn't actually return a
        # passage object for each span, we can't set self._spans until that has been run.
        # I don't like it though, because we have to raise an error now if self.get_spans()
        # is called before any feature_vectors have been calculated.
        
        # set self._spans
        #f = open(self.full_path, 'r')
        #text = f.read()
        #f.close()
        #self._spans = FeatureExtractor(text).get_spans(self.atom_type)
        self._spans = None
        
        # set self._plagiarized_spans
        self._plagiarized_spans = []
        tree = ET.parse(self._full_xml_path)
        for feature in tree.iter("feature"):
            if feature.get("name") == "artificial-plagiarism": # are there other types?
                start = int(feature.get("this_offset"))
                end = start + int(feature.get("this_length"))
                self._plagiarized_spans.append((start, end))
    
    def __repr__(self):
        return "<ReducedDoc('%s','%s')>" % (self._short_name, self.atom_type)
    
    def get_feature_vectors(self, features, session):
        '''
        Returns a list of feature vectors for each passage in the document. The components
        of the feature vectors are in order of the features list.
        '''
        return zip(*[self._get_feature_values(x, session) for x in features])
    
    def get_spans(self):
        '''
        Returns a list of lists. The ith list contains the start and end characters for the ith
        passage in this document.
        '''
        if self._spans == None:
            raise Exception("See note in ReducedDoc.__init__")
        else:
            return self._spans
    
    def span_is_plagiarized(self, span):
        '''
        Returns True if the span was plagiarized (according to the ground truth). Returns
        False otherwise. A span is considered plagiarized if the first character of the
        span is in a plagiarized span.
        
        '''
        # TODO: Consider other ways to judge if an atom is plagiarized or not. 
        #       For example, look to see if the WHOLE atom in a plagiarized segment (?)
        for s in self._plagiarized_spans:
            if s[0] <= span[0] < s[1]:
                return True
        return False
        
    def _get_feature_values(self, feature, session):
        '''
        Returns the list of feature values for the given feature, instantiating them if
        need be.
        '''
        try:
            return self._features[feature]
            
        except KeyError:
            # Read the file
            f = open(self.full_path, 'r')
            text = f.read()
            f.close()
            
            # Create a FeatureExtractor
            extractor = FeatureExtractor(text)
            
            # Save self._spans
            if self._spans:
                assert(self._spans == [list(x) for x in extractor.get_spans(self.atom_type)])
            self._spans = extractor.get_spans(self.atom_type)
            
            # Save self._features
            feature_values = [tup[0] for tup in extractor.get_feature_vectors([feature], self.atom_type)]
            self._features[feature] = feature_values
            
            session.commit()
            return self._features[feature]

class _ReducedDocFeature(Base):
    '''
    This class allows for the features dictionary on ReducedDoc.
    Explained here: http://docs.sqlalchemy.org/en/rel_0_8/orm/extensions/associationproxy.html#proxying-dictionaries
    '''
    __tablename__ = 'reduced_doc_feature'
    reduced_doc_id = Column(Integer, ForeignKey('reduced_doc.id'), primary_key=True)
    feature_id = Column(Integer, ForeignKey('feature.id'), primary_key=True)
    special_key = Column(String)
    reduced_doc = relationship(ReducedDoc, backref=backref(
            "reduced_doc_features",
            collection_class=attribute_mapped_collection("special_key"),
            cascade="all, delete-orphan"
            )
        )
    kw = relationship("_Feature")
    feature = association_proxy('kw', 'feature')
    
class _Feature(Base):
    '''
    This class allows for the features dictionary on ReducedDoc.
    Explained here: http://docs.sqlalchemy.org/en/rel_0_8/orm/extensions/associationproxy.html#proxying-dictionaries
    '''
    __tablename__ = 'feature'
    id = Column(Integer, primary_key=True)
    feature = Column(ARRAY(Float))
    def __init__(self, feature):
        self.feature = feature

class _Figure(Base):
    '''
    This class allows us to store meta data about pdf figures that we create.
    '''
    
    __tablename__ = "figure"
    id = Column(Integer, Sequence("figure_id_seq"), primary_key=True)
    timestamp = Column(DateTime)
    version_number = Column(Integer)
    
    figure_path = Column(String)
    figure_type = Column(String)
    auc = Column(Float)
    
    features = Column(ARRAY(String))
    cluster_type = Column(String)
    k = Column(Integer)
    atom_type = Column(String)
    n = Column(Integer)
    
    def __init__(self, figure_path, figure_type, auc, features, cluster_type, k, atom_type, n):
        
        self.figure_path = figure_path
        self.timestamp = datetime.datetime.now()
        self.version_number = 2
        self.figure_type = figure_type
        self.auc = auc
        self.features = features
        self.cluster_type = cluster_type
        self.k = k
        self.atom_type = atom_type
        self.n = n

    
# an Engine, which the Session will use for connection resources
url = "postgresql://%s:%s@%s" % (username, password, dbname)
engine = sqlalchemy.create_engine(url)
# create tables if they don't already exist
Base.metadata.create_all(engine)
# create a configured "Session" class
Session = sessionmaker(bind=engine)

def _test():
    
    session = Session()
    
    first_training_files = IntrinsicUtility().get_n_training_files(3)
    
    rs =  _get_reduced_docs("paragraph", first_training_files, session)
    for r in rs:
        print r.get_feature_vectors(['punctuation_percentage',
                                     'stopword_percentage',
                                     'average_sentence_length',
                                     'average_word_length',], session)
    session.close()
    
if __name__ == "__main__":
    _test()
