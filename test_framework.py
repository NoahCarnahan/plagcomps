import datetime
import xml.etree.ElementTree as ET
import time

import sklearn.metrics
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as pyplot

from cluster import StylometricCluster
import feature_extractor
import dbconstants

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

def populate_database(atom_type, num):
    '''
    Populates the database with the first num training files parsed with the given atom_type.
    This method populates all features.
    '''
    
    session = Session()
    
    test_file_listing = file('corpus_partition/training_set_files.txt')
    all_test_files = [f.strip() for f in test_file_listing.readlines()]
    test_file_listing.close()
    first_test_files = all_test_files[:num]
    
    features = ['averageSentenceLength', 'averageWordLength', 'get_avg_word_frequency_class','get_punctuation_percentage','get_stopword_percentage']
    
    count = 0
    for doc in first_test_files:
        count += 1
        if DEBUG:
            print "On document", count
        d = _get_reduced_docs(atom_type, [doc], session)[0]
        d.get_feature_vectors(features, session)

    session.close()

def evaluate_n_documents(features, cluster_type, k, atom_type, n):
    test_file_listing = file('corpus_partition/training_set_files.txt')
    all_test_files = [f.strip() for f in test_file_listing.readlines()]
    test_file_listing.close()
    first_test_files = all_test_files[:n]
    return evaluate(features, cluster_type, k, atom_type, first_test_files)

def evaluate(features, cluster_type, k, atom_type, docs):
    '''
    Returns a variety of statistics (which ones exactly TBD) telling us how the tool performs with
    the given parameters.
    '''
    session = Session()
    
    reduced_docs = _get_reduced_docs(atom_type, docs, session)
    plag_likelyhoods = []
    
    count = 0
    for d in reduced_docs:
        count += 1
        if DEBUG:
            print "On document", d, ". The", count, "th document."
        c = _cluster(d.get_feature_vectors(features, session), cluster_type, k)
        plag_likelyhoods.append(c)
    
    roc_path, roc_auc = _roc(reduced_docs, plag_likelyhoods, features, cluster_type, k, atom_type)
    
    session.close()
    
    return roc_path, roc_auc

def stats_evaluate_n_documents(features, atom_type, n):   
    test_file_listing = file('corpus_partition/training_set_files.txt')
    all_test_files = [f.strip() for f in test_file_listing.readlines()]
    test_file_listing.close()
    first_test_files = all_test_files[:n]
    return feature_stats_evaluate(features, atom_type, first_test_files)

def feature_stats_evaluate(features, atom_type, docs):
    session = Session()
    
    document_dict = {}
    percent_paragraphs = .10
    same_doc_outfile = open('test_features_on_self.txt', 'w')
    different_doc_outfile = open('test_features_on_different.txt', 'w')

    reduced_docs = _get_reduced_docs(atom_type, docs, session)
    for d in reduced_docs:
        document_dict[d.doc_name] = {}

        paragraph_feature_list = d.get_feature_vectors(features, session)
        spans = d.get_spans()
        
        # builds the dictionary for feature vectors for each non-plagiarized paragraph in each document        
        for paragraph_index in range(len(paragraph_feature_list)):
            if not d.span_is_plagiarized(spans[paragraph_index]):
                for feature_index in range(len(features)):
                    # ask zach/noah what this line does
                    document_dict[d.doc_name][features[feature_index]] = document_dict[d.doc_name].get(features[feature_index], [])+ [paragraph_feature_list[paragraph_index][feature_index]]

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

def _get_reduced_docs(atom_type, docs, session):
    '''
    Returns ReducedDoc objects for the given atom_type for each of the strings in docs. docs is a
    list of paths. This function retrieves the corresponding ReducedDocs from the database if they
    exist.
    '''
    reduced_docs = []
    for doc in docs:
        try:
            r = session.query(ReducedDoc).filter(and_(ReducedDoc.doc_name == doc, ReducedDoc.atom_type == atom_type)).one()
        except sqlalchemy.orm.exc.NoResultFound, e:
            r = ReducedDoc(doc, atom_type)
            session.add(r)
            session.commit()
        reduced_docs.append(r)
        
    return reduced_docs

def _cluster(feature_vectors, cluster_type, k):
    s = StylometricCluster()
    if cluster_type == "kmeans":
        return s.kmeans(feature_vectors, k)
    elif cluster_type == "agglom":
        return s.agglom(feature_vectors, k)
    elif cluster_type == "hmm":
        return s.hmm(feature_vectors, k)
    else:
        raise ValueError("Unacceptable cluster_type. Use 'kmeans', 'agglom', or 'hmm'.")

def _roc(reduced_docs, plag_likelyhoods, features = None, cluster_type = None, k = None, atom_type = None):
    actuals = []
    confidences = []
    
    for doc_index in xrange(len(reduced_docs)):
        doc = reduced_docs[doc_index]
        spans = doc.get_spans()
        for span_index in xrange(len(spans)):
            span = spans[span_index]
            actuals.append(1 if doc.span_is_plagiarized(span) else 0)
            confidences.append(plag_likelyhoods[doc_index][span_index])
    
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
    doc_name = Column(String)
    _full_path = Column(String)
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
    
    def __init__(self, name, atom_type):
    
        self.doc_name = name # doc_name example: '/part1/suspicious-document00536'
        base_path = "/copyCats/pan-plagiarism-corpus-2009/intrinsic-detection-corpus/suspicious-documents"
        self._full_path = base_path + self.doc_name + ".txt"
        self._full_xml_path = base_path + self.doc_name + ".xml"
        self.atom_type = atom_type
        self.timestamp = datetime.datetime.now()
        self.version_numer = 1
        
        # NOTE: because feature_evaluator.get_specific_features doesn't actually return a
        # passage object for each span, we can't set self._spans until that has been run.
        # I don't like it though, because we have to raise an error now if self.get_spans()
        # is called before any feature_vectors have been calculated.
        
        # set self._spans
        #f = open(self._full_path, 'r')
        #text = f.read()
        #f.close()
        #self._spans = feature_extractor.get_spans(text, self.atom_type)
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
        return "<ReducedDoc('%s','%s')>" % (self.doc_name, self.atom_type)
    
    def get_feature_vectors(self, features, session):
        '''
        Returns a list of feature vectors for each passage in the document. The components
        of the feature vectors are in order of the features list.
        '''
        return zip(*[self._get_feature_values(x, session) for x in features])
    
    def get_spans(self):
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
        
            # TODO: The code here is more or less copy-pasted from controller. We should
            #       probably modify controller instead.
        
            # Run our tool to get the feature values and spans
            feature_evaluator = feature_extractor.StylometricFeatureEvaluator(self._full_path)
            all_passages = []
            for i in xrange(len(feature_evaluator.getAllByAtom(self.atom_type))):
                passage = feature_evaluator.get_specific_features([feature], i, i + 1, self.atom_type)
                if passage != None:
                    all_passages.append(passage)
                #else:
                #    raise Exception("This should never happen.")
            feature_values = []
            for p in all_passages:
                feature_values.append(p.features.values()[0])            
            
            # Build self.spans
            spans = []
            for p in all_passages:
                spans.append([p.start_word_index, p.end_char_index])
                #TODO: Do we really want these "snapped out" spans to be the spans that are saved? I think not...
            if self._spans:
                assert(self._spans == spans)
                #for i in range(len(self._spans)):
                #    if self._spans[i] != spans[i]:
                #        print i, self._spans[i], spans[i]
                #        assert(False)
            self._spans = spans
                
            
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

# an Engine, which the Session will use for connection resources
url = "postgresql://%s:%s@%s" % (dbconstants.username, dbconstants.password, dbconstants.dbname)
engine = sqlalchemy.create_engine(url)
# create tables if they don't already exist
Base.metadata.create_all(engine)
# create a configured "Session" class
Session = sessionmaker(bind=engine)

def _test():
    
    session = Session()
    #rs = session.query(ReducedDoc).filter(ReducedDoc.atom_type == "paragraph").all()
    rs =  _get_reduced_docs("paragraph", ["/part1/suspicious-document00536", "/part1/suspicious-document01957", "/part2/suspicious-document03297"], session)
    for r in rs:
        print r.get_feature_vectors(['averageSentenceLength', 'averageWordLength', 'get_avg_word_frequency_class','get_punctuation_percentage','get_stopword_percentage'], session)
    session.close()

def _populate_EVERYTHING():

    test_file_listing = file('corpus_partition/training_set_files.txt')
    all_test_files = [f.strip() for f in test_file_listing.readlines()]
    test_file_listing.close()

    session = Session()

    for doc in all_test_files:
        for atom_type in ["word","sentence", "paragraph"]:
            for feature in ['averageSentenceLength', 'averageWordLength', 'get_avg_word_frequency_class','get_punctuation_percentage','get_stopword_percentage']:
                d = _get_reduced_docs(atom_type, [doc], session)[0]
                print "Calculating", feature, "for", str(d), str(datetime.datetime.now())
                d.get_feature_vectors([feature], session)
    session.close()

if __name__ == "__main__":
    #_populate_EVERYTHING()

    #populate_database("sentence", 100)
    
    #test_file_listing = file('corpus_partition/training_set_files.txt')
    #all_test_files = [f.strip() for f in test_file_listing.readlines()]
    #test_file_listing.close()
    #first_test_files = all_test_files[:26]
    #print evaluate(['averageSentenceLength', 'averageWordLength', 'get_avg_word_frequency_class'], "kmeans", 2, "word", first_test_files)
    
    #print evaluate_n_documents(['get_avg_word_frequency_class'], "hmm", 2, "paragraph", 100)
    stats_evaluate_n_documents(['get_avg_word_frequency_class'], "paragraph", 100)
