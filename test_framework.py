import datetime
import xml.etree.ElementTree as ET

from cluster import StylometricCluster
from controller import Controller
import feature_extractor
import dbconstants

import sqlalchemy
from sqlalchemy import Table, Column, Sequence, Integer, String, Float, DateTime, ForeignKey, and_
from sqlalchemy.orm import relationship, backref, sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm.collections import attribute_mapped_collection
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.associationproxy import association_proxy

Base = declarative_base()


def prepopulate_database(atom_type, num):
    
    session = Session()
    
    test_file_listing = file('corpus_partition/training_set_files.txt')
    all_test_files = [f.strip() for f in test_file_listing.readlines()]
    test_file_listing.close()
    first_test_files = all_test_files[0:num]
    
    features = ['averageSentenceLength', 'averageWordLength', 'get_avg_word_frequency_class','get_punctuation_percentage','get_stopword_percentage']
    
    reduced_docs = _get_reduced_docs(atom_type, first_test_files, session)
    for d in reduced_docs:
        d.get_feature_vectors(features, session)
    
    session.close()

def evaluate(features, cluster_type, k, atom_type, docs):
    '''
    Returns a variety of statistics (which ones exactly TBD) telling us how the tool performs with
    the given parameters.
    '''
    session = Session()
    
    reduced_docs = _get_reduced_docs(atom_type, docs, session)
    plag_likelyhoods = []
    
    for d in reduced_docs:
        c = _cluster(d.get_feature_vectors(features, session), cluster_type, k)
        plag_likelyhoods.append(c)
    
    roc_path, roc_auc = _roc(reduced_docs, plag_likelyhoods, features, cluster_type, k, atom_type)
    
    session.close()
    
    return roc_path, roc_auc

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
    pyplot.title()
    pyplot.legend(loc="lower right")
    
    path = "figures/roc"+str(time.time())+".pdf"
    pyplot.savefig(path)
    data_store.store_roc(trials, path, roc_auc)
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
        f = open(self._full_path, 'r')
        text = f.read()
        f.close()
        self._spans = feature_extractor.get_spans(text, self.atom_type)
        
        # set self._plagiarized_spans
        self._plagiarized_spans = []
        tree = ET.parse(self._full_xml_path)
        for feature in tree.iter("feature"):
            if feature.get("name") == "artificial-plagiarism": # are there other types?
                start = int(feature.get("this_offset"))
                end = start + int(feature.get("this_length"))
                spans.append((start, end))
    
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
            if s[0] <= p[0] < s[1]:
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
                else:
                    raise Exception("This should never happen.")
            feature_values = []
            for p in all_passages:
                feature_values.append(p.features.values()[0])            
            
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

if __name__ == "__main__":
    # create a Session
    session = Session()
    
    # retrive my reducedDoc
    r = session.query(ReducedDoc).filter(ReducedDoc.atom_type == "paragraph").one()
    print r._features.keys()
    r.get_feature_vectors(["averageWordLength"])
    print r._features.keys()
    print r._features
    
    #session.add(r)
    #session.commit()
    
    
    
    
    #r = ReducedDoc("/part1/suspicious-document00536", "paragraph")
    
    #print r
    #r.get_feature_vectors(["averageWordLength", "averageSentenceLength"])
    #print r.span_is_plagiarized(r.get_spans()[20])
    
    #print evaluate(["averageWordLength", "averageSentenceLength"], "kmeans", 2, "paragraph", ["/part1/suspicious-document00536"])
    
    session.close()