import datetime
import cluster.StylometricCluster

from controller import Controller
from feature_extractor import *

from sqlalchemy import Table, Column, Sequence, Integer, String, Float, DateTime, ForeignKey, and_
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm.collections import attribute_mapped_collection
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.associationproxy import association_proxy

Base = declarative_base()

def evaluate(features, cluster_type, k, atom_type, docs):
    '''
    Returns a variety of statistics (which ones exactly TBD) telling us how the tool performs with
    the given parameters.
    '''
    
    reduced_docs = _get_reduced_docs(atom_type, docs)
    plag_likelyhoods = []
    
    for d in reduced_docs:
        c = _cluster(d.get_feature_vectors(features), cluster_type, k)
        plag_likelyhoods.append(c)
    
    roc_path, roc_auc = _roc(reduced_docs, plag_likelyhoods)

def _get_reduced_docs(atom_type, docs):
    '''
    Returns ReducedDoc objects for the given atom_type for each of the strings in docs. docs is a
    list of paths. This function retrieves the corresponding ReducedDocs from the database if they
    exist.
    '''
    reduced_docs = []
    for doc in docs:
        try:
            r = session.query(ReducedDoc).filter(_and(ReducedDoc.doc_name == doc, ReducedDoc.atom_type == atom_type)).one()
        except NoResultFound, e:
            r = ReducedDoc(doc_name, atom_type)
            session.add(r)
        reduced_docs.append(r)
        
    return reduced_docs

def _cluster(feature_vectors, cluster_type, k):
    s = StylometricCluster
    if cluster_type == "kmeans":
        return s.kmeans(feature_vectors, k)
    elif cluster_type == "agglom":
        return s.agglom(feature_vectors, k)
    elif cluster_type == "hmm":
        return s.hmm(feature_vectors, k)
    else:
        raise ValueError("Unacceptable cluster_type. Use 'kmeans', 'agglom', or 'hmm'.")

def _roc(reduced_docs, plag_likelyhoods):
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
    # So, if confidences[i] = .3 and actuals[i] = 1 then passage i is plagiarised and
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
    pyplot.title('Receiver operating characteristic')
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
        self.doc_name = name
        self.atom_type = atom_type
        self.timestamp = datetime.datetime.now()
        self.version_numer = 1
        
        #TODO: init self._spans here!
        #TODO: init self._plagiarized_spans here! (see tool_tester.py line 132)
        
    
    def __repr__(self):
        return "<ReducedDoc('%s','%s')>" % (self.doc_name, self.atom_type)
    
    def get_feature_vectors(self, features):
        '''
        Returns a list of feature vectors for each passage in the document. The components
        of the feature vectors are in order of the features list.
        '''
        return zip(*[self._get_feature_values(x) for x in features])
    
    def get_spans(self):
        return self._spans
        
    
    def span_is_plagiarized(self, span):
        '''
        Returns True if the span was plagiarized (according to the ground truth).
        Returns False otherwise.
        '''
        pass
        # See tooltester line 132.
        
    def _get_feature_values(feature):
        '''
        Returns the list of feature values for the given feature, instantiating them if
        need be.
        '''
        try:
            return self._features[feature]
        except KeyError:
            # Run our tool to get the feature values...
            feature_evaluator = StylometricFeatureEvaluator(self.doc_name)
            feature_values = []
        		for i in xrange(len(feature_evaluator.getAllByAtom(self.atom_type))):
		    	passage = feature_evaluator.get_specific_features([feature], i, i + 1, self.atom_type)
		        feature_values.append(passage.features)
            self._features[feature] = feature_values
            # TODO: Do something to save these changes to the object?
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

if __name__ == "__main__":
    r = ReducedDoc("part1/foo", "sentence")
    print r
    print r.doc_name
    r._features["a"] = [.1,.2,.3]
    r._features["b"] = [.6,.3,.1]
    print r._features["a"]
    print r._features["b"]
    print r._features
    
    c1 = Clustering(r, ["a","b"], "kmeansfoo", [.1,.6], [.3,.1])
    c2 = Clustering(r, ["a","b"], "aglomfoo", [.2,.3], [.3,.1])
        
    e1 = Evaluation([c1,c2])
    print e1.clusterings
    e2 = Evaluation([c1])
    print e2.clusterings

    
    
    