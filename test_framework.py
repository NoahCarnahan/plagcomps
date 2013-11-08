from sqlalchemy import Table, Column, Sequence, Integer, String, Float, DateTime, ForeignKey, and_
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm.collections import attribute_mapped_collection
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.associationproxy import association_proxy

Base = declarative_base()

def evaluate(features, cluster_type, atom_type, docs):
    
    reduced_docs = _get_reduced_docs(atom_type, docs)
    plag_likelyhoods = []
    
    for d in reduced_docs:
        c = cluster(d.get_feature_vectors(features), cluster_type)
        plag_likelyhoods.append(c)
    
    roc_path, roc_auc = _roc(reduced_docs, plag_likelyhoods)

def _get_reduced_docs(atom_type, docs):

    reduced_docs = []
    for doc in docs:
        try:
            r = session.query(ReducedDoc).filter(_and(ReducedDoc.doc_name == doc, ReducedDoc.atom_type == atom_type)).one()
        except NoResultFound, e:
            r = ReducedDoc(doc_name, atom_type)
        reduced_docs.append(r)
        
    return reduced_docs

def _cluster(feature_vectors, cluster_type):
    
    if cluster_type == "kmeans":
        #TODO: write me
    elif cluster_type == "agglom":
        pass
        #TODO: write me
    else:
        raise ValueError("Unacceptable cluster_type")

def _roc(reduced_docs, plag_likelyhoods):
    pass


class ReducedDoc(Base):
    '''
    The class represents a suspect document that has been reduced to passages and feature
    vectors. Features are extracted lazily. 
    '''
    
    __tablename__ = "reduced_doc"
    
    id = Column(Integer, Sequence("reduced_doc_id_seq"), primary_key=True)
    doc_name = Column(String)
    atom_type = Column(String)

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
        #self.timestamp = ...
        self.version_numer = 1
    
    def __repr__(self):
        return "<ReducedDoc('%s','%s')>" % (self.doc_name, self.atom_type)
    
    def get_feature_vectors(features):
        '''
        Returns a list of feature vectors for each passage in the document. The components
        of the feature vectors are in order of the features list.
        '''
        return zip(*[self._get_feature_values(x) for x in features])
        
    def _get_feature_values(feature):
        '''
        Returns the list of feature values for the given feature, instantiating them if
        need be.
        '''
        try:
            return self._features[feature]
        except KeyError:
            # Run our tool to get the feature values...
            # feature_values = ...
            self._features[feature] = feature_values
            # Do something to save these changes to the object?
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

    
    
    