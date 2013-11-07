from sqlalchemy import Table, Column, Sequence, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm.collections import attribute_mapped_collection
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.associationproxy import association_proxy

Base = declarative_base()

def evaluate(features, cluster_type, atom_type, docs):
    
    for doc in docs:
        # attempt to retrieve a ReducedDoc from the database by doc_name and atom_type.
        # If it exists, add it to reduced_docs. Otherwise:
        r = ReducedDoc(doc, atom_type)
        # persist r
        
        feature_vectors = r.get_feature_vectors(features)
        # run our clustering machine on the feature vector
        

class Evaluation(Base):
	
	__tablename__ = "evaluation"
	
	id = Column(Integer, Sequence("evaluation_id_seq"), primary_key=True)
	# Can we create a constraint that says there must be at least one clustering? assert in constructor?
	# Why not store just the parameters?
	# Some clustering methods are non deterministic, thus it is possible that evaluations
	# would be different even if the parameters are the same...
	# Lets store parameters and Clustering objects. That way if we devise some tests that
	# care about the clusterings and not just the parameters we are covered.
	clusterings = relationship("Clustering", secondary=Table('association', Base.metadata,
															Column('left_id', Integer, ForeignKey('evaluation.id')),
														    Column('right_id', Integer, ForeignKey('clustering.id'))
													   ))
	
	au_roc = Column(Float)
	roc_path = Column(String)
	# ... and more ...
	
	def __init__(self, clusterings):
		self.clusterings = clusterings
	
	def __repr__(self):
		return "<Evaluation()>"
	

class Clustering(Base):
    '''
    A Clustering represents the a clustering of the passages from self.reduced_doc using
    self.features and the self.strategy clustering strategy.
    '''
    
    __tablename__ = "clustering"
    
    id = Column(Integer, Sequence("clustering_id_seq"), primary_key=True)
    features = Column(ARRAY(String))
    
    reduced_doc_id = Column(Integer, ForeignKey('reduced_doc.id'))
    reduced_doc = relationship("ReducedDoc")
    
    strategy = Column(String)
    seed = Column(Integer)
    plag_centroid = Column(ARRAY(Float))
    notplag_centroid = Column(ARRAY(Float))
    assignments = Column(ARRAY(Integer))
    
    def __init__(self, reduced_doc, features, strategy):
        self.reduced_doc = reduced_doc
        self.features = features
        self.strategy = strategy
        self._perform_clustering()
    
    def _perform_clustering(self):
        # Do the clustering and set 
        #   self.plag_centroid
        #   self.notplag_centroid
        #   self.seed
        #   self.assignments
        pass
        
        
    
    # This constructor exists for testing purposes only at the moment.
    # I don't think it will be useful in the completed framework.
    def __init__(self, reduced_doc, features, strategy, plag_c, notplag_c):
        self.reduced_doc = reduced_doc
        self.features = features
        self.strategy = strategy
        self.plag_centroid = plag_c
        self.notplag_centroid = notplag_c
    
    def __repr__(self):
        return "<Clustering('%s','%s')>" % (self.features, self.strategy)
    

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
        return zip(*[self._get_feature(x) for x in features])
        
    def _get_feature(feature):
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

    
    
    