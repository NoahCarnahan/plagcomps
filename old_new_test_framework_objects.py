# Should there be document objects that tell us things like how many instances of plagiarism
# they have or how long they are?

# Or maybe PrcoessedDocuments should just be called Documents. Wait no that doesn't work
# Because we want to support different atom sizes. Maybe we just have feature_vector arrays
# for each atom_size?

# Maybe we should be storing the passages, is that part of what takes so long?

# TODO: Fix PerformanceEvaluation. I think I have to do a many-to-many relationship 


from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Sequence, Integer, String, Float
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship, backref

Base = declarative_base()

# Rename to ExtractedFeatures ? ExtractedFeatureVectors ? FeatureVectors?
class ProcessedDocument(Base):
    '''
    This object represents the feature vectors that have been extracted from a given
    document. This object stores what document the vectors came from, which features the
    vectors represent, and what atom type was used to split the document into passages.
    
    On this object is also stored a list of clusterings that have been made from these
    feature vectors.
    '''
    __tablename__ = "processed_document"

    id = Column(Integer, Sequence("processed_document_id_seq"), primary_key=True)
    # The name of the document that was processed, e.g. part1/suspicious-435.txt
    name = Column(String)
    # The feature list that the feature vectors were created for
    features = Column(ARRAY(String))
    # The atom type that was used to split the document into passages
    atom_type = Column(String)
    # A list of feature vectors. feature_vectors[i] is the feature vector for passage i
    feature_vectors = Column(ARRAY(Float))
    # A list of ClusteredFeatures that have been created by clustering the feature_vectors of this doc
    clusters = relationship("ClusteredFeatures", backref="processed_document")

    def __init__(self, doc_name, features, atom_type, feature_vectors):
        self.name = doc_name
        self.features = sorted(features)
        self.atom_type = atom_type
        feature_vectors = feature_vectors
    
    def __repr__(self):
        return "<ProcessedDocument('%s','%s', '%s')>" % (self.name, self.features, self.atom_type)

# Rename to Clustering ?
class ClusteredFeatures(Base):
    '''
    Represents a clustering of the feature vectors that came from the given
    processed_document.
    '''
    # For all clustering methods can we infer cluster from centroids and feature vector?
    # This object also assumes two clusters. Should we extend support for more clusters?
    
    __tablename__ = "clustered_features"
    
    
    id =                Column(Integer, Sequence("clustered_features_id_seq"), primary_key=True)
    processed_document_id = Column(Integer, ForeignKey('processed_document.id'))
    clustering_type =   Column(String)
    plag_centroid =     Column(ARRAY(Float))
    notplag_centroid =  Column(ARRAY(Float))
    
    def __init__(self, doc, clustering_type, plag_c, notplag_c):
        self.processed_document = doc
        self.clustering_type = clustering_type
        self.plag_centroid = plag_c
        self.notplag_centroid = notplag_c

    def __repr__(self):
        return "<ClusteredFeatures('%s','%s')>" % (self.processed_document, self.clustering_type)

# Rename to Evaluation?
class PerformanceEvaluation(Base):
    '''
    This object contains various metrics that have been used to evaluate the quality of
    using a certain feature set, clustering method, and atom type on a given set of
    documents
    '''
    # So, should this thing store featureSet, clusteringMethod, atomType and documents on
    # its self? These can all be pulled from clustered_features, but it may be slow and
    # unnecessarily confusing.
    # zomg... Do all these things make up a multi primary key? If only I knew more...
    
    __tablename__ = "performace_evaluation"
    
    id =                Column(Integer, Sequence("performance_evaluation_id_seq"), primary_key=True)
    clustered_features= relationship("ClusteredFeatures")
    roc_figure =        Column(String)
    au_roc =            Column(String)
    
    def __init__(self, clustered_features, roc_fig_path, au_roc):
        self.clustered_features = clustered_features
        self.roc_figure = roc_fig_path
        self.au_roc = au_roc
    
    def __repr__(self):
        return "<PerformanceEvaluation('%s','%s')>" % (self.roc_figure, self.au_roc)


if __name__ == "__main__":
    d = ProcessedDocument("foo", ["bar", "baz"], "sent", [[.4, .2],[.4,.1],[.8,.3]])
    print d
    c = ClusteredFeatures(d, "ag", [.1,.1], [.2,.2])
    c2 = ClusteredFeatures(d, "k", [1,2], [3,4])
    print c
    print d.clusters
    print c.processed_document
    
    p = PerformanceEvaluation(c, "path/foo", .75)