# Should there be document objects that tell us things like how many instances of plagiarism
# they have or how long they are?

class ProcessedDocument(Base):
    __tablename__ = "processed_documents"

    id =                Column(Integer, Sequence("processed_document_id_seq"), primary_key=True)
    name =              Column(String)
    features =          Column(sqlalchemy.dialects.postgresql.ARRAY(String)) # http://docs.sqlalchemy.org/en/rel_0_8/core/types.html
    atom_type =         Column(String)
    feature_vectors =   Column(sqlalchemy.dialects.postgresql.ARRAY(sqlalchemy.dialects.postgresql.ARRAY(Float))) 
    clusters =          relationship("ClusteredFeatures", order_by="ClusteredFeatures.id", backref="processed_document")
    # http://docs.sqlalchemy.org/en/rel_0_9/orm/tutorial.html#building-a-relationship

    def __init__(self, doc_name, features, atom_type, feature_vectors):
        self.name = doc_name
        self.features = sorted(features)
        self.atom_type = atom_type
        feature_vectors = feature_vectors
    
    def __repr__(self):
        return "<ProcessedDocument('%s','%s', '%s')>" % (self.name, self.features, self.atom_type)

class ClusteredFeatures(Base):
    '''
    Represents a clustering of the feature vectors in self.processed_document.
    '''

    # Do we need both document and document_id?
    id =                Column(Integer, Sequence("clustered_features_id_seq"), primary_key=True)
    processed_document= relationship("ProcessedDocument", backref=backref('clusters', order_by=id))
    document_id =       Column(Integer, ForeignKey('ProcessedDocument.id'))
    clustering_type =   Column(String)
    plag_centroid =     Column(sqlalchemy.dialects.postgresql.ARRAY(Float))
    notplag_centroid =  Column(sqlalchemy.dialects.postgresql.ARRAY(Float))
    
    def __init__(self, doc, clustering_type, plag_c, notplag_c)
        self.processed_document = doc
        self.clustering_type = clustering_type
        self.plag_centroid = plag_c
        self.notplag_centroid = notplag_c

    def __repr__(self):
        return "<ClusteredFeatures('%s','%s')>" % (self.processed_document, self.clustering_type)

class PerformanceEvaluation(Base):
    id =                Column(Integer, Sequence("performance_evaluation_id_seq", primary_key=True)
    clustered_features= relationship("ClusteredFeatures")
    roc_figure        = Column(String)
    au_roc            = Column(String)
    
    def __init__(self, clustered_features, roc_fig_path, au_roc):
        self.clustered_features = clustered_features
        self.roc_figure = roc_fig_path
        self.au_roc = au_roc
    
    def __repr__(self):
        return "<PerformanceEvaluation('%s','%s')>" % (self.roc_figure, self.au_roc)
    