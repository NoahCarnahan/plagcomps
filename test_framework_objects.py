from sqlalchemy import Column, Sequence, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm.collections import attribute_mapped_collection
from sqlalchemy.dialects.postgresql import ARRAY

Base = declarative_base()

class ReducedDoc(Base):
    
    __tablename__ = "reduced_doc"
    
    id = Column(Integer, Sequence("reduced_document_id_seq"), primary_key=True)
    doc_name = Column(String)
    atom_type = Column(String)
    features = relationship("PassageValues", 
                collection_class=attribute_mapped_collection('feature_name'),
                cascade="all, delete-orphan")
    features = relationship(ARRAY,)
    timestamp = Column(DateTime)
    version_number = Column(Integer)
    
    def __init__(self, name, atom_type):
        self.doc_name = name
        self.atom_type = atom_type
    
    def __repr__(self):
        return "<ReducedDoc('%s','%s')>" % (self.doc_name, self.atom_type)


class PassageValues(Base):
    __tablename__ = 'passage_values'
    
    id = Column(Integer, Sequence("passage_values_id_seq"), primary_key=True)
    reduced_doc_id = Column(Integer, ForeignKey('reduced_doc.id'), nullable=False)
    feature_name = Column(String)
    values = ARRAY(Float)


if __name__ == "__main__":
    r = ReducedDoc("part1/foo", "sentence")
    print r
    print r.doc_name
    print r.features
    r.features = {"a":[.1,.2,.3], "b":[.4,.3,.7]}
    print r.features

    
    
    