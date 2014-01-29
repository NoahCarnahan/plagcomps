#dbexperiment.py

from ..dbconstants import username, password, dbname

import sqlalchemy
from sqlalchemy import Table, Column, Sequence, Integer, String, Text, Float
from sqlalchemy.orm import relationship, backref, sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm.collections import attribute_mapped_collection
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.associationproxy import association_proxy

Base = declarative_base()

class ATest(Base):
    
    __tablename__ = "test_a"
    
    id = Column(Integer, Sequence("atest_id_seq"), primary_key=True)
    l = Column(ARRAY(Integer, dimensions=2))
    
    def __init__(self, data):
        self.l = data

url = "postgresql://%s:%s@%s" % (username, password, dbname)
engine = sqlalchemy.create_engine(url)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine) 

session = Session()
a = ATest([[1,2, None], [3,4,5], [6, 7, None]])
session.add(a)
session.commit()