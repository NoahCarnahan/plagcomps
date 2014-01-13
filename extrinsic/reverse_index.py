# reverse_index.py
import datetime
from ..dbconstants import username, password, dbname

import sqlalchemy
from sqlalchemy import Table, Column, Sequence, Integer, String, Text, Float, DateTime, ForeignKey, and_
from sqlalchemy.orm import relationship, backref, sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm.collections import attribute_mapped_collection
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.associationproxy import association_proxy

Base = declarative_base()

reverse_indexes = {}

def _query_reverse_index(minutia, session):
    if minutia in reverse_indexes:
        return reverse_indexes[minutia]
    try:
        q = session.query(ReverseIndex).filter(ReverseIndex.minutia == minutia)
        ri = q.one()
    except sqlalchemy.orm.exc.NoResultFound, e:
        ri = ReverseIndex(minutia)
        session.add(ri)
        # session.commit()
    reverse_indexes[minutia] = ri
    return ri

class ReverseIndex(Base):
    '''
    Entry in the reverse_index table that maps single minutiae to documents that contain it in their fingerprints.
    '''
    __tablename__ = "reverse_index"
    
    minutia = Column(Integer, primary_key=True)
    fingerprint_ids = Column(ARRAY(Integer))
    timestamp = Column(DateTime)
    version_number = Column(Integer)
    
    def __init__(self, minutia, fingerprint_ids_list=[], version_number=1):
        self.minutia = minutia
        self.fingerprint_ids = fingerprint_ids_list
        self.timestamp = datetime.datetime.now()
        self.version_number = version_number

    def get_fingerprints_from_minutia(self, minutia):
        '''
        Returns a list of fingerprint ids that contain this minutia.
        '''
        # TODO: we're going to have to do some work to make sure the fingerprint methods match
        return self.fingerprint_ids

    def add_fingerprint_id(self, fingerprint_id):
        '''
        Adds the given fingerprint_id to the fingerprint_ids list if it's not already in the list. Might be a slow check...
        Return True if the fingerprint_id was actually added; else return False.
        IMPORTANT: You need to manually call session.commit() after running this function.
        '''
        if fingerprint_id not in self.fingerprint_ids:
            self.fingerprint_ids.append(fingerprint_id)
            return True
        else:
            return False


url = "postgresql://%s:%s@%s" % (username, password, dbname)
engine = sqlalchemy.create_engine(url)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
