# reverse_index.py
import datetime
from ..dbconstants import username, password, dbname

import extrinsic_processing

import sqlalchemy
from sqlalchemy import Table, Column, Sequence, Integer, String, Text, Float, DateTime, ForeignKey, and_
from sqlalchemy.orm import relationship, backref, sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm.collections import attribute_mapped_collection
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.associationproxy import association_proxy

Base = declarative_base()

reverse_indexes = {}

def _query_reverse_index(minutia, n, k, method, session):
    if method != "kth_in_sent":
        k = 0
    # key = str(minutia) + str(n) + str(k) + method
    # if key in reverse_indexes:
    #     return reverse_indexes[key]
    try:
        q = session.query(ReverseIndex).filter(and_(ReverseIndex.minutia == minutia, ReverseIndex.method == method, ReverseIndex.n == n, ReverseIndex.k == k))
        ri = q.one()
    except sqlalchemy.orm.exc.NoResultFound, e:
        ri = ReverseIndex(minutia, n, k, method)
        session.add(ri)
        session.commit()
    # reverse_indexes[key] = ri
    # if len(reverse_indexes) > 10000:
    #     reverse_indexes.clear()
    #     print 'cleared reverse_indexes cache'
    return ri

class ReverseIndex(Base):
    '''
    Entry in the reverse_index table that maps single minutiae to documents that contain it in their fingerprints.
    '''
    __tablename__ = "reverse_index"
    
    id = Column(Integer, Sequence("reverse_index_id_seq"), primary_key=True)
    minutia = Column(Integer, index=True)
    fingerprint_ids = Column(ARRAY(Integer)) # multidimensional array (fingerprint_id, atom_index)
    timestamp = Column(DateTime)
    version_number = Column(Integer)
    n = Column(Integer)
    k = Column(Integer)
    method = Column(Text, index=True)
    
    def __init__(self, minutia, n, k, method, fingerprint_ids_list=[], version_number=2):
        self.minutia = minutia
        self.fingerprint_ids = fingerprint_ids_list
        self.timestamp = datetime.datetime.now()
        self.version_number = version_number
        self.n = n
        self.k = k
        self.method = method

    def get_fingerprints_from_minutia(self, minutia):
        '''
        Returns a list of fingerprint ids that contain this minutia.
        '''
        # TODO: we're going to have to do some work to make sure the fingerprint methods match
        return self.fingerprint_ids

    def add_fingerprint_id(self, fingerprint_id, atom_index, session):
        '''
        Adds the given fingerprint_id to the fingerprint_ids list if it's not already in the list. Might be a slow check...
        Return True if the fingerprint_id was actually added; else return False.
        IMPORTANT: You need to manually call session.commit() after running this function.
        '''
        if [fingerprint_id, atom_index] not in self.fingerprint_ids:
            # can't do .append() because it won't register with sqlalchemy. One of the more infuriating things I've encountered.
            self.fingerprint_ids = self.fingerprint_ids + [[fingerprint_id, atom_index]]
            return True
        else:
            return False

def clean_reverse_index_entries():
    '''
    This scripts removes fingerprint ids that no longer exists in the database from reverse_index entries.
    We could fix this problem by using many-to-many relationships in the database, but this will require
    a lot of reworking...
    '''
    # for each reverse index ri:
    #     remove any fingerprint ids in self.fingerprint_ids that don't exist in the fingerprints table in the database
    # TODO: fix this whole function
    session = Session()
    for minutia in xrange(10000):
        print 'minutia:', minutia
        ri = _query_reverse_index(minutia, session)
        clean_ids = []
        for fid in ri.fingerprint_ids:
            if extrinsic_processing._query_fingerprint_from_id(fid, session):
                clean_ids.append(fid)
            else:
                print 'Removing', fid, 'from', minutia
        ri.fingerprint_ids = clean_ids
        session.commit()
    return


url = "postgresql://%s:%s@%s" % (username, password, dbname)
engine = sqlalchemy.create_engine(url)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

if __name__ == '__main__':
     clean_reverse_index_entries() 
