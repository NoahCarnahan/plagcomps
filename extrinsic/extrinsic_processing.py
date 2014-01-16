import datetime

from ..shared.util import ExtrinsicUtility
import fingerprint_extraction
from ..tokenization import *
from ..dbconstants import username, password, dbname
import reverse_index

import sqlalchemy
from sqlalchemy import Table, Column, Sequence, Integer, String, Text, Float, DateTime, ForeignKey, and_
from sqlalchemy.orm import relationship, backref, sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm.collections import attribute_mapped_collection
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.associationproxy import association_proxy

import cPickle # used for storing multidimensional python lists as fingerprints in the db

Base = declarative_base()

def _query_fingerprint(docs, method, n, k, atom_type, session, base_path):
    '''
    queries the database to see if the fingerprint in question exists in the database.
    If it does, it will be returned, otherwise it is created and added to the database.
    '''
    if method != "kth_in_sent" and method != "winnow-k":
        k = 0
    try:
        q = session.query(FingerPrint).filter(and_(FingerPrint.document_name == docs, FingerPrint.atom_type == atom_type, FingerPrint.method == method, FingerPrint.n == n, FingerPrint.k == k))
        fp = q.one()
    except sqlalchemy.orm.exc.NoResultFound, e:
        fp = FingerPrint(docs, method, n, k, atom_type, base_path)
        session.add(fp)
        session.commit()
    return fp

fingerprint_id_map = {}

def _query_fingerprint_from_id(fingerprint_id, session):
    if fingerprint_id in fingerprint_id_map:
        return fingerprint_id_map[fingerprint_id]
    try:
        q = session.query(FingerPrint).filter(FingerPrint.id == fingerprint_id)
        fp = q.one()
        fingerprint_id_map[fingerprint_id] = fp
        return fp
    except sqlalchemy.orm.exc.NoResultFound, e:
        print 'ERROR: No fingerprint with id=' + str(fingerprint_id) + ' is in database!'
        return


class FingerPrint(Base):
    '''
    this is the object that we store in the database.
    '''
    
    __tablename__ = "fingerprints"
    
    id = Column(Integer, Sequence("fingerprint_id_seq"), primary_key=True)
    document_name = Column(String, index=True)
    _doc_path = Column(String)
    _doc_xml_path = Column(String)
    method = Column(String)
    n = Column(Integer)
    k = Column(Integer)
    atom_type = Column(String)
    fingerprint = Column(Text)
    timestamp = Column(DateTime)
    version_number = Column(Integer)
    
    def __init__(self, doc, select_method, n, k, atom_type, base_path, version_number=2):
        '''
        initializes FingerPrint
        '''
        self.document_name = doc

        # (nj) hacky work around since it looks like we're passing relative
        # paths in some places and absolute paths in others
        if base_path in self.document_name:
            self._doc_path = self.document_name + ".txt"
            self._doc_xml_path = self.document_name + ".xml"
        else:
            self._doc_path = base_path + self.document_name + ".txt"
            self._doc_xml_path = base_path + self.document_name + ".xml"
        
        self.method = select_method
        self.n = n
        self.k = k
        self.atom_type = atom_type
        self.timestamp = datetime.datetime.now()
        self.version_number = version_number
        

    def __repr__(self):
        '''
        changes print representation for class FingerPrint
        '''
        return "<Document(%s,%s)>" % (self.document_name, self.atom_type)
        
    def get_fingerprints(self, session):
        '''
        helper function to return fingerprints by granularity of full text vs paragraph
        '''
        if self.atom_type == "paragraph":
            return self._get_paragraph_fingerprints(session)
        else:
            return self._get_fingerprint(session)
        
    def _get_fingerprint(self, session):
        '''
        pseudo-private function to return fingerprint, returns previously calculated fingerprint if it exists.
        Otherwise, it calculates the fingerprint and returns.  Uses the full text for fingerprint calculations.
        '''
        if self.fingerprint == None:
            f = open(self._doc_path, 'r')
            text = f.read()
            f.close()

            fe = fingerprint_extraction.FingerprintExtractor()
            fingerprint = fe.get_fingerprint(text, self.n, self.method, self.k)
            
            self.fingerprint = cPickle.dumps(fingerprint)

            # add each minutia to the reverse index
            if 'source' in self.document_name: # don't put suspcious documents' fingerprints in the reverse index
                for minutia in fingerprint:
                    ri = reverse_index._query_reverse_index(minutia, self.n, self.k, self.method, session)
                    ri.add_fingerprint_id(self.id, 0, session)
            session.commit()

            return cPickle.loads(str(self.fingerprint))
        else:
            return cPickle.loads(str(self.fingerprint))
        
        
    def _get_paragraph_fingerprints(self, session):
        '''
        pseudo-private function to return fingerprint, returns previously calculated fingerprint if it exists.
        Otherwise, it calculates the fingerprint and returns.  Uses paragraphs for fingerprint calculations.
        '''
        if self.fingerprint == None:
            f = open(self._doc_path, 'r')
            text = f.read()
            f.close()
            paragraph_spans = tokenize(text, self.atom_type)

            paragraph_fingerprints = []
            fe = fingerprint_extraction.FingerprintExtractor()
            print 'fingerprinting...'
            for span in paragraph_spans:
                paragraph = text[span[0]:span[1]]
                fingerprint = fe.get_fingerprint(paragraph, self.n, self.method, self.k)
                paragraph_fingerprints.append(fingerprint)
            self.fingerprint = cPickle.dumps(paragraph_fingerprints)

            # add each minutia to the reverse index
            if 'source' in self.document_name: # don't put suspcious documents' fingerprints in the reverse index
                atom_index = 0
                print 'inserting reverse_index entries...'
                for fingerprint in paragraph_fingerprints:
                    for minutia in fingerprint:
                        ri = reverse_index._query_reverse_index(minutia, self.n, self.k, self.method, session)
                        ri.add_fingerprint_id(self.id, atom_index, session)
                    atom_index += 1
            else:
                print 'not placing', self.document_name, 'fingerprint into reverse_index'

            session.commit()
            return cPickle.loads(str(self.fingerprint))
        else:
            # uncomment these lines if you want to generate reverse indexes from the existing fingerprints in the database
            # Ask Marcus if you're unsure about this!
            # paragraph_fingerprints = cPickle.loads(str(self.fingerprint))
            # print self.id
            # i = 0
            #  # add each minutia to the reverse index
            # for fingerprint in paragraph_fingerprints:
            #     print i, len(fingerprint)
            #     for minutia in fingerprint:
            #         ri = reverse_index._query_reverse_index(minutia, self.n, self.k, self.method, session)
            #         ri.add_fingerprint_id(self.id, i, session)
            #      i += 1
            # session.commit()
            # return paragraph_fingerprints
            
            return cPickle.loads(str(self.fingerprint))


def populate_database():
    '''
    Opens a session and then populates the database using filename, method, n, k.
    '''
    session = Session()

    test_file_listing = file(ExtrinsicUtility.TRAINING_SUSPECT_LOC)
    all_test_files = [f.strip() for f in test_file_listing.readlines()]
    test_file_listing.close()
    
    source_file_listing = file(ExtrinsicUtility.TRAINING_SRC_LOC)
    all_source_files = [f.strip() for f in source_file_listing.readlines()]
    source_file_listing.close()

    counter = 0
    for atom_type in ["paragraph"]:
        for method in ["full", "anchor", "kth_in_sent"]: # add other fingerprint methods
            for n in xrange(3,7):
                for k in [5]:
                    counter = 0
                    for filename in all_test_files:
                        print filename, method, n, k
                        fp = _query_fingerprint(filename, method, n, k, atom_type, session, ExtrinsicUtility.CORPUS_SUSPECT_LOC)
                        fp.get_fingerprints(session)
                        counter += 1
                        if counter%1 == 0:
                            print "Progress on suspects: ", counter/float(len(all_test_files)), '(' + str(counter) + '/' + str(len(all_test_files)) + ')'
                    counter = 0
                    for filename in all_source_files:
                        print filename, method, n, k
                        fp = _query_fingerprint(filename, method, n, k, atom_type, session, ExtrinsicUtility.CORPUS_SRC_LOC)
                        fp.get_fingerprints(session)
                        counter += 1
                        if counter%1 == 0:
                            print "Progress on sources: ", counter/float(len(all_source_files)), '(' + str(counter) + '/' + str(len(all_source_files)) + ')'

    session.close()

url = "postgresql://%s:%s@%s" % (username, password, dbname)
engine = sqlalchemy.create_engine(url)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

if __name__ == "__main__":
    #unitTest()
    populate_database()
    