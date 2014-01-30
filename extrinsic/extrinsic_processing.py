import datetime

from ..shared.util import ExtrinsicUtility
import fingerprint_extraction
from ..tokenization import *
from ..dbconstants import username, password, dbname
import reverse_index
import sys

import sqlalchemy
from sqlalchemy import Table, Column, Sequence, Integer, String, Text, Float, DateTime, ForeignKey, and_
from sqlalchemy.orm import relationship, backref, sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.orderinglist import ordering_list
from sqlalchemy.orm.collections import attribute_mapped_collection
from sqlalchemy.dialects.postgresql import ARRAY

Base = declarative_base()

def query_fingerprint(doc, method, n, k, atom_type, session, base_path):
    '''
    Query the database to see if the Fingerprint object in question exists in the database.
    If it does, return it, otherwise create it and add it to the database.
    '''
    if method not in ["kth_in_sent", "winnow-k"]:
        k = 0
    try:
        q = session.query(Fingerprint).filter(and_(Fingerprint.document_name == doc, Fingerprint.atom_type == atom_type, Fingerprint.method == method, Fingerprint.n == n, Fingerprint.k == k))
        fp = q.one()
    except sqlalchemy.orm.exc.NoResultFound, e:
        fp = Fingerprint(doc, method, n, k, atom_type, base_path)
        session.add(fp)
        session.commit()
    return fp
    
fingerprint_id_map = {}
def query_fingerprint_from_id(fingerprint_id, session, use_map = True):
    '''
    Given a fingerprint_id, return the Fingerprint object from the database with this id.
    '''
    if use_map and fingerprint_id in fingerprint_id_map:
        return fingerprint_id_map[fingerprint_id]
    try:
        q = session.query(Fingerprint).filter(Fingerprint.id == fingerprint_id)
        fp = q.one()
        if use_map:
            fingerprint_id_map[fingerprint_id] = fp
        return fp
    except sqlalchemy.orm.exc.NoResultFound, e:
        print 'ERROR: No fingerprint with id=' + str(fingerprint_id) + ' is in database!'
        return None

def populate_database():
    '''
    Open a session and then populate the database using filename, method, n, k.
    '''
    session = Session()

    test_file_listing = file(ExtrinsicUtility.TRAINING_SUSPECT_LOC)
    all_test_files = [f.strip() for f in test_file_listing.readlines()]
    test_file_listing.close()
    
    source_file_listing = file(ExtrinsicUtility.TRAINING_SRC_LOC)
    all_source_files = [f.strip() for f in source_file_listing.readlines()]
    source_file_listing.close()

    for atom_type in ["paragraph"]: # add other atom_type s
        for method in ["kth_in_sent"]: #["full", "anchor", "kth_in_sent"]: # add other fingerprint methods
            for n in [5]: # 3 to 6
                for k in [5]:
                    counter = 0
                    for filename in all_test_files:
                        print filename, method, n, k
                        fp = query_fingerprint(filename, method, n, k, atom_type, session, ExtrinsicUtility.CORPUS_SUSPECT_LOC)
                        print fp.get_print(session)
                        counter += 1
                        if counter%1 == 0:
                            print "Progress on sources (corpus=" + str(ExtrinsicUtility.TRAINING_SUSPECT_LOC) + ": ", counter/float(len(all_source_files)), '(' + str(counter) + '/' + str(len(all_source_files)) + ')'
                    counter = 0
                    for filename in all_source_files:
                        print filename, method, n, k
                        fp = query_fingerprint(filename, method, n, k, atom_type, session, ExtrinsicUtility.CORPUS_SRC_LOC)
                        print fp.get_print(session)
                        counter += 1
                        if counter%1 == 0:
                            print "Progress on sources (corpus=" + str(ExtrinsicUtility.TRAINING_SRC_LOC) + ": ", counter/float(len(all_source_files)), '(' + str(counter) + '/' + str(len(all_source_files)) + ')'

    session.close()

class Fingerprint(Base):
    __tablename__ = "st1_fingerprint"
    id = Column(Integer, Sequence("st1_fingerprint_id_seq"), primary_key=True)
    _fingerprint = relationship("_FpSubList", order_by="_FpSubList.position", collection_class=ordering_list("position"))
    
    document_name = Column(String, index=True)
    _doc_path = Column(String)
    _doc_xml_path = Column(String)
    method = Column(String)
    n = Column(Integer)
    k = Column(Integer)
    atom_type = Column(String)
    
    timestamp = Column(DateTime)
    version_number = Column(Integer)
    
    def __init__(self, doc, selection_method, n, k, atom_type, base_path, version_number=1):
        '''
        Initialize...
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
        
        self.method = selection_method
        self.n = n
        self.k = k
        self.atom_type = atom_type
        self.timestamp = datetime.datetime.now()
        self.version_number = version_number
    
    def get_print(self, session):
        '''
        Returns the list of minutiae or the list of lists of minutiae that is the fingerprint
        for document and fingerprinting technique given when constructing this Fingerprint
        object. This method will either retrieve the print from the database if it exists
        or calculate it and store it if not.
        '''
                
        # TODO: If something modifies the returned fingerprints, will they be modified in the database too!? We definitely don't want that to happen.
        if self._fingerprint == []:

            # Get the text
            f = open(self._doc_path, 'r')
            text = f.read()
            f.close()
            if self.atom_type == "full":
                atoms = [text]
            elif self.atom_type == "paragraph":
                paragraph_spans = tokenize(text, self.atom_type)
                atoms = [text[start:end] for start, end in paragraph_spans]
            else:
                raise ValueError("Invalid atom_type! Only 'full' and 'paragraph' are allowed.")
            
            # fingerprint the text
            prnt = []
            extractor = fingerprint_extraction.FingerprintExtractor()
            for atom in atoms:
                prnt.append(extractor.get_fingerprint(atom, self.n, self.method, self.k))
            
            print "ABOUT TO POPULATE REVERSE INDEX"
            # add each minutia to the reverse index
            if 'source' in self.document_name: # don't put suspcious documents' fingerprints in the reverse index
                #atom_index = 0
                #for fingerprint in prnt:
                #    for minutia in fingerprint:
                #        ri = reverse_index._query_reverse_index(minutia, self.n, self.k, self.method, session)
                #        ri.add_fingerprint_id(self.id, atom_index, session)
                #    atom_index += 1
                print 'inserting reverse_index entries...'
                i = 0
                atom_index = 0
                for fingerprint in prnt:
                    i += 1
                    if i % 5 == 0:
                        print str(i) + '/' + str(len(prnt)),
                        sys.stdout.flush()
                    for minutia in fingerprint:
                        ri = reverse_index._query_reverse_index(minutia, self.n, self.k, self.method, session)
                        ri.add_fingerprint_id(self.id, atom_index, session)
                    atom_index += 1
                print
            
            # save the fingerprint
            self._fingerprint = [_FpSubList(sub_list) for sub_list in prnt]
            session.commit()
            
        # Return the fingerprint 
        if self.atom_type == "full":
            # Return a list of minutiae so that this behaves the same way the previous implementation did.
            return self._fingerprint[0].minutiae
        else:
            return [x.minutiae for x in self._fingerprint]

class _FpSubList(Base):
    __tablename__ = "st1_fp_sub_list"
    id = Column(Integer, Sequence("st1_fp_sub_list_id_seq"), primary_key = True)
    fingerprint_id = Column(Integer, ForeignKey("st1_fingerprint.id"))
    position = Column(Integer)
    minutiae = Column(ARRAY(Integer))
    
    def __init__(self, sub_list):
        self.minutiae = sub_list

url = "postgresql://%s:%s@%s" % (username, password, dbname)
engine = sqlalchemy.create_engine(url)
#Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

def _test():
    '''
    Fingerprints one document (or retreives it from the db if it already exists there)
    '''
    session = Session()
    fp = query_fingerprint("/part6/source-document10718", "anchor", 3, 5, "paragraph", session, "/copyCats/pan-plagiarism-corpus-2009/external-detection-corpus/source-documents")
    fp.get_print(session)

def _other_test():
    session = Session()
    
    # Get the fingerprint 

if __name__ == "__main__":
    populate_database()
