import datetime

from ..shared.util import ExtrinsicUtility
import fingerprint_extraction
from ..tokenization import tokenize
from ..dbconstants import username, password, dbname

import sqlalchemy
from sqlalchemy import Table, Column, Sequence, Integer, Boolean, String, Text, Float, DateTime, ForeignKey, and_
from sqlalchemy.orm import relationship, backref, sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.orderinglist import ordering_list
from sqlalchemy.orm.collections import attribute_mapped_collection
from sqlalchemy.dialects.postgresql import ARRAY

Base = declarative_base()

def get_fingerprints(doc_name, base_path, method, n, k, atom_type, hash_indexed, session, create=True, commit=False):
    '''
    Retrieve the Fingerprint objects from the database with the given parameters. This function will
    return a Fingerprint for each atom in this document. They are NOT necessarily in order.
    New Fingerprint objects will be created if they do not already exist in the database.
    '''
    
    # WILL THIS BREAK IF ATOM_TYPE IS FULL? (tokenize might not know what to do)
    abs_path = base_path + doc_name + ".txt"
    f = open(abs_path, "r")
    text = f.read()
    f.close()
    atom_spans = tokenize(text, atom_type)
    
    fingerprints = []
    for i in range(len(atom_spans)):
        atom_text = text[atom_spans[i][0]:atom_spans[i][1]]
        fp = get_fingerprint(doc_name, base_path, method, n, k, atom_type, i, hash_indexed, session, create, commit, atom_text)
        fingerprints.append(fp)
    return fingerprints
    
def get_fingerprint(doc_name, base_path, method, n, k, atom_type, atom_number, hash_indexed, session, create=True, commit=False, atom_text = None):
    '''
    Retrieve the Fingerprint object from the database with the given parameters. If create=True,
    then a new Fingerprint will be created and returned in the event that it does not already
    exist in the database.
    '''
    
    if method not in ["kth_in_sent", "winnow-k"]:
        k = 0
        
    fp = None
    try:
        query = session.query(Fingerprint).filter(and_(Fingerprint.doc_name == doc_name, Fingerprint.method == method, Fingerprint.n == n, Fingerprint.k == k, Fingerprint.atom_type == atom_type, Fingerprint.atom_number == atom_number))
        fp = query.one()
    except sqlalchemy.orm.exc.NoResultFound, e:
        if create:
            # A note on creation:
            # The hash index can't be updated until we have an id for the Fingerprint in
            # question. Unfortunately the id isn't generated until we have added the object
            # to the session and flushed it. So, update_hash_index MUST be called immediately
            # after initialization as it does here. I don't THINK we can flush in the
            # __init__ function of Fingerprint, otherwise I would.
            fp = Fingerprint(doc_name, base_path, method, n, k, atom_type, atom_number, hash_indexed, atom_text = atom_text)
            session.add(fp)
            session.flush()
            fp.update_hash_index(session, commit)
            
            if commit:
                session.commit()
    return fp

def get_fingerprints_by_hash(hash, method, n, k, atom_type, session):
    '''
    Return a list of Fingerprint objects such that each Fingerprint contains the given
    hash in its hash_values.
    '''
    
    #q = session.query(HashIndex).filter(HashIndex.hash_value == hash).join(Fingerprint).filter(Fingerprint.id.in_(HashIndex.fingerprint_ids))
    #return q.all()
    
    try:
        q = session.query(HashIndex).filter(HashIndex.hash_value == hash)
        hi = q.one()
    except sqlalchemy.orm.exc.NoResultFound, e:
        return []
    q = session.query(Fingerprint).filter(and_(Fingerprint.id.in_(hi.fingerprint_ids), Fingerprint.method == method, Fingerprint.n == n, Fingerprint.k == k, Fingerprint.atom_type == atom_type))
    fingerprints = q.all()
    return fingerprints
       
def _populate_db(method, n, k, atom_type):
    '''
    Populate the database with Fingerprints and HashIndexs for each document in the corpus.
    Use the fingerprint method, n, and k given.
    '''
    session = Session()

    suspect_file_listing = file(ExtrinsicUtility.TRAINING_SUSPECT_LOC)
    all_suspect_files = [f.strip() for f in suspect_file_listing.readlines()]
    suspect_file_listing.close()
    
    source_file_listing = file(ExtrinsicUtility.TRAINING_SRC_LOC)
    all_source_files = [f.strip() for f in source_file_listing.readlines()]
    source_file_listing.close()

    counter = 0
    for filename in all_suspect_files:
        print filename, method, n, k
        get_fingerprints(filename, ExtrinsicUtility.CORPUS_SUSPECT_LOC, method, n, k, atom_type, False, session)
        counter += 1
        if counter%1 == 0:
            print "Progress on suspects (corpus=" + str(ExtrinsicUtility.TRAINING_SUSPECT_LOC) + ": ", counter/float(len(all_suspect_files)), '(' + str(counter) + '/' + str(len(all_suspect_files)) + ')'
            
    counter = 0
    for filename in all_source_files:
        print filename, method, n, k
        get_fingerprints(filename, ExtrinsicUtility.CORPUS_SRC_LOC, method, n, k, atom_type, True, session)
        counter += 1
        if counter%1 == 0:
            print "Progress on sources (corpus=" + str(ExtrinsicUtility.TRAINING_SRC_LOC) + ": ", counter/float(len(all_source_files)), '(' + str(counter) + '/' + str(len(all_source_files)) + ')'
                            
    session.commit()
    session.close()
    
class Fingerprint(Base):
    
    __tablename__ = "st3_fingerprint"
    id = Column(Integer, Sequence("st3_fingerprint_id_seq"), primary_key=True, index=True)
    timestamp = Column(DateTime)
       
    doc_name = Column(String, index=True)
    _doc_path = Column(String)
    _doc_xml_path = Column(String)
    method = Column(String, index = True) # The fingerprinting method used to create this fingerprint
    n = Column(Integer, index=True)
    k = Column(Integer, index = True)
    atom_type = Column(String, index = True)
    atom_number = Column(Integer, index = True) # If this is the 1th paragraph in the document, then atom_number = 1
    hash_indexed = Column(Boolean) # If true, the hash_values of this fingerprint will be put into the hash index table.
    
    hash_values = Column(ARRAY(Integer))

    def __init__(self, doc_name, base_path, method, n, k, atom_type, atom_number, hash_indexed, atom_text = None):
    
        self.doc_name = doc_name
        self.method = method
        self.n = n
        self.k = k
        self.atom_type = atom_type
        self.atom_number = atom_number
        self.hash_indexed = hash_indexed
        self.timestamp = datetime.datetime.now()
        
        if method not in ["kth_in_sent", "winnow-k"]:
            self.k = 0
        
        # Construct _doc_path and _doc_xml_path
        if base_path in self.doc_name:
            self._doc_path = self.doc_name + ".txt"
            self._doc_xml_path = self.doc_name + ".xml"
        else:
            self._doc_path = base_path + self.doc_name + ".txt"
            self._doc_xml_path = base_path + self.doc_name + ".xml"
        
        # Extract the fingerprint from the text
        self.hash_values = self._extract_fingerprint(atom_text)
        
        # THE HASH INDEX WILL BE UPDATED AFTER CALLING update_hash_index
    
    def update_hash_index(self, session, commit=False):
        '''
        Update the hash index with the hash values from this object. This method should be
        called immediately after initializing this object!!!
        '''
        if self.hash_indexed == True:
            for hash in self.hash_values:
                try:
                    query = session.query(HashIndex).filter(HashIndex.hash_value == hash)
                    hi = query.one()
                    hi.fingerprint_ids = hi.fingerprint_ids[:] + [self.id]
                    session.add(hi) #Is this line needed!?
                except sqlalchemy.orm.exc.NoResultFound, e:
                    hi = HashIndex(hash, self.id)
                    session.add(hi)
                if commit:
                    session.commit()
    
    def _extract_fingerprint(self, atom_text = None):
        '''
        Fingerprint the text associated with this object. This method should only be called
        during the initialization of this object.
        '''
        
        # Get the text
        f = open(self._doc_path, 'r')
        text = f.read()
        f.close()
        
        # Get the atom
        if atom_text != None:
            atom = atom_text
        else:
            if self.atom_type == "full":
                atom = text
            elif self.atom_type == "paragraph":
                paragraph_spans = tokenize(text, self.atom_type)
                atom_start, atom_end = paragraph_spans[self.atom_number]
                atom = text[atom_start:atom_end]
            else:
                raise ValueError("Invalid atom_type! Only 'full' and 'paragraph' are allowed.")
        
        # fingerprint the atom
        extractor = fingerprint_extraction.FingerprintExtractor()
        hash_values = extractor.get_fingerprint(atom, self.n, self.method, self.k)
        
        return hash_values

class HashIndex(Base):
    __tablename__ = "st3_hash_index"
    hash_value = Column(Integer, index=True, primary_key = True)
    fingerprint_ids = Column(ARRAY(Integer))
    
    def __init__(self, hash_value, fingerprint_id):
        '''
        Instantiate a new object/row in the database with the given hash_value and the
        given fingerprint_id as the first item in the index for this hash.
        '''
        self.hash_value = hash_value
        self.fingerprint_ids = [fingerprint_id]

def _test():
    '''
    Retrieve fingerprints from two documents, twice. Then do a hash index look up.
    '''
    session = Session()

    # Fingerprints
    base_path = "/copyCats/itty-bitty-corpus/source"
    for j in range(2): # Do it twice, once where the objects are created, and a second where they are retrieved.
        for path in ["/source-born","/source-feynman"]:
            f = open(base_path+path+".txt", "r")
            text = f.read()
            f.close()
            for i in range(len(tokenize(text, "paragraph"))):
                fp = get_fingerprint(path, base_path, "kth_in_sent", 5, 5, "paragraph", i, True, session)
                print fp.hash_values
        session.commit()
    
    # HashIndex
    print get_fingerprints_by_hash(fp.hash_values[0], "kth_in_sent", 5, 5, "paragraph", session)

def _drop_tables():
    Base.metadata.drop_all(engine)
def _create_tables():
    Base.metadata.create_all(engine)
def main():
    _populate_db("kth_in_sent", 5, 5, "paragraph")

url = "postgresql://%s:%s@%s" % (username, password, dbname)
engine = sqlalchemy.create_engine(url)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

if __name__ == "__main__":
    #_drop_tables()
    #_create_tables()
    main()
    #_test()