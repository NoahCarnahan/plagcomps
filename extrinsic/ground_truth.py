# ground_truth.py

import cPickle
from .. import tokenization
from ..shared.util import ExtrinsicUtility, BaseUtility
from ..dbconstants import username, password, dbname

import xml

import sqlalchemy
from sqlalchemy import Table, Column, Sequence, Integer, String, Text, Float, DateTime, ForeignKey, and_
from sqlalchemy.orm import relationship, backref, sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm.collections import attribute_mapped_collection
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.associationproxy import association_proxy

Base = declarative_base()

def _query_ground_truth(doc_name, atom_type, session, base_path):
    '''
    queries the database to see if the ground_truth in question exists in the database.
    If it does, it will be returned, otherwise it is created and added to the database.
    '''
    if session == None:
        session = Session()
    try:
        q = session.query(GroundTruth).filter(and_(GroundTruth.document_name == doc_name, GroundTruth.atom_type == atom_type))
        gt = q.one()
    except sqlalchemy.orm.exc.NoResultFound, e:
        gt = GroundTruth(doc_name, atom_type, base_path)
        session.add(gt)
        session.commit()
    return gt

class GroundTruth(Base):
    '''
    Ground Truth object stored in the database. The ground truths are stored
    in a integer array (0 for not-plagiarized, 1 for plagiarized) corresponding
    to atom_type as a serialized string in the database.
    '''
    __tablename__ = "ground_truths"
    
    id = Column(Integer, Sequence("ground_truth_id_seq"), primary_key=True)
    document_name = Column(String)
    _doc_path = Column(String)
    _doc_xml_path = Column(String)
    atom_type = Column(String)
    ground_truth = Column(String)
    
    def __init__(self, doc, atom_type, base_path):
        '''
        initializes FingerPrint
        '''
        
        self.document_name = doc
        if base_path in self.document_name:
            self._doc_path = self.document_name + ".txt"
            self._doc_xml_path = self.document_name + ".xml"
        else:
            self._doc_path = base_path + self.document_name + ".txt"
            self._doc_xml_path = base_path + self.document_name + ".xml"

        self.atom_type = atom_type
        
    def get_ground_truth(self, session):
        # might want to split on atom types here?
        return self._get_ground_truth(session)
    
    def _get_ground_truth(self, session):
        ''' 
        Instantiates the ground truth values for this document's paragraphs
        if it isn't in the database, yet. Returns the ground truth list.
        '''
        if self.ground_truth == None:
            doc_text = open(self._doc_path, 'r')
            spans = tokenization.tokenize(doc_text.read(), self.atom_type, n=5000)
            doc_text.close()
            
            plag_spans, source_filepaths, source_plag_spans, obfuscations = ExtrinsicUtility().get_plagiarized_spans(self._doc_xml_path)
            
            _ground_truth = []
            for s in spans:
                truth = 0
                filepaths = []
                plagiarized_spans = []
                source_plagiarized_spans = []
                obfuscation_levels = []
                for ps, filepath, source_plag_span, obfuscation in zip(plag_spans, source_filepaths, source_plag_spans, obfuscations):
                    if BaseUtility().overlap(s, ps) > 0:
                        filepath = filepath.replace(".txt", '')
                        filepath = filepath.replace(ExtrinsicUtility.CORPUS_SRC_LOC, '')
                        filepaths.append(filepath)
                        plagiarized_spans.append(ps)
                        source_plagiarized_spans.append(source_plag_span)
                        obfuscation_levels.append(obfuscation)
                        truth = 1
                _ground_truth.append((truth, filepaths, plagiarized_spans, source_plagiarized_spans, obfuscation_levels))
            
            self.ground_truth = cPickle.dumps(_ground_truth)
            session.commit()
            return _ground_truth
        else:
            return cPickle.loads(str(self.ground_truth))

def populate_database():
    session = Session()

    _, suspect_filenames = ExtrinsicUtility().get_corpus_files()
    
    counter = 0
    for filename in suspect_filenames[:5]:
        filename = filename.rstrip(".txt")
        print 'filename:', filename
        for atom_type in ["full", "paragraph"]:
            fp = _query_ground_truth(filename, atom_type, session, ExtrinsicUtility.CORPUS_SUSPECT_LOC)
            fp.get_ground_truth(session)
            print fp.get_ground_truth(session)
        counter += 1
        if counter%10 == 0:
            print str(counter) + '/' + str(len(suspect_filenames))
            print "Progress on suspect files: ", counter/float(len(suspect_filenames))




url = "postgresql://%s:%s@%s" % (username, password, dbname)
engine = sqlalchemy.create_engine(url)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine) 
    
if __name__ == "__main__":
    populate_database()

