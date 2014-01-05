import datetime

import fingerprint_extraction
from tokenization import *
from dbconstants import username, password, dbname

import sqlalchemy
from sqlalchemy import Table, Column, Sequence, Integer, String, Text, Float, DateTime, ForeignKey, and_
from sqlalchemy.orm import relationship, backref, sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm.collections import attribute_mapped_collection
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.associationproxy import association_proxy

import pickle # used for storing multidimensional python lists as fingerprints in the db

Base = declarative_base()

def _query_fingerprint(docs, method, n, k, atom_type, session, base_path):
	'''
	queries the database to see if the fingerprint in question exists in the database.
	If it does, it will be returned, otherwise it is created and added to the database.
	'''
	if method != "kth_in_sent":
		k = 0
	try:
		fp = session.query(FingerPrint).filter(and_(FingerPrint.document_name == docs, FingerPrint.atom_type == atom_type, FingerPrint.method == method, FingerPrint.n == n, FingerPrint.k == k)).one()
	except sqlalchemy.orm.exc.NoResultFound, e:
		fp = FingerPrint(docs, method, n, k, atom_type, base_path)
		session.add(fp)
		session.commit()
	return fp	

class FingerPrint(Base):
	'''
	this is the object that we store in the database.
	'''
	
	__tablename__ = "fingerprints"
	
	id = Column(Integer, Sequence("fingerprint_id_seq"), primary_key=True)
	document_name = Column(String)
	_doc_path = Column(String)
	_doc_xml_path = Column(String)
	method = Column(String)
	n = Column(Integer)
	k = Column(Integer)
	atom_type = Column(String)
	fingerprint = Column(Text)
	timestamp = Column(DateTime)
	version_number = Column(Integer)
	
	def __init__(self, doc, select_method, n, k, atom_type, base_path):
		'''
		initializes FingerPrint
		'''
		self.document_name = doc
		self._doc_path = base_path + self.document_name + ".txt"
		self._doc_xml_path = base_path + self.document_name + ".xml"
		self.method = select_method
		self.n = n
		self.k = k
		self.atom_type = atom_type
		self.timestamp = datetime.datetime.now()
		self.version_numer = 2
		

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
			if self.method == "full":
				fingerprint = fe._get_full_fingerprint(text, self.n)
			elif self.method == "kth_in_sent":
				fingerprint = fe._get_kth_in_sent_fingerprint(text, self.n, self.k)
			elif self.method == "anchor":
				fingerprint = fe._get_anchor_fingerprints(text, self.n)
			
			self.fingerprint = pickle.dumps(fingerprint)
			session.commit()
			return pickle.loads(str(self.fingerprint))
		else:
			return pickle.loads(str(self.fingerprint))
		
		
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
			for span in paragraph_spans:
				paragraph = text[span[0]:span[1]]
				if self.method == "full":
					fingerprint = fe._get_full_fingerprint(paragraph, self.n)
				elif self.method == "kth_in_sent":
					fingerprint = fe._get_kth_in_sent_fingerprint(paragraph, self.n, self.k)
				elif self.method == "anchor":
					fingerprint = fe._get_anchor_fingerprints(paragraph, self.n)
				paragraph_fingerprints.append(fingerprint)
			self.fingerprint = pickle.dumps(paragraph_fingerprints)
			session.commit()
			return pickle.loads(str(self.fingerprint))
		else:
			return pickle.loads(str(self.fingerprint))
				
class _ParagraphIndex(Base):
	__tablename__ = "paragraph_indices"
	fingerprint_id = Column(Integer, ForeignKey('fingerprints.id'), primary_key=True)
	paragraph_fingerprints_id = Column(Integer, ForeignKey('paragraph_fingerprints.id'), primary_key=True)
	special_key = Column(Integer)
	fingerprint = relationship(FingerPrint, backref=backref(
			"paragraph_fingerprints",
			collection_class=attribute_mapped_collection("special_key"),
			cascade="all, delete-orphan"
			)
		)
	kw = relationship("_ParagraphFingerprint")
	pf = association_proxy('kw', 'pf')

class _ParagraphFingerprint(Base):
	__tablename__ = "paragraph_fingerprints"
	id = Column(Integer, primary_key=True)
	pf = Column(ARRAY(Integer))
	def __init__(self, pf):
		self.pf=pf


def testRun():
	'''
	this is a testRun
	'''
	session = Session()
	documents = ["/part4/suspicious-document06242", "/part5/suspicious-document08911", "/part3/suspicious-document04127", "/part6/suspicious-document11686"]
	for docs in documents:
		fp = _query_fingerprint(docs, "full", 3, 5, "paragraph", session, '/copyCats/pan-plagiarism-corpus-2009/external-detection-corpus/suspicious-documents')
		print fp.get_fingerprints(session)[0:4]
	session.close()

def populate_database():
	'''
	Opens a session and then populates the database using filename, method, n, k.
	'''
	session = Session()

	test_file_listing = file('extrinsic_corpus_partition/extrinsic_training_suspect_files.txt')
	all_test_files = [f.strip() for f in test_file_listing.readlines()]
	test_file_listing.close()
	
	source_file_listing = file('extrinsic_corpus_partition/extrinsic_training_source_files.txt')
	all_source_files = [f.strip() for f in source_file_listing.readlines()]
	source_file_listing.close()
	
	counter = 0
	
	for filename in all_test_files:
		for atom_type in ["full", "paragraph"]:
			for method in ["full"]: # add other fingerprint methods
				for n in range(3,6):
					for k in [5,8]:
						# print "Calculating fingerprint for ", filename, " with atom_type=", atom_type, "using ", method , "and ", n, "-gram"
						fp = _query_fingerprint(filename, method, n, k, atom_type, session, '/copyCats/pan-plagiarism-corpus-2009/external-detection-corpus/suspicious-documents')
		counter += 1
		if counter%10 == 0:
			print str(counter) + '/' + str(len(all_test_files))
			print "Progress: ", counter/float(len(all_test_files))
	
	for filename in all_source_files:
		for atom_type in ["full", "paragraph"]:
			for method in ["full"]: # add other fingerprint methods
				for n in range(3,6):
					for k in [5,8]:
						# print "Calculating fingerprint for ", filename, " with atom_type=", atom_type, "using ", method , "and ", n, "-gram"
						fp = _query_fingerprint(filename, method, n, k, atom_type, session, '/copyCats/pan-plagiarism-corpus-2009/external-detection-corpus/source-documents')
		counter += 1
		if counter%10 == 0:
			print str(counter) + '/' + str(len(all_source_files))
			print "Progress: ", counter/float(len(all_source_files))
	

	session.close()

url = "postgresql://%s:%s@%s" % (username, password, dbname)
engine = sqlalchemy.create_engine(url)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

if __name__ == "__main__":
	#testRun()
	#unitTest()
	populate_database()