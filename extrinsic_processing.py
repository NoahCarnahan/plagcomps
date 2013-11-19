import datetime

import fingerprint_extraction
import feature_extractor
import dbconstants

import sqlalchemy
from sqlalchemy import Table, Column, Sequence, Integer, String, Float, DateTime, ForeignKey, and_
from sqlalchemy.orm import relationship, backref, sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm.collections import attribute_mapped_collection
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.associationproxy import association_proxy

#what does this do?
Base = declarative_base()

def _query_fingerprint(docs, method, n, k, atom_type, session):
	'''
	queries the database to see if the fingerprint in question exists in the database.
	If it does, it will be returned, otherwise it is created and added to the database.
	'''
	if method != "kth_in_sent":
		k = 0
	try:
		fp = session.query(FingerPrint).filter(and_(FingerPrint.document_name == docs, FingerPrint.atom_type == atom_type), FingerPrint.method == method, FingerPrint.n == n, FingerPrint.k == k).one()
	except sqlalchemy.orm.exc.NoResultFound, e:
		fp = FingerPrint(docs, method, n, k, atom_type)
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
	fingerprint = association_proxy(
					"paragraph_fingerprints",
					"pf",
					creator=lambda k, v:
						_ParagraphIndex(special_key=k, pf=v)
				)

	timestamp = Column(DateTime)
	version_number = Column(Integer)
	
	def __init__(self, doc, select_method, n, k, atom_type):
		'''
		initializes FingerPrint
		'''
		self.document_name = doc
		base_path = "/copyCats/pan-plagiarism-corpus-2009/external-detection-corpus/suspicious-documents"
		self._doc_path = base_path + self.document_name + ".txt"
		self._doc_xml_path = base_path + self.document_name + ".xml"
		self.method = select_method
		self.n = n
		self.k = k
		self.atom_type = atom_type
		self.timestamp = datetime.datetime.now()
		self.version_numer = 1
		
		
		
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
			return self._get_paragraph_fingerprint(session)
		else:
			return self._get_fingerprint(session)
		
	def _get_fingerprint(self, session):
		'''
		pseudo-private function to return fingerprint, returns previously calculated fingerprint if it exists.
		Otherwise, it calculates the fingerprint and returns.  Uses the full text for fingerprint calculations.
		'''
		if self.fingerprint == None or len(self.fingerprint) == 0:

			f = open(self._doc_path, 'r')
			text = f.read()
			f.close()


			fe = fingerprint_extraction.FingerprintExtractor()
			if self.method == "full":
				self.fingerprint[0] = fe._get_full_fingerprint(text, self.n)
			elif self.method == "kth_in_sent":
				self.fingerprint[0] = fe._get_kth_in_sent_fingerprint(text, self.n, self.k)
			elif self.method == "anchor":
				self.fingerprint[0] = fe._get_anchor_fingerprint(text, self.n)

			session.commit()
			return self.fingerprint
		else:
			return self.fingerprint
		
		
	def _get_paragraph_fingerprint(self, session):
		'''
		pseudo-private function to return fingerprint, returns previously calculated fingerprint if it exists.
		Otherwise, it calculates the fingerprint and returns.  Uses paragraphs for fingerprint calculations.
		'''

		if self.fingerprint == None or len(self.fingerprint) == 0:


			f = open(self._doc_path, 'r')
			text = f.read()
			f.close()

			paragraph_spans = feature_extractor.get_spans(text, self.atom_type)

			index = 0

			fe = fingerprint_extraction.FingerprintExtractor()
			for span in paragraph_spans:
				paragraph = text[span[0]:span[1]]
				if self.method == "full":
					fingerprint = fe._get_full_fingerprint(paragraph, self.n)
				elif self.method == "kth_in_sent":
					fingerprint = fe._get_kth_in_sent_fingerprint(paragraph, self.n, self.k)
				elif self.method == "anchor":
					fingerprint = fe._get_anchor_fingerprint(paragraph, self.n)
				if len(fingerprint) == 0:
					continue
				else:
					self.fingerprint[index] = fingerprint
				index += 1

			session.commit()
			return self.fingerprint

		else:

			return self.fingerprint
				
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
		fp = _query_fingerprint(docs, "full", 3, 5, "full", session)
		print fp.get_fingerprints(session)[0][0:4]
	session.close()

def populate_database():
	'''
	Opens a session and then populates the database using filename, method, n, k.
	'''
	session = Session()

	test_file_listing = file('extrinsic_corpus_partition/extrinsic_training_suspect_files.txt')
	all_test_files = [f.strip() for f in test_file_listing.readlines()]
	test_file_listing.close()
	
	counter = 0

	for filename in all_test_files:
		for method in ["full", "anchor"]:
			for n in range(3,6):
				for k in [0]:
					for atom_type in ["full", "paragraph"]:
						print counter," : Calculating fingerprint for ", filename, " using ", method , "and ", n, "-gram and granularity ", atom_type, "."
						fp = _query_fingerprint(filename, method, n, k, atom_type, session)
						print fp.get_fingerprints(session)[0][0:4]
						if counter%1000 == 0:
							print counter
							print "Progress: ", counter/float(len(all_test_files)*2*3*2)
		counter += 1

	session.close()

url = "postgresql://%s:%s@%s" % (dbconstants.username, dbconstants.password, dbconstants.dbname)
engine = sqlalchemy.create_engine(url)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

if __name__ == "__main__":
	#testRun()
	#unitTest()
	populate_database()