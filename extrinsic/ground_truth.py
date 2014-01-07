# ground_truth.py

import pickle
from .. import tokenization
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
	try:
		gt = session.query(GroundTruth).filter(and_(GroundTruth.document_name == doc_name, GroundTruth.atom_type == atom_type)).one()
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
			spans = tokenization.tokenize(doc_text.read(), self.atom_type)
			doc_text.close()
			
			plag_spans = self._get_plagiarized_spans()
			_ground_truth = []
			for s in spans:
				truth = 0
				for ps in plag_spans:
					if s[0] >= ps[0] and s[0] < ps[1]:
						truth = 1
						break
				_ground_truth.append(truth)
			
			self.ground_truth = pickle.dumps(_ground_truth)
			session.commit()
			return _ground_truth
		else:
			return pickle.loads(str(self.ground_truth))

	def _get_plagiarized_spans(self):
		'''
		Using the ground truth, return a list of spans representing the passages of the
		text that are plagiarized. Note, this method was plagiarized from Noah's intrinsic
		testing code.
		'''
		spans = []
		tree = xml.etree.ElementTree.parse(self._doc_xml_path)
		for feature in tree.iter("feature"):
			if feature.get("name") == "artificial-plagiarism":
				start = int(feature.get("this_offset"))
				end = start + int(feature.get("this_length"))
				spans.append((start, end))
		return spans

def populate_database():
	session = Session()

	test_file_listing = file('extrinsic_corpus_partition/extrinsic_training_suspect_files.txt')
	all_suspect_files = [f.strip() for f in test_file_listing.readlines()]
	test_file_listing.close()
	
	source_file_listing = file('extrinsic_corpus_partition/extrinsic_training_source_files.txt')
	all_source_files = [f.strip() for f in source_file_listing.readlines()]
	source_file_listing.close()
	
	counter = 0
	for filename in all_suspect_files:
		for atom_type in ["full", "paragraph"]:
			fp = _query_ground_truth(filename, atom_type, session, '/copyCats/pan-plagiarism-corpus-2009/external-detection-corpus/suspicious-documents')
			fp.get_ground_truth(session)
		counter += 1
		if counter%10 == 0:
			print str(counter) + '/' + str(len(all_suspect_files))
			print "Progress on suspect files: ", counter/float(len(all_suspect_files))
	
	counter = 0
	for filename in all_source_files:
		for atom_type in ["full", "paragraph"]:
			fp = _query_ground_truth(filename, atom_type, session, '/copyCats/pan-plagiarism-corpus-2009/external-detection-corpus/source-documents')
			fp.get_ground_truth(session)
		counter += 1
		if counter%10 == 0:
			print str(counter) + '/' + str(len(all_source_files))
			print "Progress on suspect files: ", counter/float(len(all_source_files))

url = "postgresql://%s:%s@%s" % (username, password, dbname)
engine = sqlalchemy.create_engine(url)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)	
	
if __name__ == "__main__":
	populate_database()

