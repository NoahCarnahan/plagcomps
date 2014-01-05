# extrinsic_testing.py

import xml
import random
import scipy
import sklearn
import sklearn.metrics
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as pyplot
import os
import time
import nltk
import fingerprint_extraction
import extrinsic_processing

import sqlalchemy
from sqlalchemy import Table, Column, Sequence, Integer, String, Float, DateTime, ForeignKey, and_
from sqlalchemy.orm import relationship, backref, sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm.collections import attribute_mapped_collection
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.associationproxy import association_proxy

class ExtrinsicTester:

	def __init__(self, atom_type, fingerprint_method, suspect_file_list, source_file_list):
		self.suspicious_path_start = "/copyCats/pan-plagiarism-corpus-2009/external-detection-corpus/suspicious-documents"
		self.corpus_path_start = "/copyCats/pan-plagiarism-corpus-2009/external-detection-corpus/source-documents"
		source_dirs = os.listdir(self.corpus_path_start)
		self.source_file_names = source_file_list # [self.corpus_path_start + f for f in source_file_list]
		self.suspect_file_names = suspect_file_list # [self.suspicious_path_start + f for f in suspect_file_list]
	
		# uncomment these two lines to test on reasonable sized corpus
		# self.source_file_names = ["sample_corpus/source1", "sample_corpus/source2", "sample_corpus/source3"]
		# self.suspect_file_names = ["sample_corpus/test1", "sample_corpus/test2", "sample_corpus/test3", "sample_corpus/test4"]

		self.atom_type = atom_type
		self.fingerprint_method = fingerprint_method
		self.suspect_file_list = suspect_file_list
		self.evaluator = fingerprint_extraction.FingerprintEvaluator(source_file_list, fingerprint_method, 3)

	def _get_trials(self):
		'''
		For each testing document, split the document into atoms and classify each atom
		as plagiarized or not-plagiarized. Build a list of classifications and a list
		of the ground truths for each atom of each document.
		'''
		classifications = []
		actuals = []
		for f in self.suspect_file_names:
			suspect_file_path = self.suspicious_path_start + f
			print f
			suspicious_document = open(suspect_file_path + '.txt')
			doc = suspicious_document.read()
			# atom_spans = feature_extractor.get_spans(doc, self.atom_type)
			# atoms = [doc[a[0]:a[1]] for a in atom_spans]
			# TODO: make this a for loop over atoms once paragraph fingerprints are supported in the database
			suspicious_document.close()
			atom_classifications = self.evaluator.classify_document(f, self.atom_type, 0, session)
			# just take the most similar source document's similarity as the confidence of plagiarism for now.
			similarity = atom_classifications[0][1]

			acts = self._get_actuals(doc, suspect_file_path + '.xml')
			actuals += acts
			classifications += [similarity for i in xrange(len(acts))] # just classify each paragraph in a document as the same similarity
		return classifications, actuals

	def _get_actuals(self, document_text, xml_filepath):
		'''
		Returns a list of the true plagiarism status for each atom of
		the given document.
		'''
		if self.atom_type == "sentence":
			tokenizer = nltk.PunktSentenceTokenizer()
			spans = tokenizer.span_tokenize(document_text)
		elif self.atom_type == "paragraph":
			paragraph_texts = document_text.splitlines()
			s = []
			start_index = 0
			for paragraph in paragraph_texts:
				start = document_text.find(paragraph, start_index)
				s.append((start, start + len(paragraph)))
				start_index = start + len(paragraph)
			spans = s

		plag_spans = self._get_plagiarized_spans(xml_filepath)
		actuals = []
		for s in spans:
			act = 0
			for ps in plag_spans:
				if s[0] >= ps[0] and s[0] < ps[1]:
					act = 1
					break
			actuals.append(act)
		return actuals

	def _get_plagiarized_spans(self, xml_file_path):
		'''
		Using the ground truth, return a list of spans representing the passages of the
		text that are plagiarized. Note, this method was plagiarized from Noah's intrinsic
		testing code.
		'''
		spans = []
		tree = xml.etree.ElementTree.parse(xml_file_path)
		for feature in tree.iter("feature"):
			if feature.get("name") == "artificial-plagiarism":
				start = int(feature.get("this_offset"))
				end = start + int(feature.get("this_length"))
				spans.append((start, end))
		return spans

	def plot_ROC_curve(self):
		'''
		Outputs an ROC figure based on our plagiarism classifications and the 
		ground truth of each atom.
		'''
		trials, actuals = self._get_trials()

		# actuals is a list of ground truth classifications for passages
		# trials is a list consisting of 0s and 1s. 1 means we think the atom is plagiarized
		fpr, tpr, thresholds = sklearn.metrics.roc_curve(actuals, trials, pos_label=1)
		roc_auc = sklearn.metrics.auc(fpr, tpr)

		# The following code is from http://scikit-learn.org/stable/auto_examples/plot_roc.html
		pyplot.clf()
		pyplot.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
		pyplot.plot([0, 1], [0, 1], 'k--')
		pyplot.xlim([0.0, 1.0])
		pyplot.ylim([0.0, 1.0])
		pyplot.xlabel('False Positive Rate')
		pyplot.ylabel('True Positive Rate')
		pyplot.title('Receiver Operating Characteristic -- Extrinsic w/ '+self.fingerprint_method+' fingerprinting')
		pyplot.legend(loc="lower right")
		
		path = "figures/roc_extrinsic_"+str(time.time())+"_"+self.fingerprint_method+".pdf"
		pyplot.savefig(path)


if __name__ == "__main__":
	session = extrinsic_processing.Session()
	# fp = extrinsic_processing._query_fingerprint('/part7/suspicious-document12675', "full", 3, 5, "paragraph", session, '/copyCats/pan-plagiarism-corpus-2009/external-detection-corpus/suspicious-documents')
	# print fp.get_fingerprints(session)
	
	num_files = 5

	suspect_file_listing = open('extrinsic_corpus_partition/extrinsic_training_suspect_files.txt', 'r')
	suspect_file_list = []
	i = 0
	for line in suspect_file_listing:
		suspect_file_list.append(line.strip())
		i += 1
		if i >= num_files:
			break
	suspect_file_listing.close()
	print suspect_file_list

	source_file_listing = open('extrinsic_corpus_partition/extrinsic_training_source_files.txt', 'r')
	source_file_list = []
	for line in source_file_listing:
		source_file_list.append(line.strip())
	source_file_listing.close()
	source_file_list = source_file_list
	
	print 'Testing first', num_files, ' suspect files using a corpus of', len(source_file_list), 'source documents:'
	print 'Suspect filenames:', suspect_file_list

	tester = ExtrinsicTester("paragraph", "full", suspect_file_list, source_file_list)
	tester.plot_ROC_curve()

	# tester = ExtrinsicTester("paragraph", "kth_in_sent", suspect_file_list, source_file_list)
	# tester.plot_ROC_curve()

	# tester = ExtrinsicTester("paragraph", "anchor", suspect_file_list, source_file_list)
	# tester.plot_ROC_curve()
