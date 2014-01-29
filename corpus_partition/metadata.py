
from scipy.stats import scoreatpercentile
from plagcomps.shared.util import IntrinsicUtility
from plagcomps.tokenization import tokenize

import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import os

CORPUS_DIR = '/copyCats/pan-plagiarism-corpus-2009/intrinsic-detection-corpus/suspicious-documents/'

class Document:

	def __init__(self, name, doc_length, plag_passage_lengths):
		self.name = name
		self.doc_length = doc_length
		self.plag_passage_lengths = plag_passage_lengths
		self.num_plag_passages = len(plag_passage_lengths)
		if len(plag_passage_lengths) > 0:
			self.prop_of_plag_text = float(sum(plag_passage_lengths)) / len(plag_passage_lengths)
		else:
			self.prop_of_plag_text = 0.0

	def __str__(self):
		return '%s\t\t %i\t\t %i\t\t %f' % (self.name, self.num_plag_passages, self.doc_length, self.prop_of_plag_text)


def summarize_data():
	for directory in os.listdir(CORPUS_DIR):
		all_docs = explore_dir(directory)
		summarize_docs(directory, all_docs)
		print '---\n'*4

def summarize_docs(dirname, docs):
	print 'In directory', dirname
	num_plag = sum([d.num_plag_passages > 0 for d in docs])
	print float(num_plag) / len(docs), 'of the documents had some plagiarism'

	print 'The document lengths had the following distribution'
	five_num_summary([d.doc_length for d in docs])

	all_plag_lengths = []
	for d in docs:
		all_plag_lengths.extend(d.plag_passage_lengths)

	print 'The lengths of the instances of plag. had the following distribution'
	five_num_summary(all_plag_lengths)


def explore_dir(dir_name):
	# Parse out base names of all files in <dir_name>
	full_dir_path = CORPUS_DIR + dir_name + '/'
	
	all_file_bases = get_base_file_names(full_dir_path)
	feature_types = set()
	all_docs = []

	for file_base in all_file_bases:
		xml_full_path = full_dir_path + file_base + '.xml'
		text_full_path = full_dir_path + file_base + '.txt'

		doc_length = len(file(text_full_path, 'r').read())

		tree = ET.parse(xml_full_path)

		plag_lengths = []

		for feature in tree.iter('feature'):
			feature_types.add(feature.get('name'))
			
			if feature.get('name') == 'artificial-plagiarism':
				length = int(feature.get('this_length'))
				plag_lengths.append(length)
				
		doc = Document(file_base, doc_length, plag_lengths)
		all_docs.append(doc)


	return all_docs

def get_base_file_names(full_dir_path):
	all_files = os.listdir(full_dir_path)
	all_file_bases = [f[:-4] for f in all_files if f[-4:] == '.xml']

	return all_file_bases

def five_num_summary(arr):
	'''
	Prints <min, 25th percentile, median, 75th percentile, max>
	of data stored in <arr>
	'''
	min_val = min(arr)
	first_quart = scoreatpercentile(arr, 25)
	median = scoreatpercentile(arr, 50)
	third_quart = scoreatpercentile(arr, 75)
	max_val = max(arr)

	print min_val, first_quart, median, third_quart, max_val
	print 'Mean', sum(arr) / float(len(arr)), '\n'

def doc_lengths(thresh=35000):
	'''
	Prints the pct. of documents which contain at least <thresh> characters
	'''
	util = IntrinsicUtility()
	training_docs = util.get_n_training_files()
	lengths = []
	long_enough = 0

	for fname in training_docs:
		f = file(fname, 'rb')
		text = f.read()
		f.close()

		lengths.append(len(text))
		if len(text) > thresh:
			long_enough += 1

	print float(long_enough) / len(training_docs), 'were long enough'


def explore_training_corpus(n=1000):
	'''
	'''
	util = IntrinsicUtility()
	training_texts = util.get_n_training_files(n)
	training_xmls = [s.replace('txt', 'xml') for s in training_texts]

	file_lengths = []
	pct_plags = []
	total_paragraphs = []

	for text_file, xml_file in zip(training_texts, training_xmls):
		with file(text_file) as f:
			text = f.read()

		paragraphs_spans = tokenize(text, 'paragraph')
		num_paragraphs = len(paragraphs_spans)

		text_len = len(text)
		plag_spans = util.get_plagiarized_spans(xml_file)
		plag_len = sum([end - start for start, end in plag_spans])
		plag_pct = float(plag_len) / text_len

		file_lengths.append(text_len)
		pct_plags.append(plag_pct)
		total_paragraphs.append(num_paragraphs)

	#outfile = os.path.join(os.path.dirname(__file__), 'training_lengths.csv')
	outfile = 'training_lengths.csv'

	f = file(outfile, 'wb')
	f.write('file_num, length, pct_plag, num_paragraphs\n')

	for i in xrange(len(file_lengths)):
		line = '%i, %i, %f, %i\n' % (i, file_lengths[i], pct_plags[i], total_paragraphs[i])
		f.write(line)
	f.close()

	return zip(file_lengths, pct_plags)



if __name__ == '__main__':
	summarize_data() 