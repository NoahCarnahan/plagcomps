import xml.etree.ElementTree as ET
import os

corpus_dir = '/copyCats/pan-plagiarism-corpus-2009/intrinsic-detection-corpus/'

def explore_dir(dir_name = 'suspicious-documents/part1/'):
	full_dir_path = corpus_dir + dir_name
	all_files = os.listdir(full_dir_path)

	xml_files = [f for f in all_files if f[-4:] == '.xml']

	# [doc_name] is the number of plagarised passages in <doc_name>
	num_plag_passages = {}

	# Length of each plagiarism case
	all_plag_lengths = []
	feature_types = set()

	avg_len_plag_passages = {}

	for xml_file_name in xml_files:
		tree = ET.parse(full_dir_path + xml_file_name)
		total_plags = 0

		for feature in tree.iter('feature'):
			feature_types.add(feature.get('name'))
			
			if feature.get('name') == 'artificial-plagiarism':
				length = int(feature.get('this_length'))
				all_plag_lengths.append(length)
				total_plags += 1

		num_plag_passages[xml_file_name] = total_plags

	print num_plag_passages.values()
	print all_plag_lengths


	print feature_types

if __name__ == '__main__':
	explore_dir() 