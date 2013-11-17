import pprint

from feature_extractor import *
from cluster import *

class Controller:

	def __init__(self, input_file):
		self.feature_evaluator = StylometricFeatureEvaluator(input_file)
		self.cluster_util = StylometricCluster()
		# Make printing legible
		self.printer = pprint.PrettyPrinter(indent = 2)

	def get_passages(self, atom_type, feature_list, cluster_method, k):
		'''
		Extracts the features in <feature_list>, splitting by <atom_type>.
		Clusters using <cluster_method>
		'''
		all_passages = []
		for i in xrange(len(self.feature_evaluator.getAllByAtom(atom_type))):
			passage = self.feature_evaluator.get_specific_features(feature_list, i, i + 1, atom_type)
			# Avoid edge cases where features aren't parseable (i.e. a 'passage' of one word)
			if passage != None:
				all_passages.append(passage)

		# Assigns cluster numbers to each passage object stored in <all_passages>
		self.cluster_features(all_passages, cluster_method, k)

		return all_passages


	def cluster_features(self, passages, method, k):
		'''
		Assigns clusters to the given passage objects after clustering with method and k.
		Returns None.
		'''
		# Extract numeric features
		# NOTE (nj) we may want to change this at some point in order to 
		# give more weight to certain features
		stylo_features = [p.features.values() for p in passages]

		if method == 'kmeans':
			assignments = self.cluster_util.kmeans(stylo_features, k)
		elif method == 'agglom':
			assignments = self.cluster_util.agglom(stylo_features, k)
		elif method == 'hmm':
			assignments = self.cluster_util.hmm(stylo_features, k)

		print 'assignments:', assignments
		for i in range(len(assignments)):
			passages[i].assign_cluster(assignments[i])

	def print_cluster_assignments(self, passages):
		'''
		Prints the sentences in each cluster, as assigned by <assignments>
		'''
		text_split_by_atom = self.feature_evaluator.getAllByAtom(passages[0].atom_type)
		all_text = self.feature_evaluator.input_file
		
		all_clusters = set([p.cluster_num for p in passages])
		cluster_to_atoms = dict([(c, []) for c in all_clusters])

		for p in passages:
			cluster_to_atoms[p.cluster_num].append(p.text)
	
		self.printer.pprint(cluster_to_atoms)


	def test(self, atom_type, feature_list, cluster_method, k):
		'''
		Clusters the sentence features and prints the resulting clusters
		'''
		print 'Using %s and %s with %i clusters\n\n' % (atom_type, cluster_method, k)
		passages = self.get_passages(atom_type, feature_list, cluster_method, k)
		for p in passages:
			print p
		self.print_cluster_assignments(passages)
		

if __name__ == '__main__':
	c = Controller('/copyCats/pan-plagiarism-corpus-2009/intrinsic-detection-corpus/suspicious-documents/part1/suspicious-document00969.txt')
	features = [
		'averageWordLength',
		'averageSentenceLength',
		'get_avg_word_frequency_class',
		'get_punctuation_percentage',
		'get_stopword_percentage'
	]
	c.test('sentence', features, 'kmeans', 2)

