import pprint

from feature_extractor import *
from cluster import *

class Controller:

	def __init__(self, input_file):
		self.feature_evaluator = StylometricFeatureEvaluator(input_file)
		self.cluster_util = StylometricCluster()
		# Make printing legible
		self.printer = pprint.PrettyPrinter(indent = 2)

	def extractFeatures(self, atom_type):
		'''
		Uses <self.feature_evaluator> to extract <atom_type> features
		'''
		features = []
		for i in xrange(len(self.feature_evaluator.getAllByAtom(atom_type))):
			# For the <i_th> atom, extract the features
			features.append(self.feature_evaluator.getFeatures(i, i + 1, atom_type))

		return features

	def clusterFeatures(self, features, method, k):
		'''
		Clusters <features> using <method> from <self.cluster_util>
		'''
		if method == 'kmeans':
			return self.cluster_util.kmeans(features, k)
		elif method == 'agglom':
			return self.cluster_util.agglom(features, k)

	def printClusterAssignments(self, assignments, atom_type):
		'''
		Prints the sentences in each cluster, as assigned by <assignments>
		'''
		all_clusters = set(assignments)
		cluster_to_atoms = {}
		all_atoms = self.feature_evaluator.getAllByAtom(atom_type)

		for cluster in all_clusters:
			atom_indices = [i for i in range(len(assignments)) if assignments[i] == cluster]
			cluster_to_atoms[cluster] = [all_atoms[i] for i in atom_indices]
		self.printer.pprint(cluster_to_atoms)


	def test(self, atom_type, method, k):
		'''
		Clusters the sentence features and prints the resulting clusters
		'''
		print 'Using %s and %s with %i clusters\n\n' % (atom_type, method, k)
		features = self.extractFeatures(atom_type)
		cluster_assignments = self.clusterFeatures(features, method, k)
		self.printClusterAssignments(cluster_assignments, atom_type)

if __name__ == '__main__':
	Controller('foo.txt').test('sentence', 'agglom', 3)


