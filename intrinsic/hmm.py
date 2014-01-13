import math
import random
import scipy.stats

class _State :
	def __init__(self, num, ft_list, trans_probs, means_list, variances):
		self.feature_list = ft_list
		self.num = num
		# TODO: make sure the randomly-initialized means/variances are relatively close to actual values being produced
		# TODO: maybe pick specifc points from the feature vectors?
		self.ft_list_means = means_list
		self.ft_list_variances = variances
		self.trans_probs = trans_probs 
	
	def get_emission_probability(self, feature_vector):
		''' Returns the probability of this state outputting the given feature_vector. 
		    This is done by quantizing the continous values described by the gaussian
		    distribution for this state. '''
						
		probability = 1.0
		if feature_vector == self.ft_list_means:
			return probability
		for i in xrange(len(feature_vector)):
			z_score = (feature_vector[i] - self.ft_list_means[i]) / self.ft_list_variances[i]
			z_score *= 10
			upper = math.floor(z_score+1)/10 * self.ft_list_variances[i] + self.ft_list_means[i]
			lower = math.floor(z_score)/10 * self.ft_list_variances[i] + self.ft_list_means[i]
			prob = scipy.stats.norm(self.ft_list_means[i], self.ft_list_variances[i]).cdf(upper) - scipy.stats.norm(self.ft_list_means[i], self.ft_list_variances[i]).cdf(lower)
			'''#print statements 
			print 'z_score is : ', z_score
			print upper
			print lower
			print 'prob is : ', prob
			'''
			probability *= prob
		#print probability + 0.00001
		return math.log(probability + 0.00001)
		
	def get_confidences(self, stylo_vectors, centroids, cluster_assignments):
		'''
		Given the stylo vector list, centroid values and cluster assignments, returns a list of confidences
		where confidences[i] pertains to stylo_vectors[i] -- only works for 2 clusters'''
		for i in xrange(stylo_vectors):
			#confidence distance
			list_differences[i] = math.abs(stylo_vectors[i] - centroids[cluster_assignments[i]])
			#do some math
			#for each centroid, create a list of tuples where tuples are (list_differences[i],i)
			#sort lists by ascending first elements of tuples
			#normalize distance values so that the largest distance is 1
			#return a list called confidences where confidences[i] = (normlized_distance of stylo_vectors[i])
		
def hmm_cluster(stylo_vectors, k):
	'''
	Return a list of k centroids and a list of the assigned clusters.
	'''
	# initialize uniform transition probabilities with respect to k
	trans_probs = {}
	for i in xrange(k):
		trans_probs[i] = {}
		for j in xrange(k):
			trans_probs[i][j] = math.log(1.0/k)
	
	# calculate variances of observed outputs with each state
	means_sum = [0 for j in xrange(len(stylo_vectors[0]))]
	for vector in stylo_vectors:
		for i in xrange(len(vector)):
			means_sum[i] += vector[i]
	for i in xrange(len(means_sum)):
		means_sum[i] /= float(len(stylo_vectors))
	
	variances = [0 for j in xrange(len(stylo_vectors[0]))]
	for vector in stylo_vectors:
		for i in xrange(len(vector)):
			variances[i] += (vector[i] - means_sum[i])**2
	for i in xrange(len(variances)):
		variances[i] /= float(len(stylo_vectors))
	
	states = [_State(i, stylo_vectors[0], trans_probs[i].copy(), stylo_vectors[len(stylo_vectors)/k*i], variances[:]) for i in range(k)]
	# initial probabilitis are uniform
	initial_state_probs = [math.log(1.0/k)]*k
	
	trained_path, trained_path_viterbi_prob = train_parameters(stylo_vectors, states, initial_state_probs)
	clusters_indices = {i: [] for i in xrange(k)}
	for i in xrange(len(trained_path)):
		clusters_indices[trained_path[i]].append(i)

	# calculate centroids of each cluster
	centroids = []
	for i in xrange(k):
		centroid = [0.0 for x in xrange(len(stylo_vectors[0]))]
		for s in clusters_indices[i]:
			centroid = [sum(a) for a in zip(centroid, stylo_vectors[s])]
		if len(clusters_indices[i]) > 0:
			centroids.append([a/len(clusters_indices[i]) for a in centroid])
		else:
			centroids.append(centroid)

	return centroids, trained_path
	
def train_parameters(feature_vectors, states, initial_state_probs):
	prev_viterbi_max = 1.0
	viterbi_path, viterbi_max = viterbi(feature_vectors, states, initial_state_probs)
	
	percent_change = 100.0
	while percent_change > 0.01:
		num_states = [viterbi_path.count(i) for i in xrange(len(states))]

		# calculate means of observed outputs with each state
		means_sum = [[0 for j in xrange(len(feature_vectors[0]))] for i in xrange(len(states))]
		index = 0
		for state in viterbi_path:
			means_sum[state] = [x+y for x, y in zip(means_sum[state], feature_vectors[index])]
			index += 1
		means = [state.ft_list_means for state in states]
		for i in xrange(len(states)):
			num = num_states[i]
			if num > 0:
				means[i] = [mean/num for mean in means_sum[i]]
		
		# calculate variances of boserved outputs with each state
		variances_sum = [[0 for j in xrange(len(feature_vectors[0]))] for i in xrange(len(states))]
		index = 0
		for state in viterbi_path:
			variances_sum[state] = [x+(mean-y)**2 for x, y, mean in zip(variances_sum[state], feature_vectors[index], means[state])]
			index += 1
		variances = [state.ft_list_variances for state in states]
		for i in xrange(len(states)):
			num = num_states[i]
			if num > 0:
				variances[i] = [variance/num for variance in variances_sum[i]]
		
		# updates gaussian parameters
		for i in xrange(len(states)):
			states[i].ft_list_means = [(x+y)/2 for x, y in zip(means[i], states[i].ft_list_means)]
			# states[i].ft_list_variances = [(x+y)/2 for x, y in zip(variances[i], states[i].ft_list_variances)]

		#update transission probabilities
		trans_possibilities = {}
		trans_counts = {}
		for s in xrange(len(states)):
			trans_possibilities[s] = {}
			trans_counts[s] = {}
			for t in xrange(len(states)):
				trans_possibilities[s][t] = 0
				trans_counts[s][t] = 0

		for i in xrange(len(viterbi_path)-1):
			state = viterbi_path[i]
			next_state = viterbi_path[i+1]
			for s in xrange(len(states)):
				trans_possibilities[state][s] += 1
			trans_counts[state][next_state] += 1

		for s in xrange(len(states)):
			for t in xrange(len(states)):
				old_prob = states[s].trans_probs[t]
				if trans_possibilities[s][t] > 0:
					new_prob = trans_counts[s][t] / float(trans_possibilities[s][t])
					# states[s].trans_probs[t] = (old_prob + new_prob) / 2

		percent_change = abs(prev_viterbi_max - viterbi_max) / prev_viterbi_max
		prev_viterbi_max = viterbi_max
		viterbi_path, viterbi_max = viterbi(feature_vectors, states, initial_state_probs)

	return viterbi_path, viterbi_max


def viterbi(feature_vectors, states, initial_state_probs):
	''' Performs the viterbi algorithm where <feature_vectors> is a list of observed feature vectors.
	    <states> is a list of state objects, and <initial_state_probs> is a list that corresponds
	    to the probabilities of starting in each state (these entried should add to 1). 
	    Returns a list of the most probable sequence of states given the observed outputs. '''
	# table stores the probability lattice
	# table2 stores the back pointers for each cell in the lattice
	table = [[-float('inf') for j in xrange(len(states))] for i in xrange(len(feature_vectors))]
	table2 = [[-1 for j in xrange(len(states))] for i in xrange(len(feature_vectors))]
	for i in xrange(len(table[0])):
		table[0][i] = initial_state_probs[i] + states[i].get_emission_probability(feature_vectors[0])
	
	# use dynamic programming to fill out the probability and back-pointer matrices
	for i in xrange(1,len(table)):
		for j in xrange(len(states)):
			cur_max = -float('inf')
			prev_state = 0
			for x in xrange(len(states)):
				prob = table[i-1][x] + states[x].trans_probs[j] + states[j].get_emission_probability(feature_vectors[i])
				if prob > cur_max:
					cur_max = prob
					prev_state = x
			table[i][j] = cur_max
			table2[i][j] = prev_state
			
	# find the state in the final column with maximum probability
	cur_max = -float('inf')
	max_state = 0
	for i in xrange(len(states)):
		if table[-1][i] > cur_max:
			cur_max = table[-1][i]
			max_state = i
	
	# construct the viterbi path using the back-pointer matrix
	vpath = [max_state]
	cur_state = max_state
	for i in xrange(len(table)-1, 0, -1):
		prev_state = table2[i][cur_state]
		vpath.insert(0, prev_state)
		cur_state = table2[i-1][prev_state]
	return vpath, cur_max