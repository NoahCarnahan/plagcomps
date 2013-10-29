import cluster, feature_extractor
import math, random
import scipy.stats

class State :
	def __init__(self, ft_list, trans_probs):
		self.feature_list = ft_list
		self.ft_list_means = [random.random()*15.0 for i in xrange(len(self.feature_list))]
		self.ft_list_variances = [random.random()*5.0 for i in xrange(len(self.feature_list))]
		self.trans_probs = trans_probs 
	
	def emit(self):
			return [random.normalvariate(ft_list_means[i], ft_list_variances[i]) for i in xrange(len(self.feature_list))]
			
	def transition(self):
			return random.random() < self.trans_prob
	
	def get_emission_probability(self, feature_vector):
		''' Returns the probability of this state outputting the given feature_vector. 
		    This is done by quantizing the continous values described by the gaussian
		    distribution for this state. '''
		probability = 1.0
		for i in xrange(len(feature_vector)):
			z_score = (feature_vector[i] - self.ft_list_means[i]) / self.ft_list_variances[i]
			upper = math.floor(z_score+1) * self.ft_list_variances[i] + self.ft_list_means[i]
			lower = math.floor(z_score) * self.ft_list_variances[i] + self.ft_list_means[i]
			prob = scipy.stats.norm(self.ft_list_means[i], self.ft_list_variances[i]).cdf(upper) - scipy.stats.norm(self.ft_list_means[i], self.ft_list_variances[i]).cdf(lower)
			probability *= prob
		return probability

def baum_welch(filename):
	'''
	Performs the Baum-Welch algorithm to learn the unknown state transition and emission
	probabilities. The results of this algorithm will be used in turn by the viterbi
	algorithm.
	
	We may or may not implement this later. It's extremely tricky and confusing to update
	the gaussian parameters...
	'''
	pass
	
def train_parameters(feature_vectors, states, initial_state_probs):
	for a in xrange(1000):
		viterbi_path = viterbi(feature_vectors, states, initial_state_probs)
		print viterbi_path
		for state in states:
			print 'means:',state.ft_list_means
			print 'vars: ',state.ft_list_variances
		print
		means_sum = [[0 for j in xrange(len(feature_vectors[0]))] for i in xrange(len(states))]
		index = 0
		for state in viterbi_path:
			means_sum[state] = [x+y for x, y in zip(means_sum[state], feature_vectors[index])]
			index += 1
		means = [state.ft_list_means for state in states]
		for i in xrange(len(states)):
			num_states = viterbi_path.count(i)
			if num_states > 0:
				means[i] = [mean/num_states for mean in means_sum[i]]
		
		variances_sum = [[0 for j in xrange(len(feature_vectors[0]))] for i in xrange(len(states))]
		index = 0
		for state in viterbi_path:
			variances_sum[state] = [x+(mean-y)**2 for x, y, mean in zip(variances_sum[state], feature_vectors[index], means[state])]
			index += 1
		variances = [state.ft_list_variances for state in states]
		for i in xrange(len(states)):
			num_states = viterbi_path.count(i)
			if num_states > 0:
				variances[i] = [variance/num_states for variance in variances_sum[i]]
		
		for i in xrange(len(states)):
			states[i].ft_list_means = [(x+y)/2 for x, y in zip(means[i], states[i].ft_list_means)]
			states[i].ft_list_variances = [(x+y)/2 for x, y in zip(variances[i], states[i].ft_list_variances)]

	return viterbi(feature_vectors, states, initial_state_probs)


def viterbi(feature_vectors, states, initial_state_probs):
	''' Performs the viterbi algorithm where <feature_vectors> is a list of observed feature vectors.
	    <states> is a list of state objects, and <initial_state_probs> is a list that corresponds
	    to the probabilities of starting in each state (these entried should add to 1). 
	    Returns a list of the most probable sequence of states given the observed outputs. '''
	# table stores the probability lattice
	# table2 stores the back pointers for each cell in the lattice
	table = [[0 for j in xrange(len(states))] for i in xrange(len(feature_vectors))]
	table2 = [[-1 for j in xrange(len(states))] for i in xrange(len(feature_vectors))]
	for i in xrange(len(table[0])):
		table[0][i] = initial_state_probs[i]
	
	# use dynamic programming to fill out the probability and back-pointer matrices
	for i in xrange(1,len(table)):
		for j in xrange(len(states)):
			cur_max = 0
			prev_state = 0
			for x in xrange(len(states)):
				prob = table[i-1][x] * states[x].trans_probs[j]*states[j].get_emission_probability(feature_vectors[i])
				if prob > cur_max:
					cur_max = prob
					prev_state = x
			table[i][j] = cur_max
			table2[i][j] = prev_state
			
	# find the state in the final column with maximum probability
	cur_max = 0
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
	return vpath
	
def test():
	a = State(["averageWordLength", "averageSentenceLength"], (0.98,0.02))
	b = State(["averageWordLength", "averageSentenceLength"], (0.02,0.98))
	b.ft_list_means = [10.0, 14.0]
	a.ft_list_means = [5.0, 9.0]
	observed_features = [[5.0, 9.0], [5.0, 9.0], [10.0, 14.0], [5.0, 9.0], [10.0, 14.0]]
	states = [a, b]
	initial_probs = [0.98, 0.02]
	viterbi_path = train_parameters(observed_features, states, initial_probs)
	# this test should return [0, 0, 1, 0, 1]
	print viterbi_path

if __name__ == "__main__":
	test()