class Trial:

	def __init__(self, docname, features, num_correct, total):
		self.docname = docname
		self.features = features
		self.num_correct = num_correct
		self.total = total

	def format_csv(self, all_possible_features):
		'''
		Returns a csv representation of this trial. If <all_possible_features> has 
		<n> features, the first <n> fields are binary indicators (1 if the
		feature was used in this trial). The following 3 fields are the document's 
		name, the number of correctly classified passages, and the total number
		of passages
		'''
		# feature_present_vec[i] = 0 iff all_possible_features[i] 
		# was used in this trial; 1 otherwise
		feature_present_vec = ['1' if f in self.features else '0' for f in all_possible_features ]

		output = feature_present_vec + [self.docname, str(self.num_correct), str(self.total)]
		print 'output in format', output
		# Gives a csv representation of this trial
		return ', '.join(output) + '\n'


	def get_file_part_and_name(self, full_path):
		# TODO write this. Output would be nicer if it wasn't the full path
		# to the file used in this trial
		pass

	def get_pct_correct(self):
		return float(self.num_correct) / self.total	

