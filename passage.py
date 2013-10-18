class Passage:
	'''
	A class to store/query information about a passage of text, including
	where it came from (doc_name), the type of atom used (char, word, sentence, 
	or paragraph, what chunk it refers to (text from start -> end), and its 
	stylometric features (a dictionary mapping feature => value)

	Once clustering is performed, sets the passage's cluster number
	as well
	'''
	
	def __init__(self, doc_name, atom_type, start, end, features):
		self.doc_name = doc_name
		self.atom_type = atom_type
		self.start = start
		self.end = end
		self.features = {}
		self.cluster_num = None


