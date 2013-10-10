# feature_extractor.py
# An early-stage prototype for extracting stylometric features used for intrinsic plagiarism detection
# plagcomps rules
# by Marcus, Noah, and Cole

import nltk, re

class StylometricFeatureEvaluator:
	def __init__(self, filepath):
		self.setDocument(filepath)
		self.punctuation_re = re.compile(r'[\W\d]+', re.UNICODE) # slightly modified from nltk source
	
	''' Reads in the file and constructs a list of words, sentences, and paragraphs from it. '''
	def setDocument(self, filepath):
		f = open(filepath, 'r')
		self.input_file = f.read()
		f.close()
		self.words = self.parseWords(self.input_file)
		self.sentences = self.parseSentences(self.input_file)
		self.paragraphs = self.parseParagraphs(self.input_file)
	
	def parseWords(self, text):
		return nltk.wordpunct_tokenize(text)
		
	def parseParagraphs(self, text):
		return text.splitlines()
	
	def parseSentences(self, text):
		return nltk.sent_tokenize(text)
	
	''' Returns a list of extracted stylometric features from the specified chunk of the document.
	    The start and end indices use the same logic as indexing a string or list in Python.
	    The atom_type parameter specifies what your start_index and end_index parameters refer to.
	    For example, if you wanted to query the stylometric features for sentences 0 through 4, you 
	    would call this function as getFeatures(0, 4, "sentence"). '''
	def getFeatures(self, start_index, end_index, atom_type):
		if atom_type == 'char':
			input_file_chunk = self.input_file[start_index : end_index]
			avg_word_length = self.averageWordLength(self.parseWords(input_file_chunk))
			avg_sentence_length = self.averageSentenceLength(self.parseSentences(input_file_chunk))
		elif atom_type == 'word':
			word_chunk = self.words[start_index : end_index]
			avg_word_length = self.averageWordLength(word_chunk)
			avg_sentence_length = self.averageSentenceLength(self.parseSentences(" ".join(word_chunk)))
		elif atom_type == 'sentence':
			sentence_chunk = self.sentences[start_index : end_index]
			avg_word_length = self.averageWordLength(self.parseWords(" ".join(sentence_chunk)))
			avg_sentence_length = self.averageSentenceLength(self.parseSentences(sentence_chunk))
		elif atom_type == 'paragraph':
			paragraph_chunk = self.paragraphs[start_index : end_index]
			avg_word_length = self.averageWordLength(self.parseWords(" ".join(paragraph_chunk)))
			avg_sentence_length = self.averageSentenceLength(self.parseSentences(" ".join(paragraph_chunk)))
		else:
			avg_word_length = -1
			avg_sentence_length = -1
		
		return [avg_word_length, avg_sentence_length]
	
	''' Returns the average word length for the given list of words. 
	    Doesn't count puncuation as words. '''
	def averageWordLength(self, words):
		total = 0
		num_words = 0
		for word in words:
			# if the word is just punctuation, don't count it as a word
			if not (self._is_punctuation(word)):
				total += len(word)
				num_words += 1
		return float(total) / max(num_words, 1) # if there are no legitimate words, just set denominator to 1 to avoid division by 0
		
	''' Returns the average words-per-sentence for the given list of sentences. '''
	def averageSentenceLength(self, sentences):
		total = 0
		num_sentences = 0
		for sentence in sentences:
			num_words = 0
			for word in self.parseWords(sentence):
				if not (self._is_punctuation(word)):
					num_words += 1
			total += num_words
		return  float(total) / max(len(sentences), 1) # avoid division by 0
	
	''' Returns true if the given word is just punctuation. '''
	def _is_punctuation(self, word):
		match_obj = re.match(self.punctuation_re, word)
		return match_obj and len(match_obj.group()) == len(word)
	
	def test(self):
		print 'words: ', self.words
		print 'sentences: ', self.sentences
		print 'paragraphs: ', self.paragraphs
		print
		print 'Extracted Stylometric Feature Vector: <avg_word_length, avg_words_in_sentence>'		
		print self.getFeatures(0, len(self.input_file), "char")
		
StylometricFeatureEvaluator("foo.txt").test()
