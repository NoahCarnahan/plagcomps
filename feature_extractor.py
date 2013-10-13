# feature_extractor.py
# An early-stage prototype for extracting stylometric features used for intrinsic plagiarism detection
# plagcomps rules
# by Marcus, Noah, and Cole

import nltk, re, math

class StylometricFeatureEvaluator:

    def __init__(self, filepath):
        self.setDocument(filepath)
        self.punctuation_re = re.compile(r'[\W\d]+', re.UNICODE) # slightly modified from nltk source
    
    
    def setDocument(self, filepath):
        ''' Reads in the file and constructs a list of words, sentences, and paragraphs from it. '''
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

    def getAllByAtom(self, atom_type):
        '''
        Returns document as parsed by <atom_type>
        '''
        if atom_type == 'word':
            return self.words
        elif atom_type == 'sentence':
            return self.sentences
        elif atom_type == 'paragraph':
            return self.paragraphs
        elif atom_type == 'char':
            print 'Will fix this later'
        else:
            raise ValueError("atom_type string must be 'char', 'word', 'sentence' or 'paragraph', not '" + str(atom_type) + "'.")
    
    def getFeatures(self, start_index, end_index, atom_type):
        ''' Returns a list of extracted stylometric features from the specified chunk of the document.
            The start and end indices use the same logic as indexing a string or list in Python.
            The atom_type parameter specifies what your start_index and end_index parameters refer to.
            For example, if you wanted to query the stylometric features for sentences 0 through 4, you 
            would call this function as getFeatures(0, 4, "sentence"). '''
            
        if atom_type == 'char':
            input_file_chunk = self.input_file[start_index : end_index]
            word_chunk = self.parseWords(input_file_chunk)
            sentence_chunk = self.parseSentences(input_file_chunk)
        elif atom_type == 'word':
            word_chunk = self.words[start_index : end_index]
            sentence_chunk = self.parseSentences(" ".join(word_chunk))
            # This method of building sentences might not be ideal. "$3.88" will be parsed
            # into ["$","3",".","88"] then joined back together as "$ 3 . 88". Unsure of
            # what other situations this could be problematic for...
        elif atom_type == 'sentence':
            sentence_chunk = self.sentences[start_index : end_index]
            word_chunk = self.parseWords(" ".join(sentence_chunk))
        elif atom_type == 'paragraph':
            paragraph_chunk = self.paragraphs[start_index : end_index]
            word_chunk = self.parseWords(" ".join(paragraph_chunk))
            sentence_chunk = self.parseSentences(" ".join(paragraph_chunk))
        else:
            raise ValueError("atom_type string must be 'char', 'word', 'sentence' or 'paragraph', not '" + str(atom_type) + "'.")

        avg_word_length = self.averageWordLength(word_chunk)
        avg_sentence_length = self.averageSentenceLength(sentence_chunk)
        
        return [avg_word_length, avg_sentence_length]
    

    def averageWordLength(self, words):
        ''' Returns the average word length for the given list of words. 
            Doesn't count puncuation as words. '''
        total = 0
        num_words = 0
        for word in words:
            # if the word is just punctuation, don't count it as a word
            if not (self._is_punctuation(word)):
                total += len(word)
                num_words += 1
        return float(total) / max(num_words, 1) # if there are no legitimate words, just set denominator to 1 to avoid division by 0
        
    
    def averageSentenceLength(self, sentences):
        ''' Returns the average words-per-sentence for the given list of sentences. '''
        total = 0
        num_sentences = 0
        for sentence in sentences:
            num_words = 0
            for word in self.parseWords(sentence):
                if not (self._is_punctuation(word)):
                    num_words += 1
            total += num_words
        return  float(total) / max(len(sentences), 1) # avoid division by 0
    
    
    def averageWordFrequencyClass(self, words):
        # This feature is defined here:
        # http://www.uni-weimar.de/medien/webis/publications/papers/stein_2006d.pdf
        #
        # What should we do if the frequency of a given word is 0? (causes div by 0 problem)
        # Unfortunately the reference in the Meyer Zu Eissen and Stein article is to a
        # German website and seems unrelated: http://wortschatz.uni-leipzig.de/
        #
        # One option is to use some sort of smoothing. Plus-one would be the simpleset, but
        # we might want to do a bit of thinking about how much effect this will have on word
        # class
        
        # Additionally, perhaps we should create a corpus class that supports functions such
        # as corpus.getFrequency(word) and a corpus could be passed into averageWordFrequency
        # as an argument. Lets do this.
    
    
        # This dictionary could perhaps be replaced by a nltk.probability.FreqDist
        # http://nltk.org/api/nltk.html#nltk.probability.FreqDist
        word_freq_dict = {}
        for word in nltk.corpus.brown.words():
            word = word.lower() # Do we care about case?
            word_freq_dict[word] = word_freq_dict.get(word, 0) + 1
        
        corpus_word_freq_by_rank = sorted(word_freq_dict.items(), key=lambda x: x[1], reverse=True)
        occurences_of_most_freq_word = corpus_word_freq_by_rank[0][1]
    
        total = 0
        word_total = 0
        for word in words:            
            freq_class = math.floor(math.log((float(occurences_of_most_freq_word)/word_freq_dict.get(word.lower(), 0)),2))
            total += freq_class
            word_total += 1
        return total/word_total
        
    
    
    def _is_punctuation(self, word):
        ''' Returns true if the given word is just punctuation. '''
        match_obj = re.match(self.punctuation_re, word)
        return match_obj and len(match_obj.group()) == len(word)
    
    def test(self):
        print 'words: ', self.words
        print 'sentences: ', self.sentences
        print 'paragraphs: ', self.paragraphs
        print
        print 'Extracted Stylometric Feature Vector: <avg_word_length, avg_words_in_sentence>'      
        print self.getFeatures(0, len(self.input_file), "char")
        print self.getFeatures(0, 2, "sentence")
        print
        print "Average word frequency class of 'The small cat jumped'"
        print self.averageWordFrequencyClass(["The", "small", "cat", "jumped"])
        

if __name__ == "__main__":
    StylometricFeatureEvaluator("foo.txt").test()
