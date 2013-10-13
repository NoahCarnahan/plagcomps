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
        
        self.words = self.getWordIndices(self.input_file)
        self.sentences = self.getSentenceIndices(self.input_file)
        self.paragraphs = self.getParagraphIndices(self.input_file)
    
        self.word_length_sum_table = self.initWorldLengthSumTable()
        self.sentence_length_sum_table = self.initSentenceLengthSumTable()
    
    def getWordIndices(self, text):
        '''
        Returns a list of tuples representing the locations of words in the document.
        Each tuple contains the start charcter index and end character index of the word.
        For example getWordIndices("Hi there") = [(0,1),(3,7)]
        '''
        pass
    
    def getSentenceIndices(self, text):
        '''
        Returns a list of tuples representing the locations of sentences in the document.
        Each tuple contains the start character index and end character index of the sentence.
        For example getWordIndices("Hi there. Whats up!") = [(0,8),(10,18)]
        '''
        pass
    
    def getParagraphIndices(self, text):
        '''
        Returns a list of tuples representing the locations of paragraphs in the document.
        Each tuple contains the start character index and end character index of the paragraph.
        For example getWordIndices("Hi there. Whats up!") = [(0,18)]
        '''
        pass
    
    def getWordIndices(self, start_index, end_index):
        '''
        Returns the start index and end index into the self.word list corresponding to the words between
        the given character indicies.
        Example:
        words = [(0, 1), (3, 8), (10, 14), (16, 18)]
        getWordsIndices(4, 13) = (1, 2)
        getWordsIndices(9, 15) = (2, 2)
        getWordsIndices(15, 15) = exception!
        '''
        pass
    
    def getSentenceIndices(self, start_index, end_index):
        '''
        Returns the start index and end index into the self.sentences list corresponding to the words
        between the given character indicies.
        Example:
        sentences = [(0,8),(10,18)]
        getSentenceIndices(1, 15) = (0, 1)
        '''
        pass
    
    def getParagraphIndices(self, start_index, end_index):
        '''
        Returns the start index and end index into the self.paragrpahs list corresponding to the words
        between the given character indicies.
        Example:
        paragraphs = [(0, 18)]
        getParagraphIndices(1, 15) = (1, 1)
        '''
        pass
    
    def initWorldLengthSumTable(self):
        '''
        Initializes the word_length_sum_table. word_length_sum_table[i] is the sum of the lengths
        of words from 0 to i.
        
        TODO: Check if words are punctuation?
        '''
        sum_table = [0] # This value allows the for loop to be cleaner. Notice that I remove it later.
        
        for start, end in self.words:
            sum_table.append(len(self.input_file[start, end]) + sum_table[-1])
        sum_table.pop(0)
        
        return sum_table
    
    def initSentenceLengthSumTable(self):
        '''
        Initializes the sentence_length_sum_table. sentence_length_sum_table[i] is the sum of the number
        of words in sentences 0 to i.
        
        TODO: Check if words are punctuation?
        '''
        sum_table = [0]
        for start, end in self.sentences:
            sum = 0
            start_index_into_word_list, end_index_into_word_list = getWordIndices(start, end)
            for index_into_word_list in range(start_index_into_word_list, end_index_into_word_list):
                start_index_into_characters, end_index_into_characters = self.words[index_into_word_list]
                word = self.input_file[start_index_into_characters, end_index_into_characters]
                sum += 1
            sum_table.append(sum + sum_table[-1])
        sum_table.pop(0)
        return sum_table
    
    # TODO: Refactor this method
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
    

    def averageWordLength(self, word_list_index_start, word_list_index_end):
        '''
        Returns the average word length of words between the given indicies into self.words.
        
        TODO: Words that are just punctuation?
        '''
        total_word_length = self.word_length_sum_table[word_list_index_end] - word_length_sum_table[word_list_index_start]
        num_of_words = (word_list_index_end + 1) - word_list_index_start
        return float(total_word_length)/max(num_of_words, 1) # if there are no legitimate words, just set denominator to 1 to avoid division by 0
    
    def averageSentenceLength(self, sentence_list_index_start, sentence_list_index_end):
        '''
        Returns the average words-per-sentence for the sentences betwen the given indicies into self.sentences.
        
        TODO: Words that are just punctuation?    
        '''
        total_words_per_sentences = self.sentence_length_sum_table[sentence_list_index_end] - sentence_length_sum_table[sentence_list_index_start]
        num_of_sentences = (sentence_list_index_start + 1) - sentence_list_index_end
        
        return float(total_word_length)/max(num_of_words, 1) # avoid division by 0
    
    #TODO: Refactor this method
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
