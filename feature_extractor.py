# feature_extractor.py
# An early-stage prototype for extracting stylometric features used for intrinsic plagiarism detection
# plagcomps rules
# by Marcus, Noah, and Cole

import nltk, re, math

'''
Class that extends the nltk PunktWordTokenizer. Unfortunately, PunktWordTokenizer doesn't 
implement the span_tokenize() method, so we implemented it here.
'''
class CopyCatPunktWordTokenizer(nltk.tokenize.punkt.PunktBaseClass,nltk.tokenize.punkt.TokenizerI):
    def __init__(self, train_text=None, verbose=False, lang_vars=nltk.tokenize.punkt.PunktLanguageVars(), token_cls=nltk.tokenize.punkt.PunktToken):
        nltk.tokenize.punkt.PunktBaseClass.__init__(self, lang_vars=lang_vars, token_cls=token_cls)

    '''Returns a list of strings that are the individual words of the given text.'''
    def tokenize(self, text):
        return self._lang_vars.word_tokenize(text)

    '''Returns a list of tuples, each containing the start and end index for the respective
       words returned by tokenize().'''
    def span_tokenize(self, text):
        return [(sl[0], sl[1]) for sl in self._slices_from_text(text)]

    def _slices_from_text(self, text):
        last_break = 0
        indices = []
        for match in self._lang_vars._word_tokenizer_re().finditer(text):
            context = match.group()
            indices.append((match.start(), match.end()))
        return indices


class StylometricFeatureEvaluator:

    def __init__(self, filepath):
        self.setDocument(filepath)
        self.punctuation_re = re.compile(r'[\W\d]+', re.UNICODE) # slightly modified from nltk source
    
    
    def setDocument(self, filepath):
        ''' Reads in the file and constructs a list of words, sentences, and paragraphs from it. '''
        f = open(filepath, 'r')
        self.input_file = f.read()
        f.close()
        
        self.word_spans = self.initWordList(self.input_file)
        self.sentence_spans = self.initSentenceList(self.input_file)
        self.paragraph_spans = self.initParagrpahList(self.input_file)
    
        #self.word_length_sum_table = self.initWorldLengthSumTable()
        #self.sentence_length_sum_table = self.initSentenceLengthSumTable()
    
    def initWordList(self, text):
        '''
        Returns a list of tuples representing the locations of words in the document.
        Each tuple contains the start charcter index and end character index of the word.
        For example initWordList("Hi there") = [(0,1),(3,7)]
        '''
        tokenizer = CopyCatPunktWordTokenizer()
        return tokenizer.span_tokenize(text)
    
    def initSentenceList(self, text):
        '''
        Returns a list of tuples representing the locations of sentences in the document.
        Each tuple contains the start character index and end character index of the sentence.
        For example initSentenceList("Hi there. Whats up!") = [(0,8),(10,18)]
        '''
        tokenizer = nltk.PunktSentenceTokenizer()
        return tokenizer.span_tokenize(text)
    
    def initParagrpahList(self, text):
        '''
        Returns a list of tuples representing the locations of paragraphs in the document.
        Each tuple contains the start character index and end character index of the paragraph.
        For example initParagrpahList("Hi there. Whats up!") = [(0,19)]
        '''
        # It's unclear how a paragraph is defined. For now, just treat newlines as paragraph separators
        paragraph_texts = text.splitlines()
        spans = []
        start_index = 0
        for paragraph in paragraph_texts:
            start = text.find(paragraph, start_index)
            spans.append((start, start+len(paragraph)))
            start_index = start + len(paragraph)
        return spans

    def _binarySearchForSpanIndex(self, spans, index, first):
        ''' Perform a binary search across the list of spans to find the index in the spans that
            corresponds to the given character index from the source document. The parameter <first>
            indicates whether or not we're searching for the first or second element of the spans. '''
        # clamps to the first or last character index
        if index >= spans[-1][1]:
            index = spans[- 1][1]
        elif index < 0:
            index = 0
        element = 0 if first else 1
        lower = 0
        upper = len(spans)-1
        prev_span_index = (upper+lower)/2
        cur_span_index = prev_span_index
        cur_span = spans[cur_span_index]
        while True:
            if cur_span[element] == index: # found the exact index
                return cur_span_index
            elif cur_span[element] > index: # we need to look to the left
                if lower >= upper:
                    if element == 0:
                        return cur_span_index - 1
                    else:
                        if index >= cur_span[0]:
                            return cur_span_index
                        elif index <= spans[cur_span_index-1][1]:
                            return cur_span_index - 1
                        else:
                            return cur_span_index
                prev_span_index = cur_span_index
                upper = cur_span_index - 1
                cur_span_index = (upper+lower)/2
                cur_span = spans[cur_span_index]
            elif cur_span[element] < index: # we need to look to the right
                if lower >= upper:
                    if element == 0:
                        if index <= cur_span[1]:
                            return cur_span_index
                        else:
                            return cur_span_index + 1
                    else:
                        return cur_span_index + 1
                prev_span_index = cur_span_index
                lower = cur_span_index + 1
                cur_span_index = (upper+lower)/2
                cur_span = spans[cur_span_index]

    def getWordSpans(self, start_index, end_index):
        '''
        Returns a list of word spans from the self.word_spans list corresponding to the
        words between the given character indicies.

        Example:
        self.word_spans = [(0, 1), (3, 8), (10, 14), (16, 18)]
        getWordsIndices(4, 13) = [(3,8), (10,14)]
        getWordsIndices(9, 15) = [(10, 14)]
        getWordsIndices(15, 16) = exception! --> Should it be an exception? It's not right now.
        '''
        first_index = self._binarySearchForSpanIndex(self.word_spans, start_index, True)
        second_index = self._binarySearchForSpanIndex(self.word_spans, end_index, False)
        return [span for span in self.word_spans[first_index : second_index+1]]
    
    def getSentenceSpans(self, start_index, end_index):
        '''
        Returns a list of sentence spans from the self.sentence_spans list corresponding
        to the words between the given character indicies.
        
        Example:
        self.sentence_spans = [(0,8),(10,19)]
        getSentenceIndices(1, 15) = [(0,8),(10,19)]
        '''
        pass
    
    def getParagraphSpans(self, start_index, end_index):
        '''
        Returns a list of paragraphs spans from the self.paragraph_spans list
        corresponding to the words between the given character indicies.
        
        Example:
        self.paragraph_spans = [(0, 18)]
        getParagraphIndices(1, 15) = [(0, 18)]
        '''
        pass
    
    def initWorldLengthSumTable(self):
        '''
        Initializes the word_length_sum_table. word_length_sum_table[i] is the sum of the lengths
        of words from 0 to i.
        
        TODO: Check if words are punctuation?
        '''
        sum_table = [0] # This value allows the for loop to be cleaner. Notice that I remove it later.
        
        for start, end in self.word_spans:
            sum_table.append(len(self.input_file[start : end]) + sum_table[-1])
        sum_table.pop(0)
        
        return sum_table
    
    def initSentenceLengthSumTable(self):
        '''
        Initializes the sentence_length_sum_table. sentence_length_sum_table[i] is the sum of the number
        of words in sentences 0 to i.
        
        TODO: Check if words are punctuation?
        '''
        #TODO: make this method use initWordList
        sum_table = [0]
        for start, end in self.sentence_spans:
            sum = 0
            start_index_into_word_list, end_index_into_word_list = getWordIndices(start, end)
            for index_into_word_list in range(start_index_into_word_list, end_index_into_word_list):
                start_index_into_characters, end_index_into_characters = self.word_spans[index_into_word_list]
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
            word_chunk = self.word_spans[start_index : end_index]
            sentence_chunk = self.parseSentences(" ".join(word_chunk))
            # This method of building sentences might not be ideal. "$3.88" will be parsed
            # into ["$","3",".","88"] then joined back together as "$ 3 . 88". Unsure of
            # what other situations this could be problematic for...
        elif atom_type == 'sentence':
            sentence_chunk = self.sentence_spans[start_index : end_index]
            word_chunk = self.parseWords(" ".join(sentence_chunk))
        elif atom_type == 'paragraph':
            paragraph_chunk = self.paragraph_spans[start_index : end_index]
            word_chunk = self.parseWords(" ".join(paragraph_chunk))
            sentence_chunk = self.parseSentences(" ".join(paragraph_chunk))
        else:
            raise ValueError("atom_type string must be 'char', 'word', 'sentence' or 'paragraph', not '" + str(atom_type) + "'.")

        avg_word_length = self.averageWordLength(word_chunk)
        avg_sentence_length = self.averageSentenceLength(sentence_chunk)
        
        return [avg_word_length, avg_sentence_length]
    

    def averageWordLength(self, word_list_index_start, word_list_index_end):
        '''
        Returns the average word length of words between the given indicies into self.word_spans.
        
        TODO: Words that are just punctuation?
        '''
        total_word_length = self.word_length_sum_table[word_list_index_end] - word_length_sum_table[word_list_index_start]
        num_of_words = (word_list_index_end + 1) - word_list_index_start
        return float(total_word_length)/max(num_of_words, 1) # if there are no legitimate words, just set denominator to 1 to avoid division by 0
    
    def averageSentenceLength(self, sentence_list_index_start, sentence_list_index_end):
        '''
        Returns the average words-per-sentence for the sentences betwen the given indicies into self.sentence_spans.
        
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
        print 'words: ', self.word_spans
        print 'sentences: ', self.sentence_spans
        print 'paragraphs: ', self.paragraph_spans
        print
        print 'word spans: ', self.getWordSpans(0, 17)
        print
        print 'Extracted Stylometric Feature Vector: <avg_word_length, avg_words_in_sentence>'      
        print self.getFeatures(0, len(self.input_file), "char")
        print self.getFeatures(0, 2, "sentence")
        print
        print "Average word frequency class of 'The small cat jumped'"
        print self.averageWordFrequencyClass(["The", "small", "cat", "jumped"])
        

if __name__ == "__main__":
    StylometricFeatureEvaluator("foo.txt").test()
