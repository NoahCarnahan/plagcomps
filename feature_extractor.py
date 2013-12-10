 #feature_extractor.py
# An early-stage prototype for extracting stylometric features used for intrinsic plagiarism detection
# plagcomps rules
# by Marcus, Noah, and Cole

import nltk, re, math, os, inspect, random, scipy
from passage import *

# Define a paragraph to be bounded by multiple (2 or more) newline characters
PARAGRAPH_RE = r'\n{2,}'

class CopyCatPunktWordTokenizer(nltk.tokenize.punkt.PunktBaseClass,nltk.tokenize.punkt.TokenizerI):
    '''
    Class that extends the nltk PunktWordTokenizer. Unfortunately, PunktWordTokenizer doesn't 
    implement the span_tokenize() method, so we implemented it here.
    '''
    def __init__(self, train_text=None, verbose=False, lang_vars=nltk.tokenize.punkt.PunktLanguageVars(), token_cls=nltk.tokenize.punkt.PunktToken):
        nltk.tokenize.punkt.PunktBaseClass.__init__(self, lang_vars=lang_vars, token_cls=token_cls)

    
    def tokenize(self, text):
        '''Returns a list of strings that are the individual words of the given text.'''
        return self._lang_vars.word_tokenize(text)

    
    def span_tokenize(self, text):
        '''Returns a list of tuples, each containing the start and end index for the respective
            words returned by tokenize().'''
        return [(sl[0], sl[1]) for sl in self._slices_from_text(text)]

    def _slices_from_text(self, text):
        last_break = 0
        indices = []
        for match in self._lang_vars._word_tokenizer_re().finditer(text):
            context = match.group()
            indices.append((match.start(), match.end()))
        return indices

def get_spans(text, atom_type):
    '''
    Returns list of passage spans from the given text atomized by atom_type.
    '''
    if atom_type == "word":
        tokenizer = CopyCatPunktWordTokenizer()
        spans = tokenizer.span_tokenize(text)
    elif atom_type == "sentence":
        tokenizer = nltk.PunktSentenceTokenizer()
        spans = tokenizer.span_tokenize(text)
    elif atom_type == "paragraph":
        # Idea from this gem: http://stackoverflow.com/a/4664889/3083983
        # boundaries[i][0] == start of <ith> newline sequence (i.e. 2+ newlines)
        # boundaries[i][1] == end of <ith> newline sequence (i.e. 2+ newlines)
        boundaries = [(m.start(), m.end()) for m in re.finditer(PARAGRAPH_RE, text)]

        if len(boundaries) == 0:
            spans = [(0, len(text))]
        else:
            spans = [(0, boundaries[0][0])]
            for i in range(len(boundaries) - 1):
                cur_span = (boundaries[i][1], boundaries[i + 1][0])
                spans.append(cur_span)

            # NOTE could be an edge-case if there's no new-line at the end of the text
            if boundaries[-1][1] != len(text):
                spans.append((boundaries[-1][1], len(text)))
        
    else:
        raise ValueError("Unacceptable atom type.")
    
    return spans
   
class StylometricFeatureEvaluator:

    def __init__(self, filepath):
        if filepath != None:
            self.document_name = os.path.basename(filepath)
            self.setDocument(filepath)
        self.punctuation_re = re.compile(r'[\W\d]+', re.UNICODE) # slightly modified from nltk source

    def extract_corpus_paragraph_features(self, corpus_paras, feature_name):
        '''instantiates a StylometricFeatureEvaluator from a nltk corpus paragraph list
        example: StylometricFeatureEvaluator(nltk.corpus.gutenberg.paras('austen-jane')
        returns a list of values of the given feature for each paragraph'''

        self.document_name = "corpus_document"
        full_text = ""
    
        #print 'we have', len(corpus_paras), 'paras to combine'
        for paragraph in corpus_paras:
            paragraph_text = " ".join(paragraph[0])
            full_text += paragraph_text + "\n\n"
        #    print 'my full text is', full_text

        self.input_file = full_text

        self.word_spans = self.initWordList(self.input_file)
        self.sentence_spans = self.initSentenceList(self.input_file)
        self.paragraph_spans = self.initParagrpahList(self.input_file)
        self.posTags = self.initTagList(self.input_file)

        self.word_length_sum_table = self.initWordLengthSumTable()
        self.sentence_length_sum_table = self.initSentenceLengthSumTable()
        self.pos_frequency_count_table = self.initPosFrequencyTable()
        self.word_frequency_class_table = self.init_word_frequency_class_table(self.input_file)
        self.punctuation_table = self.init_punctuation_table()
        self.stopWords_table = self.init_stopWord_table(self.input_file)

        feature_vector = []
        for i in range(len(self.paragraph_spans) - 1):
            feature_vector.append(self.get_specific_features([feature_name], i, i+1, 'paragraph'))
        return feature_vector
    

    def setDocument(self, filepath):
        ''' Reads in the file and constructs a list of words, sentences, and paragraphs from it. '''
        f = open(filepath, 'r')
        self.input_file = f.read()
        f.close()
        
        self.word_spans = self.initWordList(self.input_file)
        self.sentence_spans = self.initSentenceList(self.input_file)
        self.paragraph_spans = self.initParagrpahList(self.input_file)
        self.posTags = self.initTagList(self.input_file)

        # <number of WORDS in doc> entries
        self.word_length_sum_table = self.initWordLengthSumTable()

        # <number of SENTENCES in doc> entries
        self.sentence_length_sum_table = self.initSentenceLengthSumTable()

        # <number of WORDS in doc> entries
        self.pos_frequency_count_table = self.initPosFrequencyTable()

        # <number of WORDS in doc> entries
        self.word_frequency_class_table = self.init_word_frequency_class_table(self.input_file)

        # <number of CHARACTERS in doc> entries
        self.punctuation_table = self.init_punctuation_table()

        # <number of WORDS in doc> entries
        self.stopWords_table = self.init_stopWord_table(self.input_file)
    
    def initWordList(self, text):
        '''
        Returns a list of tuples representing the locations of words in the document.
        Each tuple contains the start charcter index and end character index of the word.
        For example initWordList("Hi there") = [(0,2),(3,8)]
        '''
        return get_spans(text, "word")
    
    def initSentenceList(self, text):
        '''
        Returns a list of tuples representing the locations of sentences in the document.
        Each tuple contains the start character index and end character index of the sentence.
        For example initSentenceList("Hi there. Whats up!") = [(0,9),(10,19)]
        '''
        return get_spans(text, "sentence")
    
    def initParagrpahList(self, text):
        '''
        Returns a list of tuples representing the locations of paragraphs in the document.
        Each tuple contains the start character index and end character index of the paragraph.
        For example initParagrpahList("Hi there. Whats up!") = [(0,19)]
        '''
        # It's unclear how a paragraph is defined. For now, just treat newlines as paragraph separators
        return get_spans(text, "paragraph")

    def initTagList(self, text):
        '''
        Return a list of tuples of (word, part of speech) where the ith item in the list is the 
        (ith word, ith part of speech) from the text.
        '''

        taggedWordTuples = []
        sentenceSpans = self.getSentenceSpans(0, len(text))[0]
        for sentence in sentenceSpans:
            sentence = self.input_file[sentence[0]:sentence[1]]
            # run part-of-speech tagging on the sentence following the word_tokenize-ation of it
            taggedWordTuples += nltk.tag.pos_tag(nltk.word_tokenize(sentence))

        return taggedWordTuples

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

    def getAllByAtom(self, atom_type):
        ''' Returns document as parsed by <atom_type> '''
        if atom_type == 'word':
            return self.word_spans
        elif atom_type == 'sentence':
            return self.sentence_spans
        elif atom_type == 'paragraph':
            return self.paragraph_spans
        elif atom_type == 'char':
            print 'Will fix this later'
        else:
            raise ValueError("atom_type string must be 'char', 'word', 'sentence' or 'paragraph', not '" + str(atom_type) + "'.")

    def getWordSpans(self, start_index, end_index):
        '''
        Returns a list of word spans from the self.word_spans list corresponding to the
        words between the given character indicies. start_index and end_index are character
        indices from the original document. Also returns a tuple containing the actual
        indeces of the words in the word_spans list.

        Example:
        self.word_spans = [(0, 1), (3, 8), (10, 14), (16, 18)]
        getWordSpans(4, 13) = [(3,8), (10,14)]
        getWordSpans(9, 15) = [(10, 14)]
        getWordSpans(15, 16) = []
        '''
        first_index = self._binarySearchForSpanIndex(self.word_spans, start_index, True)
        second_index = self._binarySearchForSpanIndex(self.word_spans, end_index, False)
        return self.word_spans[first_index : second_index+1], (first_index, second_index)
    
    def getSentenceSpans(self, start_index, end_index):
        '''
        Returns a list of sentence spans from the self.sentence_spans list corresponding
        to the words between the given character indicies. start_index and end_index are character
        indices from the original document. Also returns a tuple containing the actual
        indeces of the sentences in the sentence_spans list.
        
        Example:
        self.sentence_spans = [(0,8),(10,19)]
        getSentenceSpans(1, 15) = [(0,8),(10,19)]
        '''
        first_index = self._binarySearchForSpanIndex(self.sentence_spans, start_index, True)
        second_index = self._binarySearchForSpanIndex(self.sentence_spans, end_index, False)
        return self.sentence_spans[first_index : second_index+1], (first_index, second_index)
    
    def getParagraphSpans(self, start_index, end_index):
        '''
        Returns a list of paragraphs spans from the self.paragraph_spans list
        corresponding to the words between the given character indicies. Also returns a tuple 
        containing the actual indeces of the paragraphs in the paragraph_spans list.
        
        Example:
        self.paragraph_spans = [(0, 18)]
        getParagraphIndices(1, 15) = [(0, 18)]
        '''
        first_index = self._binarySearchForSpanIndex(self.paragraph_spans, start_index, True)
        second_index = self._binarySearchForSpanIndex(self.paragraph_spans, end_index, False)
        return self.paragraph_spans[first_index : second_index+1], (first_index, second_index)

    def get_character_boundaries(self, atom_type, atom_start, atom_end):
        '''
        Returns the starting and ending indices, in all of <self.input_file>, of the 
        atom_type starting at <atom_start> and ending at <atom_end> - 1, i.e. in the 
        same manner as indexing a python list or string
        '''
        if atom_type == 'char':
            start_index = self.character_spans[atom_start][0]
            end_index = self.character_spans[atom_end - 1][1]
        elif atom_type == 'word':
            start_index = self.word_spans[atom_start][0]
            end_index = self.word_spans[atom_end - 1][1]
        elif atom_type == 'sentence':
            start_index = self.sentence_spans[atom_start][0]
            end_index = self.sentence_spans[atom_end - 1][1]
        elif atom_type == 'paragraph':
            start_index = self.paragraph_spans[atom_start][0]
            end_index = self.paragraph_spans[atom_end - 1][1]

        return start_index, end_index
    
    def initWordLengthSumTable(self):
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
        sum_table = [0]
        for start, end in self.sentence_spans:
            word_sum = 0
            word_spans = self.getWordSpans(start, end)[0]
            word_sum += len(word_spans)
            sum_table.append(word_sum + sum_table[-1])
        sum_table.pop(0)
        return sum_table

    def initPosFrequencyTable(self):
    	'''
    	instantiates a table of part-of-speech counts 
        currently tracks the following categories:
        0) conjunctions -- tags CC, IN (though we will try to ignore common prepositions that are not conjunctions)
        1) WH-pronouns -- tags WP, WP$
        2) Verbs -- tags VB, VBD, VBG, VBN, VBP, VBZ
        3) None of the above
    	'''
        sum_table = []
        total_count = [0,0,0,0]
        for posTuple in self.posTags:
            word = posTuple[0].lower()
            tag = posTuple[1]
            # get the current count
            current_count = total_count[:] 
            # we want conjunctions -- IN contains coordinating conjunctions and prepositions, 
            # so we will explicitly filter out some common prepositions
            if tag in ["CC", "IN"] and word not in ["to", "of", "in", "from", "on"]:
                current_count[0] += 1
            elif tag in ["WP", "WP$"]:
                current_count[1] += 1
            elif tag.startswith("VB"):
                current_count[2] += 1
            else:
                current_count[3] += 1
            # maintain the count outside this iteration
            total_count = current_count[:]
            sum_table.append(current_count)

        return sum_table

    def init_punctuation_table(self):
    	'''
    	instatiates the table for the punctuation counts which allows for constant-time
    	querying of punctuation percentages within a particular passage
    	'''
        sum_table = []
        myCount = 0
        for char in self.input_file:
            if char in ",./<>?;':[]{}\|!@#$%^&*()`~-_\"":
                myCount += 1
            sum_table.append(myCount)
        return sum_table

    def init_stopWord_table(self, text):
    	'''
    	instatiates the table for stopword counts which allows for constant-time
    	querying of stopword percentages within a particular passage
    	'''

        sum_table = []
        myCount = 0
        wordSpans = self.getWordSpans(0, len(text))
        for span in wordSpans[0]:
            word = self.input_file[span[0]:span[1]]
            if word in nltk.corpus.stopwords.words('english'):
                myCount += 1
            sum_table.append(myCount)
        return sum_table

    #TODO: Refactor this method
    def init_word_frequency_class_table(self, text):
    	'''
            This feature is defined here:
            http://www.uni-weimar.de/medien/webis/publications/papers/stein_2006d.pdf
            
            What should we do if the frequency of a given word is 0? (causes div by 0 problem)
            Unfortunately the reference in the Meyer Zu Eissen and Stein article is to a
            German website and seems unrelated: http://wortschatz.uni-leipzig.de/
            
            One option is to use some sort of smoothing. Plus-one would be the simpleset, but
            we might want to do a bit of thinking about how much effect this will have on word
            class
            
            Additionally, perhaps we should create a corpus class that supports functions such
            as corpus.getFrequency(word) and a corpus could be passed into averageWordFrequency
            as an argument. Lets do this.
    	'''
        
    
        # This dictionary could perhaps be replaced by a nltk.probability.FreqDist
        # http://nltk.org/api/nltk.html#nltk.probability.FreqDist
        word_freq_dict = {}
        wordSpans = self.getWordSpans(0, len(text))
        for span in wordSpans[0]:
            word = self.input_file[span[0]:span[1]]
            word = word.lower() # Do we care about case?
            word_freq_dict[word] = word_freq_dict.get(word, 0) + 1
        
        corpus_word_freq_by_rank = sorted(word_freq_dict.items(), key=lambda x: x[1], reverse=True)
        occurences_of_most_freq_word = corpus_word_freq_by_rank[0][1]
    

        # zach/tony:
        # this creates a table where each entry holds the summation of the WFC of all previous words
        # division (for averaging) will be done in the querying step
        word_frequency_class_table = []
        total = 0
        for i in range(len(wordSpans[0])):
            span = wordSpans[0][i]
            word = self.input_file[span[0]:span[1]]         
            freq_class = math.floor(math.log((float(occurences_of_most_freq_word)/word_freq_dict.get(word.lower(), 0)),2))
            total += freq_class
            word_frequency_class_table.append(total)

        return word_frequency_class_table
    
    def get_specific_features(self, feature_list, start_index, end_index, atom_type):
        ''' 
        Extracts the features passed in <feature_list> of <atom_type> between
        <start_index> and <end_index>, using "snap-out" indexing (see _fetch_boundary_indices)

        Every feature in <feature_list> should be a string corresponding to a method name
        in this class, and should accept as arguments one of:
        (1)'first_sentence_index' and 'last_sentence_index'
        (2) 'first_word_index' and 'last_word_index'
        (3) 'first_char_index' and 'last_char_index'

        Returns a Passage object, which stores information about the document, location of excerpt,
        and more. See the <Package> class for more documentation 
        '''
        boundaries = self._fetch_boundary_indices(atom_type, start_index, end_index)
        
        # If the passage is one word long, we hit issues in trying to parse grammatical features
        if boundaries['first_word_index'] == boundaries['last_word_index']:
            return None

        # For each feature in <feature_list>, store the result of calling <func_name> i.e.
        # func_name => func_name()
        passage_features = {}
        for func_name in feature_list:
            # Parse the function from <self> and pass it only the parameters it expects
            # to receive
            func = getattr(self, func_name)
            accepted_params = inspect.getargspec(func).args
            params_to_pass = {p : boundaries[p] for p in boundaries if p in accepted_params}
            passage_features[func_name] = func(**params_to_pass)

        
        # Store the character start/end of the given passage in Passage objects
        
        global_start, global_end = boundaries['first_char_index'], boundaries['last_char_index']
        text = self.input_file[global_start : global_end]
        
        passage = Passage(self.document_name, atom_type, global_start, global_end, text, passage_features)
        return passage

    def _fetch_boundary_indices(self, atom_type, start_index, end_index):
        '''
        Returns a dictionary with keys 
        'first_char_index', 'last_char_index',
        'first_word_index', 'last_word_index',
        'first_sentence_index', 'last_sentence_index'
        of <atom_type> starting at <start_index> and expanding to <end_index>
        Uses "snap-out" indexing, i.e. includes entire words/sentences even if the indices
        don't include an entire word/sentence. 
        Example:

        "The cow jumped over the moon. It was awesome. "
        If we queried start_index of 3 and end_index of 7, the <first_sentence_index> returned
        would be 0 and the <last_sentence_index> would be the index of the "."
        '''
        start_index = max(start_index, 0)
        first_char_index, last_char_index = self.get_character_boundaries(atom_type, start_index, end_index)

        if atom_type == 'char':
            end_index = min(end_index, len(self.input_file)-1) # clamp to end

            # fetch word spans specified by character indices
            word_spans, word_spans_indices = self.getWordSpans(start_index, end_index)
            first_word_index = word_spans_indices[0]
            last_word_index = word_spans_indices[1]

            # fetch sentence spans specified by character indices
            sentence_spans, sentence_spans_indices = self.getSentenceSpans(start_index, end_index)
            first_sentence_index = sentence_spans_indices[0]
            last_sentence_index = sentence_spans_indices[1]
        elif atom_type == 'word':
            end_index = min(end_index, len(self.word_spans)-1)

            first_word_index = start_index
            last_word_index = end_index

            # fetch sentence spans specified by word indices
            sentence_spans, sentence_spans_indices = self.getSentenceSpans(self.word_spans[start_index][0], self.word_spans[end_index][1])
            first_sentence_index = sentence_spans_indices[0]
            last_sentence_index = sentence_spans_indices[1]
        elif atom_type == 'sentence':
            end_index = min(end_index, len(self.sentence_spans)-1)

            # fetch word spans specified by sentence indices
            word_spans, word_spans_indices = self.getWordSpans(self.sentence_spans[start_index][0], self.sentence_spans[end_index][1])
            first_word_index = word_spans_indices[0]
            last_word_index = word_spans_indices[1]

            first_sentence_index = start_index
            last_sentence_index = end_index
        elif atom_type == 'paragraph':
            end_index = min(end_index, len(self.paragraph_spans)-1)

            # fetch word spans specified by paragraph indices
            word_spans, word_spans_indices = self.getWordSpans(self.paragraph_spans[start_index][0], self.paragraph_spans[end_index][1])
            first_word_index = word_spans_indices[0]
            last_word_index = word_spans_indices[1]

            # fetch sentence spans specified by paragraph indices
            sentence_spans, sentence_spans_indices = self.getSentenceSpans(self.paragraph_spans[start_index][0], self.paragraph_spans[end_index][1])
            first_sentence_index = sentence_spans_indices[0]
            last_sentence_index = sentence_spans_indices[1]
        else:
            raise ValueError("atom_type string must be 'char', 'word', 'sentence' or 'paragraph', not '" + str(atom_type) + "'.")

        return {
            'first_char_index' : first_char_index,
            'last_char_index' : last_char_index,
            'first_word_index' : first_word_index,
            'last_word_index' : last_word_index,
            'first_sentence_index' : first_sentence_index,
            'last_sentence_index' : last_sentence_index
        }

    def _getSumTableEntry(self, sum_table, index):
        if index < 0:
            return 0
        return sum_table[index]

    def averageWordLength(self, first_word_index, last_word_index):
        '''
        Returns the average word length of words between the given indicies into self.word_spans.
        
        TODO: Words that are just punctuation?
        '''
        total_word_length = self._getSumTableEntry(self.word_length_sum_table, last_word_index) - self._getSumTableEntry(self.word_length_sum_table, first_word_index-1)
        num_words = (last_word_index + 1) - first_word_index
        return float(total_word_length)/max(num_words, 1) # if there are no legitimate words, just set denominator to 1 to avoid division by 0
    
    def averageSentenceLength(self, first_sentence_index, last_sentence_index):
        '''
        Returns the average words-per-sentence for the sentences betwen the given indicies into self.sentence_spans.
        
        TODO: Words that are just punctuation?  
        '''
        total_words_per_sentences = self._getSumTableEntry(self.sentence_length_sum_table, last_sentence_index) - self._getSumTableEntry(self.sentence_length_sum_table, first_sentence_index-1)
        num_sentences = (last_sentence_index + 1) - first_sentence_index
        return float(total_words_per_sentences)/max(num_sentences, 1) # avoid division by 0

    def getPosPercentageVector(self, first_word_index, last_word_index):
        '''
        Take in two entries from the frequency count table and create a vector with percentages of
        PoS occurrences between those entries.
        '''
        endWord = self.pos_frequency_count_table[last_word_index]
        startWord = self.pos_frequency_count_table[first_word_index]
        newList = []
        for i in range(len(endWord)):
            newList.append((endWord[i] - startWord[i])/float(last_word_index - first_word_index))
        return newList

    def get_avg_word_frequency_class(self, first_word_index, last_word_index):
        '''
        Perform querying on the word_frequency_class_table and divide by the number of words
        '''
        total = self.word_frequency_class_table[last_word_index] - self.word_frequency_class_table[first_word_index]
        return total / float(last_word_index - first_word_index)

    def get_punctuation_percentage(self, first_char_index, last_char_index):
        '''
        Gets the punctuation count in the text between two indices 
        Note that <self.get_character_boundaries> indexes like a Python list, 
        i.e. the end index is NOT inclusive. If we are passed 
        first_char_index=0, last_char_index=5, the positions we should look at in the
        punctuation table are table[4] - table[0] (which is still 5 - 0 == 5 characters long)
        '''
        total = self.punctuation_table[last_char_index - 1] - self.punctuation_table[first_char_index]
        return total / float(last_char_index - first_char_index)

    def get_stopword_percentage(self, first_word_index, last_word_index):
        '''
        Return the percentage of words that are stop words in the text between the two given indicies.
        '''
        total = self.stopWords_table[last_word_index] - self.stopWords_table[first_word_index]
        return total / float(last_word_index - first_word_index)

    def get_syntactic_complexity(self, first_word_index, last_word_index):
        '''
        This feature is a modified version of the "Index of Syntactic Complexity" taken from
        Szmrecsanyi, Benedikt. "On operationalizing syntactic complexity." Jadt-04 2 (2004): 1032-1039. 
        found at http://www.benszm.net/omnibuslit/Szmrecsanyi2004.pdf
        it tallies various part-of-speech counts which are approximated by NLTK's POS tags.
        the original version accounts for Noun Phrases, which are not counted in the NLTK tagger, and so are ignored.
        '''
        num_conjunctions = self.pos_frequency_count_table[last_word_index][0] - self.pos_frequency_count_table[first_word_index][0]
        num_wh_pronouns = self.pos_frequency_count_table[last_word_index][1] - self.pos_frequency_count_table[first_word_index][1]
        num_verb_forms = self.pos_frequency_count_table[last_word_index][2] - self.pos_frequency_count_table[first_word_index][2]

        return 2 * num_conjunctions + 2 * num_wh_pronouns + num_verb_forms
    
    def _is_punctuation(self, word):
        ''' Returns true if the given word is just punctuation. '''
        match_obj = re.match(self.punctuation_re, word)
        return match_obj and len(match_obj.group()) == len(word)
        
    def test_general_extraction(self, atom_type='paragraph'):
        feature_list = [
            'averageWordLength',
            'averageSentenceLength',
            'getPosPercentageVector',
            'get_avg_word_frequency_class',
            'get_punctuation_percentage',
            'get_stopword_percentage',
            'get_syntactic_complexity'
        ]

        print 'Testing', atom_type, 'using features:', feature_list, '\n'
        all_of_atom_type = self.getAllByAtom(atom_type)
        for i in xrange(len(all_of_atom_type)):
            passage = self.get_specific_features(feature_list, i, i + 1, atom_type)
            print 'Passage Text:'
            print passage.text
            print 'Passage Features:'
            print passage.features
            print '-'*40 + '\n'
        

def independent_feature_test():
    '''
    This function writes data to two files. The data is used by an R script to generate box plots
    that show the significance of the given features within a corpus of project gutenberg texts.
    Returns None.
    '''
    
    same_doc_outfile = open('test_features_on_self.txt', 'w')
    similar_doc_outfile = open('test_features_on_similar.txt', 'w')
    different_doc_outfile = open('test_features_on_different.txt', 'w')
    percent_paragraphs = .10
    feature_tester = StylometricFeatureEvaluator(None)
    my_corpus = nltk.corpus.gutenberg
    feature_dict = {}
    for feature in ['averageWordLength', 'averageSentenceLength', 'get_punctuation_percentage', 'get_avg_word_frequency_class', 'get_stopword_percentage']:
        document_comparison_dict = {}
        for fileid in my_corpus.fileids():
            num_paragraphs = min(100, int(percent_paragraphs * len( my_corpus.paras(fileid))))
            main_file_feature_vector = []
            try:
                passages = feature_tester.extract_corpus_paragraph_features(random.sample(my_corpus.paras(fileid), num_paragraphs), feature)
            except ValueError:
                passages = feature_tester.extract_corpus_paragraph_features(my_corpus.paras(fileid), feature)
            for passage in passages:
                main_file_feature_vector.append(passage.features[feature])
            for other_file in my_corpus.fileids():
                pair = min(fileid, other_file) + "+" + max(fileid, other_file)
                if not document_comparison_dict.get(pair):
                    file_feature_vector = []
                    num_paragraphs = min(100, int(percent_paragraphs * len( my_corpus.paras(other_file))))
                    try:
                        passages = feature_tester.extract_corpus_paragraph_features(random.sample(my_corpus.paras(fileid), num_paragraphs), feature)
                    except ValueError:
                        passages = feature_tester.extract_corpus_paragraph_features(my_corpus.paras(fileid), feature)
                    for passage in passages:
                        file_feature_vector.append(passage.features[feature])
                    stats = scipy.stats.ttest_ind(main_file_feature_vector, file_feature_vector)
                    document_comparison_dict[pair] = stats[1]
                    print feature, fileid + " + " + other_file, stats[1]
                    if fileid == other_file:
                        outfile = same_doc_outfile
                    elif fileid.startswith(other_file.split('-')[0]):
                        outfile = similar_doc_outfile
                    else:
                        outfile = different_doc_outfile
                    outfile.write(feature + "\t" + fileid + "+" + other_file + "\t" + str(stats[1]) + "\r\n")
    
if __name__ == "__main__":
    evaluator = StylometricFeatureEvaluator("foo.txt")
    evaluator.test_general_extraction()

    #independent_feature_test()
