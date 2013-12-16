from .. import tokenization
from .. import spanutils

import nltk
import inspect
import string
import math

# To add a new feature, write a new method for FeatureExtractor.
# The method must take two arguments. They may be named (char_index_start,
# char_index_end), (word_spans_index_start, word_spans_index_end),
# (sent_spans_index_start, sent_spans_index_end), or (para_spans_index_start,
# para_spans_index_end).
#
# These arguments are indices into self.word_spans, self.sentence_spans, or
# self.paragraph_spans. These indices represent the first (inclusive) and last (exclusive)
# character, word, sentence, or paragraph that the feature is being extracted from.
#
# For example, average_word_length(4, 10) returns the average length of words 4 through 9.
#
# A certain amount of preprocessing may be desirable. Add a method called
# _init_my_new_feature_name and call it in the __init__ method.

class FeatureExtractor:
    
    def __init__(self, text):
        self.text = text
        self.word_spans = tokenization.tokenize(text, "word")
        self.sentence_spans = tokenization.tokenize(text, "sentence")
        self.paragraph_spans = tokenization.tokenize(text, "paragraph")
        self.pos_tags = self._init_tag_list(text)
    
        self.average_word_length_initialized = False
        self.average_sentence_length_initialized = False
        self.pos_percentage_vector_initialized = False
        self.stopword_percentage_initialized = False
        self.punctuation_percentage_initiliazed = False
        self.avg_internal_word_freq_class_initialized = False
        self.avg_external_word_freq_class_initialized = False
 
    def get_spans(self, atom_type):
        if atom_type == "word":
            return self.word_spans
        elif atom_type == "sentence":
            return self.sentence_spans
        elif atom_type == "paragraph":
            return self.paragraph_spans
        else:
            raise ValueError("Invalid atom_type")
    
    def _init_tag_list(self, text):
        '''
        Return a list of tuples of (word, part of speech) where the ith item in the list is the 
        (ith word, ith part of speech) from the text. Used to give a value to self.pos_tags.
        '''

        taggedWordTuples = []
        sentenceSpans = spanutils.slice(self.sentence_spans, 0, len(text), return_indices = False)
        assert(self.sentence_spans == sentenceSpans)
        for sentence in sentenceSpans:
            sentence = self.text[sentence[0]:sentence[1]]
            # run part-of-speech tagging on the sentence following the word_tokenize-ation of it
            taggedWordTuples += nltk.tag.pos_tag(nltk.word_tokenize(sentence))

        return taggedWordTuples
    
    def get_feature_vectors(self, features, atom_type):
        '''
        Return feature vectors (e.g. (4.3, 12, 0.05)) for each passage in the text
        (as parsed by atom_type). Each feature vector contains components for each
        feature in features. The components of the returned feature vectors are in an
        order corresponding to the order of the features argument.
        '''
        passage_spans = None
        if atom_type == "word":
            passage_spans = self.word_spans
        elif atom_type == "sentence":
            passage_spans = self.sentence_spans
        elif atom_type == "paragraph":
            passage_spans = self.paragraph_spans
        else:
            raise ValueError("Unacceptable atom_type value")
        
        vectors = []
        for passage_span in passage_spans:
            vectors.append(self._get_feature_vector(features, passage_span[0], passage_span[1]))
        
        return vectors
    
    def _get_feature_vector(self, features, start_index, end_index):
        '''
        Return a feature vector (e.g. (4.3, 12, 0.05)) with a component for each string in
        *features* for the text between the two given indices.
        This method "snaps-out".
        '''
        #TODO: Explain the snapping
        
        vect = []
        for feat_name in features:
            if self._feature_type(feat_name) == "char": 
                start, end = start_index, end_index
            else:                
                if self._feature_type(feat_name) == "word":
                    spans = self.word_spans
                elif self._feature_type(feat_name) == "sentence":
                    spans = self.sentence_spans
                elif self._feature_type(feat_name) == "paragraph":
                    spans = self.paragraph_spans
                start, end = spanutils.slice(spans, start_index, end_index, return_indices = True)
                
            actual_feature_function = getattr(self, feat_name)
            vect.append(actual_feature_function(start, end))
            
        return tuple(vect)
    
    def _feature_type(self, func_name):
        '''
        Return at what atom level the feature with the given name operates.
        For example _feature_type("average_word_length") returns "word".
        '''
        try:
            func = getattr(self, func_name)
        except AttributeError:
            raise ValueError("Invalid feature name.")
            
        accepted_params = inspect.getargspec(func).args
        if "char_index_start" in accepted_params:
            return "char"
        elif "word_spans_index_start" in accepted_params:
            return "word"
        elif "sent_spans_index_start" in accepted_params:
            return "sentence"
        elif "para_spans_index_start" in accepted_params:
            return "paragraph"
        else:
            raise ValueError


    
    def _init_average_word_length(self):
        '''
        Initializes the word_length_sum_table. word_length_sum_table[i] is the sum of the
        lengths of words from 0 to i-1.
        '''
        # TODO: Check if words are punctuation/have punctuation on them?
        
        sum_table = [0]
        for start, end in self.word_spans:
            sum_table.append((end - start) + sum_table[-1])
        self.word_length_sum_table = sum_table
        
        self.average_word_length_initialized = True
    
    def average_word_length(self, word_spans_index_start, word_spans_index_end):
        '''
        Return the average word length for words [word_spans_index_start : word_spans_index_end].
        
        For example: If self.text = "The brown fox jumped" then self.word_spans = [(0, 3),
        (4, 9), (10, 13), (14, 20)]. So, average_word_length(1, 3) returns 4, the average
        length of "brown" and "fox" (which are designated by the spans (4, 9) and
        (10, 13)).
        '''
        if not self.average_word_length_initialized:
            self._init_average_word_length()
            
        total_word_length = self.word_length_sum_table[word_spans_index_end] - self.word_length_sum_table[word_spans_index_start]
        num_words = word_spans_index_end - word_spans_index_start
        return float(total_word_length) / max(num_words, 1)
    
    def _init_average_sentence_length(self):
        '''
        Initializes the sentence_length_sum_table. sentence_length_sum_table[i] is the sum of the number
        of words in sentences 0 to i-1.
        '''
        #TODO: Check if words are punctuation?
        sum_table = [0]
        for start, end in self.sentence_spans:
            word_sum = 0
            word_spans = spanutils.slice(self.word_spans, start, end, return_indices = False)
            word_sum += len(word_spans)
            sum_table.append(word_sum + sum_table[-1])
        self.sent_length_sum_table = sum_table
        
        self.average_sentence_length_initialized = True
    
    def average_sentence_length(self, sent_spans_index_start, sent_spans_index_end):
        '''
        Return the average number of words in sentences [sent_spans_index_start : sent_spans_index_end]
        '''
        if not self.average_sentence_length_initialized:
            self._init_average_sentence_length()
        
        total_sentence_length = self.sent_length_sum_table[sent_spans_index_end] - self.sent_length_sum_table[sent_spans_index_start]
        num_words = sent_spans_index_end - sent_spans_index_start
        return float(total_sentence_length) / max(num_words, 1)
    
    def _init_pos_frequency_table(self):
    	'''
    	instantiates a table of part-of-speech counts 
        currently tracks the following categories:
        0) conjunctions -- tags CC, IN (though we will try to ignore common prepositions that are not conjunctions)
        1) WH-pronouns -- tags WP, WP$
        2) Verbs -- tags VB, VBD, VBG, VBN, VBP, VBZ
        3) None of the above
    	'''
        sum_table = [[0,0,0,0]]
        total_count = [0,0,0,0]
        for posTuple in self.pos_tags:
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

        self.pos_frequency_count_table = sum_table
        
        self.pos_percentage_vector_initialized = True
    
    def pos_percentage_vector(self, word_spans_index_start, word_spans_index_end):
        # TODO: What the hell is this feature?
        # Oh... This feature is a vector itself? not a single value...
        if not self.pos_percentage_vector_initialized:
            self._init_pos_frequency_table()
        
        total_vect = [a - b for a, b in zip(self.pos_frequency_count_table[word_spans_index_end], self.pos_frequency_count_table[word_spans_index_start])]
        num_words = word_spans_index_end - word_spans_index_start
        return tuple([a / float(num_words) for a in total_vect])
    
    def _init_stopword_percentage(self):
    	'''
    	instatiates the table for stopword counts which allows for constant-time
    	querying of stopword percentages within a particular passage
    	'''
        sum_table = [0]
        count = 0
        for span in self.word_spans:
            word = self.text[span[0]:span[1]]
            word = word.lower()
            # This line courtesy of http://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string-in-python
            word = word.translate(string.maketrans("",""), string.punctuation)
            if word in nltk.corpus.stopwords.words('english'):
                count += 1
            sum_table.append(count)
        self.stopword_sum_table = sum_table
        
        self.stopword_percentage_initialized = True
        
    def stopword_percentage(self, word_spans_index_start, word_spans_index_end):
        '''
        Return the percentage of words that are stop words in the text between the two given indices.
        '''
        if not self.stopword_percentage_initialized:
            self._init_stopword_percentage()
        
        total_stopwords = self.stopword_sum_table[word_spans_index_end] - self.stopword_sum_table[word_spans_index_start]
        num_sents = word_spans_index_end - word_spans_index_start
        return float(total_stopwords) / max(num_sents, 1)
        
    def _init_punctuation_percentage(self):
    	'''
    	instatiates the table for the punctuation counts which allows for constant-time
    	querying of punctuation percentages within a particular passage
    	'''
        sum_table = [0]
        count = 0
        for char in self.text:
            if char in ",./<>?;':[]{}\|!@#$%^&*()`~-_\"": # Use string.punctuation here instead? string.punctuation also includes "+" and "=".
                count += 1
            sum_table.append(count)
        self.punctuation_sum_table = sum_table
        
        self.punctuation_percentage_initiliazed = True
    
    def punctuation_percentage(self, char_index_start, char_index_end):
        if not self.punctuation_percentage_initiliazed:
            self._init_punctuation_percentage()
        
        total_punctuation = self.punctuation_sum_table[char_index_end] - self.punctuation_sum_table[char_index_start]
        num_chars = char_index_end - char_index_start
        return float(total_punctuation) / max(num_chars, 1)

    def syntactic_complexity(self, word_spans_index_start, word_spans_index_end):
        '''
        This feature is a modified version of the "Index of Syntactic Complexity" taken from
        Szmrecsanyi, Benedikt. "On operationalizing syntactic complexity." Jadt-04 2 (2004): 1032-1039. 
        found at http://www.benszm.net/omnibuslit/Szmrecsanyi2004.pdf
        it tallies various part-of-speech counts which are approximated by NLTK's POS tags.
        the original version accounts for Noun Phrases, which are not counted in the NLTK tagger, and so are ignored.
        '''
        # Note that this feature uses the same initialization that pos_percentage_vector does.
        if not self.pos_percentage_vector_initialized:
            self._init_pos_frequency_table()
            
        num_conjunctions = self.pos_frequency_count_table[word_spans_index_end][0] - self.pos_frequency_count_table[word_spans_index_start][0]
        num_wh_pronouns = self.pos_frequency_count_table[word_spans_index_end][1] - self.pos_frequency_count_table[word_spans_index_start][1]
        num_verb_forms = self.pos_frequency_count_table[word_spans_index_end][2] - self.pos_frequency_count_table[word_spans_index_start][2]

        return 2 * num_conjunctions + 2 * num_wh_pronouns + num_verb_forms
    
    def _init_internal_word_freq_class(self):
        '''
        Initializes the internal_freq_class_table. internal_freq_class_table[i]
        is the sum of the classes of words 0 to i-1.
        '''
        word_freq_dict = {}
        for span in self.word_spans:
            word = self.text[span[0]:span[1]].lower().translate(string.maketrans("",""), string.punctuation)
            word_freq_dict[word] = word_freq_dict.get(word, 0) + 1
        
        corpus_word_freq_by_rank = sorted(word_freq_dict.items(), key=lambda x: x[1], reverse=True)
        occurences_of_most_freq_word = corpus_word_freq_by_rank[0][1]
    
        # this creates a table where each entry holds the summation of the WFC of all previous words
        table = [0]
        total = 0
        for i in range(len(self.word_spans)):
            span = self.word_spans[i]
            word = self.text[span[0]:span[1]].lower().translate(string.maketrans("",""), string.punctuation)         
            freq_class = math.floor(math.log((float(occurences_of_most_freq_word)/word_freq_dict.get(word, 0)),2))
            total += freq_class
            table.append(total)
        self.internal_freq_class_table = table
        self.avg_internal_word_freq_class_initialized = True
    
    def avg_internal_word_freq_class(self, word_spans_index_start, word_spans_index_end):
        '''
        Returns a feature like avg_external_word_freq_class except that the frequenecy
        classes of words are calculated based on the occurrences of words within this
        text, not the brown corpus.
        '''
        if not self.avg_internal_word_freq_class_initialized:
            self._init_internal_word_freq_class()
        total = self.internal_freq_class_table[word_spans_index_end] - self.internal_freq_class_table[word_spans_index_start]
        return total / float(max(1, word_spans_index_end - word_spans_index_start))
        
    def _init_external_word_freq_class(self):

        word_freq_dict = {}
        for word in nltk.corpus.brown.words():
            word = word.lower()
            word_freq_dict[word] = word_freq_dict.get(word, 0) + 1
        
        corpus_word_freq_by_rank = sorted(word_freq_dict.items(), key=lambda x: x[1], reverse=True)
        occurences_of_most_freq_word = corpus_word_freq_by_rank[0][1]
    
        class_sum_table = [0]
    
        for span in self.word_spans:
            word = self.text[span[0]:span[1]].lower().translate(string.maketrans("",""), string.punctuation)
            # plus one smoothing is used here!
            freq_class = math.floor(math.log(float(occurences_of_most_freq_word + 1) / (word_freq_dict.get(word.lower(), 0) + 1),2))
            class_sum_table.append(class_sum_table[-1] + freq_class)
        
        self.external_freq_class_table = class_sum_table
        self.avg_external_word_freq_class_initialized = True
        
    def avg_external_word_freq_class(self, word_spans_index_start, word_spans_index_end):
        '''
        This feature is defined here:
        http://www.uni-weimar.de/medien/webis/publications/papers/stein_2006d.pdf
        
        Plus one smoothing is used.
        '''
        if not self.avg_external_word_freq_class_initialized:
            self._init_external_word_freq_class()
        total = self.external_freq_class_table[word_spans_index_end] - self.external_freq_class_table[word_spans_index_start]
        num_words = word_spans_index_end - word_spans_index_start
        return float(total) / max(num_words, 1)
        

def _test():

    text = "The brown fox ate. Believe it. I go to the school." # length = 50. Last valid index is 49
    f = FeatureExtractor(text)
    
    #print f.get_feature_vectors(["punctuation_percentage"], "sentence")
    if f.get_feature_vectors(["punctuation_percentage"], "sentence") == [(0.05555555555555555,), (0.09090909090909091,), (0.05263157894736842,)]:
        print "punctuation_percentage test passed"
    else:
        print "punctuation_percentage test FAILED"
    
    #print f.get_feature_vectors(["stopword_percentage"], "sentence")
    if f.get_feature_vectors(["stopword_percentage"], "sentence") == [(0.25,), (0.5,), (0.6,)]:
        print "stopword_percentage test passed"
    else:
        print "stopword_percentage test FAILED"
    
    #print f.get_feature_vectors(["average_sentence_length"], "paragraph")
    if f.get_feature_vectors(["average_sentence_length"], "paragraph") == [(3.6666666666666665,)]:
        print "average_sentence_legth test passed"
    else:
        print "average_sentence_legth test FAILED"
    
    #print f.get_feature_vectors(["average_word_length"], "sentence")
    if f.get_feature_vectors(["average_word_length"], "sentence") == [(3.75,), (5.0,), (3.0,)]:
        print "average_word_length test passed"
    else:
        print "average_word_length test FAILED"

    # TODO: ADD TEST FOR INTERNAL WORD FREQ CLASS
    f.get_feature_vectors(["avg_internal_word_freq_class"], "sentence")
    print "avg_internal_word_freq_class test DOES NOT EXIST" 
   
    
    # TODO: ADD TEST FOR EXTERNAL WORD FREQ CLASS
    f.get_feature_vectors(["avg_external_word_freq_class"], "sentence")
    print "avg_external_word_freq_class DOES NOT EXIST"
    

    f = FeatureExtractor("The brown fox ate. I go to the school. Believe it.")
    #print f.get_feature_vectors(["pos_percentage_vector"], "sentence")
    if f.get_feature_vectors(["pos_percentage_vector"], "sentence") == [(0,0,0,1.0), (0,0,.2,.8), (0,0,0,1.0)]:
        print "pos_percentage_vector test passed"
    else:
        print "pos_percentage_vector test FAILED"
    
    f = FeatureExtractor("The brown fox ate. I go to the school. Believe it. I go.")
    #print f.get_feature_vectors(["syntactic_complexity"], "sentence")
    if f.get_feature_vectors(["syntactic_complexity"], "sentence") == [(0,), (1,), (0,), (1,)]:
        print "syntactic_complexity test passed"
    else:
        print "syntactic_complexity test FAILED"
    
if __name__ == "__main__":
    _test()
    
    
    
    
    
    
    
    

