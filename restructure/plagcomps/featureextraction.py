import tokenization
import spanutils

import nltk
import inspect
import string

# To add a new feature, write a new method for FeatureExtractor.
# The method must take two arguments. They may be named (char_index_start,
# char_index_end), (word_span_index_start, word_span_index_end),
# (sent_span_index_start, sent_span_index_end), or (para_span_index_start,
# para_span_index_end).
#
# These arguments are indicies into self.word_spans, self.sentence_spans, or
# self.paragraph_spans. These indicies represent the first (inclusive) and last (exclusive)
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
    
        ### ADD FEATURE INITIALIZATION METHODS HERE:
        self._init_average_word_length()
        self._init_average_sentence_length()
        self._init_pos_frequency_table()
        self._init_stopword_percentage()
        self._init_punctuation_percentage()
    
    def _init_tag_list(self, text):
        '''
        Return a list of tuples of (word, part of speech) where the ith item in the list is the 
        (ith word, ith part of speech) from the text. Used to give a value to self.pos_tags.
        '''

        taggedWordTuples = []
        sentenceSpans = spanutils.slice(self.sentence_spans, 0, len(text), return_indicies = False)
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
        *features* for the text between the two given indicies.
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
                start, end = spanutils.slice(spans, start_index, end_index, return_indicies = True)
                
            actual_feature_function = getattr(self, feat_name)
            vect.append(actual_feature_function(start, end))
            
        return tuple(vect)
    
    def _feature_type(self, func_name):
        '''
        Return at what atom level the feature with the given name operates.
        For example _feature_type("average_word_length") returns "word".
        '''
        func = getattr(self, func_name)
        accepted_params = inspect.getargspec(func).args
        if "char_index_start" in accepted_params:
            return "char"
        elif "word_span_index_start" in accepted_params:
            return "word"
        elif "sent_span_index_start" in accepted_params:
            return "sentence"
        elif "para_span_index_start" in accepted_params:
            return "paragraph"
        else:
            raise ValueError


    
    # an example of an actual feature function:
    def _my_word_level_feature(self, word_span_index_start, word_span_index_end):
        '''
        Return a value for this feature for the words from word_span_index_start to
        word_span_index_end.
        
        For example: If self.text = "The brown fox jumped" then self.word_spans = [(0, 3),
        (4, 9), (10, 13), (14, 20)]. So, _my_word_level_feature(1, 3) returns what ever
        the value of this feature is on the words designated by the spans (4, 9) and
        (10, 13), that is, "brown" and "fox".
        '''


    
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
    
    def average_word_length(self, word_span_index_start, word_span_index_end):
        '''
        Return the average word length for words [word_span_index_start : word_span_index_end].
        
        For example: If self.text = "The brown fox jumped" then self.word_spans = [(0, 3),
        (4, 9), (10, 13), (14, 20)]. So, average_word_length(1, 3) returns 4, the average
        length of "brown" and "fox" (which are designated by the spans (4, 9) and
        (10, 13)).
        '''
        
        total_word_length = self.word_length_sum_table[word_span_index_end] - self.word_length_sum_table[word_span_index_start]
        num_words = word_span_index_end - word_span_index_start
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
            word_spans = spanutils.slice(self.word_spans, start, end, return_indicies = False)
            word_sum += len(word_spans)
            sum_table.append(word_sum + sum_table[-1])
        self.sent_length_sum_table = sum_table
    
    def average_sentence_length(self, sent_span_index_start, sent_span_index_end):
        '''
        Return the average number of words in sentences [sent_span_index_start : sent_span_index_end]
        '''
        total_sentence_length = self.sent_length_sum_table[sent_span_index_end] - self.sent_length_sum_table[sent_span_index_start]
        num_words = sent_span_index_end - sent_span_index_start
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
    
    def pos_percentage_vector(self, word_span_index_start, word_span_index_end):
        # TODO: What the hell is this feature?
        # Oh... This feature is a vector itself? not a single value...
        
        total_vect = [a - b for a, b in zip(self.pos_frequency_count_table[word_span_index_end], self.pos_frequency_count_table[word_span_index_start])]
        num_words = word_span_index_end - word_span_index_start
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
        
    def stopword_percentage(self, word_span_index_start, word_span_index_end):
        '''
        Return the percentage of words that are stop words in the text between the two given indicies.
        '''
        total_stopwords = self.stopword_sum_table[word_span_index_end] - self.stopword_sum_table[word_span_index_start]
        num_sents = word_span_index_end - word_span_index_start
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
    
    def punctuation_percentage(self, char_index_start, char_index_end):
        
        total_punctuation = self.punctuation_sum_table[char_index_end] - self.punctuation_sum_table[char_index_start]
        num_chars = char_index_end - char_index_start
        return float(total_punctuation) / max(num_chars, 1)

def _test():
    text = "The brown fox ate. Believe it. I go to the school." # length = 50. Last valid index is 49
    f = FeatureExtractor(text)
    
    print f.get_feature_vectors(["punctuation_percentage"], "sentence")
    for start, end in f.sentence_spans:
        print f.text[start:end]


if __name__ == "__main__":
    _test()
    
    
    
    
    
    
    
    

