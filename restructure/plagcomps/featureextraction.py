import tokenization
import spanutils

import inspect

# To add a new feature, write a new method for FeatureExtractor.
# The method must take two arguments. They may be named (word_span_index_start,
# word_span_index_end), (sent_span_index_start, sent_span_index_end), or
# (para_span_index_start, para_span_index_end).
#
# These arguments are indicies into self.word_spans, self.sentence_spans, or
# self.paragraph_spans. These indicies represent the first and last (inclusive) word,
# sentence, or paragraph that the feature is being extracted from.
#
# For example, average_word_length(4, 10) returns the average length of words 4 through 10
# (inclusive).
#
# A certain amount of preprocessing may be desirable. Add a method called
# _init_my_new_feature_name and call it in the __init__ method.

class FeatureExtractor:
    
    def __init__(self, text):
        self.text = text
        self.word_spans = tokenization.tokenize(text, "word")
        self.sentence_spans = tokenization.tokenize(text, "sentence")
        self.paragraph_spans = tokenization.tokenize(text, "paragraph")
    
        ### ADD FEATURE INITIALIZATION METHODS HERE:
        
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
        This method "snaps-out"
        '''
        #TODO: Explain the snapping
        
        vect = []
        for feat_name in features:
            if self._feature_type(feat_name) == "word":
                spans = self.word_spans
            elif self._feature_type(feat_name) == "sentence":
                spans = self.sentence_spans
            elif self._feature_type(feat_name) == "paragraph":
                spans = self.paragraph_spans
        
            start, end = spanutils.snapout(spans, start_index, end_index)
            
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
        if "word_span_index_start" in accepted_params:
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
    
    def average_word_length(self, word_span_index_start, word_span_index_end):
        total_length = 0
        words = 0.0
        for i in range(word_span_index_start, word_span_index_end + 1):
            words += 1
            total_length += self.word_spans[i][1] - self.word_spans[i][0]
        return total_length/words

def _test():
    text = "The brown fox ate."
    f = FeatureExtractor(text)
    print f.get_feature_vectors(["average_word_length"], "sentence")

if __name__ == "__main__":
    _test()
    print "done"
    
    
    
    
    
    
    
    

