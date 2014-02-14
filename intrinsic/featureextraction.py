from .. import tokenization
from .. import spanutils
from ..shared.passage import IntrinsicPassage

import nltk
import inspect
import string
import math
import cPickle
import os.path
from itertools import groupby

from nltk.corpus import cmudict

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
# A certain amount of preprocessing may be desirable. Add a method called
# _init_my_new_feature_name and call it in the __init__ method.

class FeatureExtractor:

    @staticmethod
    def get_all_feature_function_names(include_nested=False):
        '''
        Returns the names of all functions which compute features by
        examining their arguments.

        If <include_nested>, then functions like 'char_to_word_average',
        'sentence_to_paragraph_average' and others which can be nested
        are also returned.

        By default, nested features are NOT returned
        '''
        feature_function_names = []

        feature_arg_options = set([
            'char_index_start', 
            'word_spans_index_start', 
            'sent_spans_index_start', 
            'para_spans_index_start'
        ])

        # all_methods[i] == (<func_name>, <unbound_method_obj>)
        all_methods = inspect.getmembers(FeatureExtractor, predicate=inspect.ismethod)

        # We will track different kinds of features to include nested features
        char_features, word_features, sent_features = [], [], []
        not_nestable = ["word_unigram", "pos_trigram", "vowelness_trigram"]

        for func_name, func in all_methods:
            func_args = set(inspect.getargspec(func).args)

            valid_func = len(feature_arg_options.intersection(func_args)) > 0


            # Has overlap and is not a helper function for nested features
            if valid_func and 'subfeatures' not in func_args:
                feature_function_names.append(func_name)

                # if we want nested features, save functions in their respective lists
                if include_nested and func_name not in not_nestable:
                    if "char_spans_index_start" in func_args:
                        char_features.append(func_name)
                    if "word_spans_index_start" in func_args:
                        word_features.append(func_name)
                    if "sent_spans_index_start" in func_args:
                        sent_features.append(func_name)

        if include_nested:
            # If we want nested features, we will now add in all allowable nestings
            for char_feature in char_features:
                for nesting in [["avg"], ["std"], ["avg", "avg"], ["avg", "std"], ["avg", "avg", "avg"], ["avg", "avg", "std"]]:
                    feature_function_names.append("(".join(nesting) + "(" + char_feature + ")" * len(nesting))
            for word_feature in word_features:
                for nesting in [["avg"], ["std"], ["avg", "avg"], ["avg", "std"]]:
                    feature_function_names.append("(".join(nesting) + "(" + word_feature + ")" * len(nesting))
            for sent_feature in sent_features:
                for nesting in [["avg"], ["std"]]:
                    feature_function_names.append("(".join(nesting) + "(" + sent_feature + ")" * len(nesting))
        
        # pos_trigram, word_unigram, and vowelness_trigram are special cases
        valid_pos_trigrams = [
            ("NN", "VB", "NN"),
            ("NN", "NN", "VB"),
            ("VB", "NN", "NN"),
            ("NN", "IN", "NP"), # noun preposition propernoun
            ("NN", "NN", "CC"), # noun noun coordinatingconjunction
            ("NNS", "IN", "DT"), # nounplural preposition determiner
            ("DT", "NNS", "IN"), # determiner nounplural preposition
            ("VB", "NN", "VB"),
            ("DT", "NN", "IN"), # determiner noun preposition
            ("NN", "NN", "NN"),
            ("NN", "IN", "DT"), # noun preposition determiner
            ("NN", "IN", "NN"), # noun preposition noun
            ("VB", "IN", "DT"), #verb preposition determiner
        ]
        feature_function_names.remove("pos_trigram")
        for tags in valid_pos_trigrams:
            feature_function_names.append("pos_trigram,%s,%s,%s" % tags)
            
        valid_vowelness_trigrams = [
            ("C", "V", "C"),
            ("C", "V", "V"),
            ("V", "V", "C"),
            ("V", "V", "V"),
            
        ]
        feature_function_names.remove("vowelness_trigram")
        for tris in valid_vowelness_trigrams:
            feature_function_names.append("vowelness_trigram,%s,%s,%s" % tris)
            
        valid_word_unigrams = ["is", "of", "been", "the"]
        feature_function_names.remove("word_unigram")
        for w in valid_word_unigrams:
            feature_function_names.append("word_unigram,%s" % w)
        

        return feature_function_names

    
    @staticmethod
    def get_stein_paper_feature_names():
        return [
            'average_word_length',
            'average_syllables_per_word',
            'flesch_kincaid_grade',
            'gunning_fog_index',
            'honore_r_measure',
            'yule_k_characteristic',
            # Unclear whether paper uses internal or external
            'avg_external_word_freq_class'
        ]

    def __init__(self, text, char_span_length=5000):
        self.text = text
        self.word_spans = tokenization.tokenize(text, "word")
        self.sentence_spans = tokenization.tokenize(text, "sentence")
        self.paragraph_spans = tokenization.tokenize(text, "paragraph")
        self.nchar_spans = tokenization.tokenize(text, "nchars", char_span_length)
    
        self.features = {}

        self._cmud = None # initialized by self._syl()
        self.pos_tags = None
        self.pos_tagged = False
        self.lancaster_stemmer = None

        #TODO evaluate whether we want to keep this?
        self.pos_frequency_count_table_initialized = False
 
    def get_spans(self, atom_type):
        if atom_type == "word":
            return self.word_spans
        elif atom_type == "sentence":
            return self.sentence_spans
        elif atom_type == "paragraph":
            return self.paragraph_spans
        elif atom_type == "nchars":
            return self.nchar_spans
        else:
            raise ValueError("Invalid atom_type")
    
    def _init_tag_list(self, text):
        '''
        Sets self.pos_tags to be a list of tuples of (word, part of speech) where the ith tuple in the list is the 
        (ith word, ith part of speech) from the text. 
        '''

        taggedWordTuples = []
        sentenceSpans = spanutils.slice(self.sentence_spans, 0, len(text), return_indices = False)
        assert(self.sentence_spans == sentenceSpans)
        for sentence in sentenceSpans:
            sentence = self.text[sentence[0]:sentence[1]]
            # run part-of-speech tagging on the sentence following the word_tokenize-ation of it
            tokens = tokenization.tokenize(sentence, 'word', return_spans=False)
            taggedWordTuples += nltk.tag.pos_tag(tokens)

        self.pos_tags = taggedWordTuples
        self.pos_tagged = True

    def get_passages(self, features, atom_type):
        '''
        Return a list of IntrinsicPassage objects for each passage in the text
        (as parsed by atom_type). Each passage object contains metadata about
        the passage (starting/ending index, actual text etc. See IntrinsicPassage class
        for more details) and a dictionary of its features.
        '''
        passage_spans = self.get_spans(atom_type)
        feature_vecs = self.get_feature_vectors(features, atom_type)
        all_passages = []

        for span, feature_vals in zip(passage_spans, feature_vecs):
            # Parse out relevant data to pass into the IntrinsicPassage object
            start, end = span
            text = self.text[start : end]
            feat_dict = {f : fval for f, fval in zip(features, feature_vals)}

            passage = IntrinsicPassage(start, end, text, atom_type, feat_dict)
            all_passages.append(passage)

        return all_passages

    def get_feature_vectors(self, features, atom_type):
        '''
        Return feature vectors (e.g. (4.3, 12, 0.05)) for each passage in the text
        (as parsed by atom_type). Each feature vector contains components for each
        feature in features. The components of the returned feature vectors are in an
        order corresponding to the order of the features argument.
        '''
        passage_spans = self.get_spans(atom_type)

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
            # if the feature has a ( in it, we assume it includes a meta-feature such as "avg( )"
            if '(' in feat_name:
                nested_feature_name = self._deconstruct_nested_feature_name(feat_name)
                feat_name = nested_feature_name[-1]
            else:
                nested_feature_name = None 
            
            if nested_feature_name != None:
                #print "nfc", nested_feature_name
                # Each meta-feature we apply (avg, std, etc.) "expands" the feature type
                # for example, num_chars is a word feature, avg(num_chars) is a setence feature,
                #    and avg(avg(num_chars)) is a paragraph feature

                subfeatures = []
                subfeatures.insert(0, nested_feature_name.pop(-1))

                # We will index the atom types in the following array so that we can easily advance them
                atom_types = ["char", "word", "sentence", "paragraph"]
                cur_atom_type = self._feature_type(feat_name)
                #print "my cur atom type is", cur_atom_type

                while len(nested_feature_name) > 0:
                    try:
                        cur_atom_type = atom_types[atom_types.index(cur_atom_type) + 1]
                    except IndexError:
                        raise ValueError("You tried to apply too many metafeatures! You can not apply a metafeature to a paragraph-level feature.")
                    cur_meta_feature = nested_feature_name.pop(-1)
                    #print "my cur atom type is", cur_atom_type

                    if cur_meta_feature == "avg":
                        if cur_atom_type == "word":
                            subfeatures.insert(0, "char_to_word_average")
                        elif cur_atom_type == "sentence":
                            subfeatures.insert(0, "word_to_sentence_average")
                        elif cur_atom_type == "paragraph":
                            subfeatures.insert(0, "sentence_to_paragraph_average")
                        else:
                           raise ValueError("Unable to average that kind of feature")
                    
                    elif cur_meta_feature == "std":
                        if cur_atom_type == "word":
                            subfeatures.insert(0, "char_to_word_std")
                        elif cur_atom_type == "sentence":
                            subfeatures.insert(0, "word_to_sentence_std")
                        elif cur_atom_type == "paragraph":
                            subfeatures.insert(0, "sentence_to_paragraph_std")
                        else:
                           raise ValueError("Unable to std that kind of feature")
                    
                
                # Set the spans of our meta feature to be on the most external call
                # for example, if we are doing avg(avg(num_chars)), then the most-outer average
                # gives us a paragraph feature, and we should therefore have paragraph spans
                if cur_atom_type == "char": 
                    start, end = start_index, end_index
                else:                
                    if cur_atom_type == "word":
                        spans = self.word_spans
                    elif cur_atom_type == "sentence":
                        spans = self.sentence_spans
                    elif cur_atom_type == "paragraph":
                        spans = self.paragraph_spans
                    start, end = spanutils.slice(spans, start_index, end_index, return_indices = True)
                vect.append(self.meta_feature(subfeatures, start, end))
                    
            else:
                # No nested features, so things are easy

                #special case for pos_trigram, vowelness_trigram, word_unigram
                if feat_name.startswith("pos_trigram"):
                    feat_func, first, second, third = feat_name.split(",")
                    start, end = spanutils.slice(self.word_spans, start_index, end_index, return_indices = True)
                    vect.append(getattr(self, feat_func)(start, end, (first, second, third)))
                    
                elif feat_name.startswith("vowelness_trigram"):
                    feat_func, first, second, third = feat_name.split(",")
                    vect.append(getattr(self, feat_func)(start_index, end_index, (first, second, third)))
                    
                elif feat_name.startswith("word_unigram"):
                    feat_func, target_word = feat_name.split(",")
                    start, end = spanutils.slice(self.word_spans, start_index, end_index, return_indices = True)
                    vect.append(getattr(self, feat_func)(start, end, target_word))
                
                
                else:
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

    def _deconstruct_nested_feature_name(self, feat_name):    
        '''
        Decompose a nested feature call such as "avg(std(char_length))" into [avg, std, char_length]
        '''
        return_list = []
        while '(' in feat_name:
            paren_index = feat_name.find('(')
            func = feat_name[:paren_index]
            return_list.append(func)
            feat_name = feat_name[paren_index + 1:]
        return_list.append(feat_name.replace(")", ""))

        return return_list

    def _construct_nested_feature_name(self, feat_list):    
        '''
        compose a nested feature call such as "avg(std(char_length))" from [avg, std, char_length]
        '''
        return "(".join(feat_list) + ")" * (len(feat_list) - 1)

    def _feature_type(self, func_name):
        '''
        Return at what atom level the feature with the given name operates.
        For example _feature_type("num_chars") returns "word".
        '''
        try:
            func = getattr(self, func_name)
        except AttributeError:
            raise ValueError("Invalid feature name: %s" % func_name)
            
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

    def _init_num_chars(self):
        '''count the number of characters within words'''

        sum_table = [0]
        for start, end in self.word_spans:
            sum_table.append((end - start) + sum_table[-1])
        self.features["num_chars"] = sum_table

    def num_chars(self, word_spans_index_start, word_spans_index_end):
        '''return the number of characters used in these words'''
        if "num_chars" not in self.features:
            self._init_num_chars()

        return self.features["num_chars"][word_spans_index_end] - self.features["num_chars"][word_spans_index_start]

    def meta_feature(self, subfeatures, spans_index_start, spans_index_end):
        '''
        wrapper function that calls the first subfeature with the rest of the subfeatures as its argument
        for example:
        self.meta_feature([word_to_sentence_average, num_chars], 0, 10)
        will return the average number of characters per word pulled from the first 10 sentences
        '''
        #print "running a metafeature with", subfeatures

        outer_feature = getattr(self, subfeatures[0])
        inner_features = subfeatures[1:]
        if len(inner_features) > 0:
            return outer_feature(subfeatures[1:], spans_index_start, spans_index_end) 
        else:
            return outer_feature(spans_index_start, spans_index_end)
    
    def sentence_to_paragraph_average(self, subfeatures, para_spans_index_start, para_spans_index_end):
        '''
        takes a sentence feature (or a sentence-level meta-feature) and applies it to every sentence
        in the specified paragraphs, and then averages its result over the number of sentences
        '''
        # get the start and end character indices from our paragraph indices
        start = self.paragraph_spans[para_spans_index_start][0]
        end = self.paragraph_spans[para_spans_index_end - 1][1]
        # get the index range for the words we will consider
        sentence_index_range = spanutils.slice(self.sentence_spans, start, end, return_indices = True)

        # if we only have a single subfeature, we will use that as our sentence feature
        if len(subfeatures) == 1:
            sentence_feature = getattr(self, subfeatures[0])
            # sum the values of our sentence feature over each of our sentences    
            sentence_sum = sentence_feature(sentence_index_range[0], sentence_index_range[1])
        else:
            sentence_sum = self.meta_feature(subfeatures, sentence_index_range[0], sentence_index_range[1])

        # return the average
        return float(sentence_sum) / max(1, sentence_index_range[1] - sentence_index_range[0]) 

    def word_to_sentence_average(self, subfeatures, sent_spans_index_start, sent_spans_index_end):
        '''
        takes in a word feature (or a word-level meta-feature) and sentence span range
        applies the word feature to each word in the specified sentences, and then averages the results
        over the number of words.'''

        # get the start and end character indices from our sentence indices
        start = self.sentence_spans[sent_spans_index_start][0]
        end = self.sentence_spans[sent_spans_index_end - 1][1]
        # get the index range for the words we will consider
        word_index_range = spanutils.slice(self.word_spans, start, end, return_indices = True)

        # if we only have a single subfeature, we will use that as our word feature
        if len(subfeatures) == 1:
            word_feature = getattr(self, subfeatures[0])
            # sum the values of our word feature over each of our words    
            word_sum = word_feature(word_index_range[0], word_index_range[1])
        else:
            word_sum = self.meta_feature(subfeatures, word_index_range[0], word_index_range[1])

        # return the average
        return float(word_sum) / max(1, word_index_range[1] - word_index_range[0]) 

    def char_to_word_average(self, subfeatures, word_spans_index_start, word_spans_index_end):
        '''
        takes in a char feature (or a char-level meta-feature) and word span range
        applies the char feature to each char in the specified words, and then averages the results
        over the number of chars.'''

        # get the start and end character indices from our word indices
        start = self.word_spans[word_spans_index_start][0]
        end = self.word_spans[word_spans_index_end - 1][1]

        # if we only have a single subfeature, we will use that as our char feature
        if len(subfeatures) == 1:
            char_feature = getattr(self, subfeatures[0])
            # sum the values of our char feature over each of our chars    
            char_sum = char_feature(start, end)
        else:
            char_sum = self.meta_feature(subfeatures, start, end)

        # return the average
        return float(char_sum) / max(1, end - start)

    def _init_feature_std(self, nested_feature_name):
        '''take in a nested std feature and build sum tables for querying std of that feature'''
        subfeatures = self._deconstruct_nested_feature_name(nested_feature_name)

        if len(subfeatures) == 2:
            # if we're just doing the std of a non-nested feature, we can use that features sum table
            # this would apply for something like "std(num_chars)"
            feature_name = subfeatures[1]
            if feature_name not in self.features:
                init_func = getattr(self, "_init_" + feature_name)
                init_func()

            # build the sum tables for std calculation
            sum_x = self.features[feature_name]
            x2 = [0] + [(sum_x[i+1] - sum_x[i]) ** 2 for i in range(len(sum_x) - 1)]
            sum_x2 = [sum(x2[:1+i]) for i in range(len(x2))]
    
            self.features[nested_feature_name] = [sum_x, sum_x2]
        else:
            # if we want the std of a nested feature, we need to figure that out
            raise NotImplementedError("You cannot take the std of a nested feature yet")
           
    def sentence_to_paragraph_std(self, subfeatures, para_spans_index_start, para_spans_index_end):
        '''
        Takes in a subfeature list and sentence spans and returns the standard deviation
        of the subfeatures taken on each queried paragraph 
        '''
        # get the start and end character indices from our sentence indices
        start = self.paragraph_spans[para_spans_index_start][0]
        end = self.paragraph_spans[para_spans_index_end - 1][1]
        
        nested_feature_name = self._construct_nested_feature_name(["std"] + subfeatures)

        # make sure the sum tables for the std feature is built
        if nested_feature_name not in self.features:
            self._init_feature_std(nested_feature_name)
        
        sum_x = self.features[nested_feature_name][0]
        sum_x2 = self.features[nested_feature_name][1]

        sentence_spans = spanutils.slice(self.sentence_spans, start, end, True)
        sentence_start, sentence_end = sentence_spans

        # math
        square_sum = sum_x2[sentence_end] - sum_x2[sentence_start]
        u = float(sum_x[sentence_end] - sum_x[sentence_start]) / (sentence_end - sentence_start)
    
        x = square_sum - 2 * u * (sum_x[sentence_end] - sum_x[sentence_start]) + (sentence_end - sentence_start) * u * u
        return math.sqrt(x / float(sentence_end - sentence_start))

    def word_to_sentence_std(self, subfeatures, sent_spans_index_start, sent_spans_index_end):
        '''
        Takes in a subfeature list and sentence spans and returns the standard deviation
        of the subfeatures taken on each queried sentence
        '''

        # get the start and end character indices from our sentence indices
        start = self.sentence_spans[sent_spans_index_start][0]
        end = self.sentence_spans[sent_spans_index_end - 1][1]
        
        nested_feature_name = self._construct_nested_feature_name(["std"] + subfeatures)

        # make sure the sum tables for the std feature is built
        if nested_feature_name not in self.features:
            self._init_feature_std(nested_feature_name)
        
        sum_x = self.features[nested_feature_name][0]
        sum_x2 = self.features[nested_feature_name][1]

        word_spans = spanutils.slice(self.word_spans, start, end, True)
        word_start, word_end = word_spans

        # math
        square_sum = sum_x2[word_end] - sum_x2[word_start]
        u = float(sum_x[word_end] - sum_x[word_start]) / (word_end - word_start)
    
        x = square_sum - 2 * u * (sum_x[word_end] - sum_x[word_start]) + (word_end - word_start) * u * u
        return math.sqrt(x / float(word_end - word_start))

    def char_to_word_std(self, subfeatures, word_spans_index_start, word_spans_index_end):
        '''
        Takes in a subfeature list and word spans and returns the standard deviation
        of the subfeatures taken on each queried word
        '''

        # get the start and end character indices from our word indices
        char_start = self.word_spans[word_spans_index_start][0]
        char_end = self.word_spans[word_spans_index_end - 1][1]
        
        nested_feature_name = self._construct_nested_feature_name(["std"] + subfeatures)

        # make sure the sum tables for the std feature is built
        if nested_feature_name not in self.features:
            self._init_feature_std(nested_feature_name)
        
        sum_x = self.features[nested_feature_name][0]
        sum_x2 = self.features[nested_feature_name][1]

        # math
        square_sum = sum_x2[char_end] - sum_x2[char_start]
        u = float(sum_x[char_end] - sum_x[char_start]) / (char_end - char_start)
    
        x = square_sum - 2 * u * (sum_x[char_end] - sum_x[char_start]) + (char_end - char_start) * u * u
        return math.sqrt(x / float(char_end - char_start))

    def _init_lancaster_stemmer(self):
        '''
        Initialize the sum table for yule's K characteristic
        '''
        from nltk.stem.lancaster import LancasterStemmer
        self.lancaster_stemmer = LancasterStemmer()

    def yule_k_characteristic(self, sent_spans_index_start, sent_spans_index_end):
        '''
        query the yule k characteristic
        '''
        # TODO figure out a way to make this constant time?
        # right now we are very slow

        if not self.lancaster_stemmer:
            self._init_lancaster_stemmer()
    
        spans = self.sentence_spans[sent_spans_index_start:sent_spans_index_end]
        start, end = spans[0][0], spans[-1][1]
        text = self.text[start:end]

        freqs = {}
        tokens = tokenization.tokenize(text, 'word', return_spans=False)
        for token in tokens:
            stem = self.lancaster_stemmer.stem(token.strip())
            freqs[stem] = freqs.get(stem, 0) + 1

        # Do Yule's calculations
        M1 = float(len(freqs))
        M2 = sum([len(list(g)) * (freq ** 2) for freq, g in groupby(sorted(freqs.values()))])

        return 10000 * (M2 - M1) / max(1, (M1 ** 2))

    def _init_evolved_feature_two(self):
        '''
        save the feature vectors for the features we care about for this feature
        '''

        self.features["evolved_feature_two"] = self.get_feature_vectors(["avg_internal_word_freq_class", "honore_r_measure", "syntactic_complexity", "pos_trigram,NN,NN,VB", "pos_trigram,NN,NN,NN", "pos_trigram,NN,IN,DT"], "paragraph")
        
    def evolved_feature_two(self, para_spans_index_start, para_spans_index_end):
        '''
        calculate out the evolved formula
        '''
        if "evolved_feature_two" not in self.features:
            self._init_evolved_feature_two()

        feature_vectors = self.features["evolved_feature_two"][para_spans_index_start:para_spans_index_end]

        D, H, L, P, X, Y = [sum([feature_tuple[i] for feature_tuple in feature_vectors]) for i in range(len(feature_vectors[0]))]
        computed = (((((1.5*D)-Y)-(-12.0**10.0))+(((L+10.0)+10.0)+10.0))*(10.0*(((1.5+P)**(-12.0-H))-((X*-12.0)+-1.0))))

        return computed

    def _init_evolved_feature_three(self):
        '''
        save the feature vectors for the features we care about for this feature
        F = evolved_feature_two; K = num_chars
        M = stopword_percentage; O = syntactic complexity
        '''

        self.features["evolved_feature_three"] = self.get_feature_vectors(["evolved_feature_two", "num_chars", "stopword_percentage", "syntactic_complexity"], "paragraph")
        
    def evolved_feature_three(self, para_spans_index_start, para_spans_index_end):
        '''
        calculate out the evolved formula
        '''
        if "evolved_feature_three" not in self.features:
            self._init_evolved_feature_three()

        feature_vectors = self.features["evolved_feature_three"][para_spans_index_start:para_spans_index_end]

        F, K, M, O = [sum([feature_tuple[i] for feature_tuple in feature_vectors]) for i in range(len(feature_vectors[0]))]
        computed = F - 2 * K - M + O

        return computed

    def _init_evolved_feature_four(self):
        '''
        save the feature vectors for the features we care about for this feature
        '''

        self.features["evolved_feature_four"] = self.get_feature_vectors(["evolved_feature_two", "flesch_kincaid_grade", "flesch_reading_ease", "stopword_percentage", "pos_trigram,DT,NNS,IN"], "paragraph")
        
    def evolved_feature_four(self, para_spans_index_start, para_spans_index_end):
        '''
        calculate out the evolved formula
        '''
        if "evolved_feature_four" not in self.features:
            self._init_evolved_feature_four()

        feature_vectors = self.features["evolved_feature_four"][para_spans_index_start:para_spans_index_end]

        F, G, H, M, W = [sum([feature_tuple[i] for feature_tuple in feature_vectors]) for i in range(len(feature_vectors[0]))]
        computed = F + G + 6 * H + M - W

        return computed

    def _init_evolved_feature_five(self):
        '''
        save the feature vectors for the features we care about for this feature
        '''

        self.features["evolved_feature_five"] = self.get_feature_vectors(["average_syllables_per_word", "honore_r_measure"], "paragraph")
        
    def evolved_feature_five(self, para_spans_index_start, para_spans_index_end):
        '''
        calculate out the evolved formula
        '''
        if "evolved_feature_five" not in self.features:
            self._init_evolved_feature_five()

        feature_vectors = self.features["evolved_feature_five"][para_spans_index_start:para_spans_index_end]

        B, J = [sum([feature_tuple[i] for feature_tuple in feature_vectors]) for i in range(len(feature_vectors[0]))]

        try:
            if J > 1.5 and B != int(B):
                # this will cause us to root a negative number
                computed = 5 * 4 ** ( (1.5 - J)**int(B) - B - J - 3)
            else:
                computed = 5 * 4 ** ( (1.5 - J)**B - B - J - 3)
        except OverflowError:
            # on overflow, return max value
            return 1.0e+255

        return computed

    def _init_evolved_feature_six(self):
        '''
        save the feature vectors for the features we care about for this feature
        '''

        self.features["evolved_feature_six"] = self.get_feature_vectors(["evolved_feature_two", "flesch_reading_ease", "pos_trigram,VB,NN,VB", "pos_trigram,DT,NN,IN", "word_unigram,been"], "paragraph")
        
    def evolved_feature_six(self, para_spans_index_start, para_spans_index_end):
        '''
        calculate out the evolved formula
        '''
        if "evolved_feature_six" not in self.features:
            self._init_evolved_feature_six()

        feature_vectors = self.features["evolved_feature_six"][para_spans_index_start:para_spans_index_end]

        F, H, X, Y, d = [sum([feature_tuple[i] for feature_tuple in feature_vectors]) for i in range(len(feature_vectors[0]))]
        computed = 20 * d * H * 6.5 ** (X ** Y ) + F

        return computed

    def _init_num_complex_words(self):
        '''
        init sum table of number of complex words (3+ syllables)
        '''

        sum_table = [0]
        for start, end in self.word_spans:
            num = sum_table[-1]
            num_syl = self._syl(start, end)
            if num_syl > 2:
                num += 1
            sum_table.append(num)

        self.features["num_complex_words"] = sum_table
    
    def gunning_fog_index(self, sent_spans_index_start, sent_spans_index_end):
        '''
        A measure of sentence complexity
        definition off http://en.wikipedia.org/wiki/Gunning_fog_index
        '''

        average_sentence_length = self.average_sentence_length(sent_spans_index_start, sent_spans_index_end)
 
        spans = self.sentence_spans[sent_spans_index_start:sent_spans_index_end]
        start, end = spans[0][0], spans[-1][1]
        word_start, word_end = spanutils.slice(self.word_spans, start, end, True)

        if "num_complex_words" not in self.features:
            self._init_num_complex_words()
        complex_words_table = self.features["num_complex_words"]
        percent_complex_words = (complex_words_table[word_end] - complex_words_table[word_start]) / float(max(1, word_end - word_start))

        return .4 * (average_sentence_length + 100 * percent_complex_words)

    def honore_r_measure(self, sent_spans_index_start, sent_spans_index_end):
        '''
        R = 100 logN / (1 - V_1 / V)
        where V_1 = # words appearing only once, V = total vocab size,
        N = number of words
        '''
        if not self.lancaster_stemmer:
            self._init_lancaster_stemmer()

        spans = self.sentence_spans[sent_spans_index_start:sent_spans_index_end]
        start, end = spans[0][0], spans[-1][1]
        word_start, word_end = spanutils.slice(self.word_spans, start, end, True)

        words = {}
        for start, end in self.word_spans[word_start:word_end]:
            word = self.lancaster_stemmer.stem(self.text[start:end])
            if word in words:
                words[word] = 0
            else:
                words[word] = 1

        V = float(len(words)) + 1
        V1 = float(sum(words.values()))
        N = word_end - word_start

        return 100 * math.log(N) / (1 - (V1 / V))

    def _init_average_syllables_per_word(self):
        '''
        Initializes the average syllables sum table. sum_table[i] is the sum of the number
        of syllables of words from 0 to i-1.
        '''
        sum_table = [0]
        for start, end in self.word_spans:
            sum_table.append(sum_table[-1]+self._syl(start, end))
        self.features["average_syllables_per_word"] = sum_table
    
    def average_syllables_per_word(self, word_spans_index_start, word_spans_index_end):
        '''
        Return the average number of syllables per word.
        '''
        if "average_syllables_per_word" not in self.features:
            self._init_average_syllables_per_word()
        
        sum_table = self.features["average_syllables_per_word"]
        total_syllables = sum_table[word_spans_index_end] - sum_table[word_spans_index_start]
        num_words = word_spans_index_end - word_spans_index_start
        return float(total_syllables) / max(num_words, 1)
    
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

        self.features["average_sentence_length"] = sum_table
    
    def average_sentence_length(self, sent_spans_index_start, sent_spans_index_end):
        '''
        Return the average number of words in sentences [sent_spans_index_start : sent_spans_index_end]
        '''
        if "average_sentence_length" not in self.features:
            self._init_average_sentence_length()
        
        sum_table = self.features["average_sentence_length"]
        total_sentence_length = sum_table[sent_spans_index_end] - sum_table[sent_spans_index_start]
        num_words = sent_spans_index_end - sent_spans_index_start
        return float(total_sentence_length) / max(num_words, 1)
    
    def _init_pos_frequency_table(self):
        '''
        instantiates a dictionary of part-of-speech counts over word indices 
        tracks all tags from the penn treebank tagger as well as some additional features
        keys to the dictionary are parts of speech like "VERBS" or specific tags like "VBZ"
        '''

        # make sure that we have done PoS tagging
        if not self.pos_tagged:
            self._init_tag_list(self.text)

        sum_dict = {}
        # temp_dict will hold temporary values that we'll later use to construct the sum table for each key
        temp_dict = {}
        for index in range(len(self.pos_tags)):
            word = self.pos_tags[index][0].lower()
            tag = self.pos_tags[index][1]
            #print "looking at", word, "which is a", tag
            temp_dict[tag] = temp_dict.get(tag, []) + [index]

            # Now we want to populate special keys that we care about:
            # verbs
            if tag.startswith("VB"):
                temp_dict["VERBS"] = temp_dict.get("VERBS", []) + [index]
            # wh-words (why, who, what, where...)
            if tag in ["WP", "WP$", "WDT", "WRB"]:
                temp_dict["WH"] = temp_dict.get("WH", []) + [index]
            # subordinating conjunctions
            if word in ["since", "because", "as", "that", "while", "unless", "until", "than", "though", "although"]:
                temp_dict["SUB"] = temp_dict.get("SUB", []) + [index]

        # now go through the stored indices to create the actual sum table
        for key in temp_dict:
            sum_dict[key] = [0]
            prev_index = 0
            prev_value = 0
            for index in temp_dict[key]:
                sum_dict[key] += [prev_value] * (index - prev_index)
                prev_index = index
                prev_value += 1
            sum_dict[key] += [prev_value] * (len(self.pos_tags) - prev_index)

        # so each tag and special key holds a list that looks like [0,0,0,1,1,1,2,2,3,4, ... ]
        # where list[i] is the number of tags for that key that have happened _before_ the ith word 
        # note that this is exclusive of the final index, so sum_dict[tag][0] = 0 for all tags
        # so sum_dict["VERBS"][10] is the number of verbs that have shown up in words 0 through 9.

        self.pos_frequency_count_table = sum_dict
        
        self.pos_frequency_count_table_initialized = True
    
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
        self.features["stopword_percentage"] = sum_table
        
    def stopword_percentage(self, word_spans_index_start, word_spans_index_end):
        '''
        Return the percentage of words that are stop words in the text between the two given indices.
        '''
        if "stopword_percentage" not in self.features:
            self._init_stopword_percentage()
        
        sum_table = self.features["stopword_percentage"]
        total_stopwords = sum_table[word_spans_index_end] - sum_table[word_spans_index_start]
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

        self.features["punctuation_percentage"] = sum_table
    
    def punctuation_percentage(self, char_index_start, char_index_end):
        if "punctuation_percentage" not in self.features:
            self._init_punctuation_percentage()
        
        sum_table = self.features["punctuation_percentage"]
        total_punctuation = sum_table[char_index_end] - sum_table[char_index_start]
        num_chars = char_index_end - char_index_start
        return float(total_punctuation) / max(num_chars, 1)

    def _init_syntactic_complexity(self):
        sum_table = [0]
        for start, end in self.word_spans:
            word_spans = spanutils.slice(self.word_spans, start, end, True)
            try:
                conjunctions_table = self.pos_frequency_count_table.get("SUB", None)
                if conjunctions_table != None:
                    #print "conjunctions", conjunctions_table[word_spans[0]:word_spans[1]+1]
                    num_conjunctions = conjunctions_table[word_spans[1]] - conjunctions_table[word_spans[0]]
                else:
                    num_conjunctions = 0

                wh_table = self.pos_frequency_count_table.get("WH", None)
                if wh_table != None:
                    #print "wh", wh_table[word_spans[0]:word_spans[1]+1]
                    num_wh_pronouns = wh_table[word_spans[1]] - wh_table[word_spans[0]]
                else:
                    num_wh_pronouns = 0

                verb_table = self.pos_frequency_count_table.get("VERBS", None)
                if verb_table != None:
                    #print "verbs", verb_table[word_spans[0]:word_spans[1]+1]
                    num_verb_forms = verb_table[word_spans[1]] - verb_table[word_spans[0]]
                else:
                    num_verb_forms = 0

            except IndexError:
                with open("SyntacticComplexityError.txt", "w") as error_file:
                    error_file.write(str(self.text[self.word_spans[word_spans_index_start][0]:]))
                    error_file.write("-------")
                    error_file.write(self.word_spans[word_spans_index_start:])
                    error_file.write("-------")
                    error_file.write(self.pos_frequency_count_table)
                raise IndexError("There appears to be an indexing problem with POS tags! Alert Zach!")
        
            complexity = 2 * num_conjunctions + 2 * num_wh_pronouns + num_verb_forms
            sum_table.append(complexity + sum_table[-1])

        self.features["syntactic_complexity"] = sum_table

    def syntactic_complexity(self, word_spans_index_start, word_spans_index_end):
        '''
        This feature is a modified version of the "Index of Syntactic Complexity" taken from
        Szmrecsanyi, Benedikt. "On operationalizing syntactic complexity." Jadt-04 2 (2004): 1032-1039. 
        found at http://www.benszm.net/omnibuslit/Szmrecsanyi2004.pdf
        it tallies various part-of-speech counts which are approximated by NLTK's POS tags.
        the original version accounts for Noun Phrases, which are not counted in the NLTK tagger, and so are ignored.
        '''
        # Note that this feature uses the same initialization that pos_percentage_vector does.
        if not self.pos_frequency_count_table_initialized:
            self._init_pos_frequency_table()
        if "syntactic_complexity" not in self.features:
            self._init_syntactic_complexity()

        #print "querying syntactic complexity of", word_spans_index_start, "to", word_spans_index_end    

        sum_table = self.features["syntactic_complexity"]
        return sum_table[word_spans_index_end] - sum_table[word_spans_index_start]

    def _init_syntactic_complexity_average(self):
        '''
        Initializes the syntactic_complexity_sum_table. syntactic_complexity_sum_table[i] is the sum of the sum
        of syntactic complexities in sentences 0 to i-1.
        '''
        sum_table = [0]
        for start, end in self.sentence_spans:
            # word_spans holds the (start, end) tuple of indices for the words in the currently examined sentence
            word_spans = spanutils.slice(self.word_spans, start, end, True)
            complexity = self.syntactic_complexity(word_spans[0], word_spans[1])
            sum_table.append(complexity + sum_table[-1])

        self.features["syntactic_complexity_average"] = sum_table

    def syntactic_complexity_average(self, sent_spans_index_start, sent_spans_index_end):
        '''
        Computes the average syntactic complexity for each sentence in the given paragraphs and averages them
        '''

        if "syntactic_complexity_average" not in self.features:
            self._init_syntactic_complexity_average() 
        
        sum_table = self.features["syntactic_complexity_average"]
        end_sum = sum_table[sent_spans_index_end]
        start_sum = sum_table[sent_spans_index_start]
        total_syntactic_complexity = end_sum - start_sum
        num_sents = sent_spans_index_end - sent_spans_index_start
        return float(total_syntactic_complexity) / max(num_sents, 1)
    
    def _init_avg_internal_word_freq_class(self):
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

        self.features["avg_internal_word_freq_class"] = table
    
    def avg_internal_word_freq_class(self, word_spans_index_start, word_spans_index_end):
        '''
        Returns a feature like avg_external_word_freq_class except that the frequenecy
        classes of words are calculated based on the occurrences of words within this
        text, not the brown corpus.
        '''
        if "avg_internal_word_freq_class" not in self.features:
            self._init_avg_internal_word_freq_class()

        sum_table = self.features["avg_internal_word_freq_class"]
        total = sum_table[word_spans_index_end] - sum_table[word_spans_index_start]
        return total / float(max(1, word_spans_index_end - word_spans_index_start))
        
    def _init_avg_external_word_freq_class(self):

        #word_freq_dict = {}
        #for word in nltk.corpus.brown.words():
        #    word = word.lower()
        #    word_freq_dict[word] = word_freq_dict.get(word, 0) + 1
        word_freq_dict = cPickle.load(open(os.path.join(os.path.dirname(__file__), "word_freq.pkl",), "rb"))
        
        corpus_word_freq_by_rank = sorted(word_freq_dict.items(), key=lambda x: x[1], reverse=True)
        occurences_of_most_freq_word = corpus_word_freq_by_rank[0][1]
    
        class_sum_table = [0]
        
        for span in self.word_spans:
            word = self.text[span[0]:span[1]].lower().translate(string.maketrans("",""), string.punctuation)
            # plus one smoothing is used here!
            freq_class = math.floor(math.log(float(occurences_of_most_freq_word + 1) / (word_freq_dict.get(word.lower(), 0) + 1),2))
            class_sum_table.append(class_sum_table[-1] + freq_class)
        
        self.features["avg_external_word_freq_class"] = class_sum_table
        
    def avg_external_word_freq_class(self, word_spans_index_start, word_spans_index_end):
        '''
        This feature is defined here:
        http://www.uni-weimar.de/medien/webis/publications/papers/stein_2006d.pdf
        
        Plus one smoothing is used.
        '''
        if "avg_external_word_freq_class" not in self.features:
            self._init_avg_external_word_freq_class()

        sum_table = self.features["avg_external_word_freq_class"]
        total = sum_table[word_spans_index_end] - sum_table[word_spans_index_start]
        num_words = word_spans_index_end - word_spans_index_start
        return float(total) / max(num_words, 1)
    
    def _init_syl(self):
    
        self._cmud = cmudict.dict()
    
    def _syl(self, start, end):
        '''
        Returns the number of syllables in the word designated by the given start and end
        word indices.
        '''
        # NOTE This is not a feature, but this function is used by other features.
        
        # The following link explains syllabifaction
        # https://groups.google.com/forum/#!topic/nltk-users/mCOh_u7V8_I
        # The syllabification code here is taken from that discussion thread.
        
        if self._cmud == None:
            self._init_syl()
        
        word = self.text[start:end].strip(".").lower()
        try:
            syls = [len(list(y for y in x if y[-1].isdigit())) for x in self._cmud[word]][0]
        except KeyError:
            syls = (end-start)%3
        return syls
        
    def flesch_reading_ease(self, para_spans_index_start, para_spans_index_end):
        '''
        This is a paragraph level feature that returns the Flesch readability score for the given
        paragraph.
        '''
        # The following link explains the feature
        # http://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests
        
        # get the span for the paragraph in question
        para_span = self.paragraph_spans[para_spans_index_start:para_spans_index_end][0]
        # get number of sentences in the paragraph
        num_sentences = len(spanutils.slice(self.sentence_spans, para_span[0], para_span[1]))
        word_spans = spanutils.slice(self.word_spans, para_span[0], para_span[1])
        num_words = len(word_spans)
        
        num_syl = 0  # total number of sylables in the paragraph                                                                          
        for start, end in word_spans:
            num_syl += self._syl(start, end)
        try:
            return 206.835 - 1.015*(num_words / float(num_sentences)) - 84.6*(num_syl / float(num_words))
        except ZeroDivisionError:
            # Either num of word or num of sentences was zero, so return 100, an "easy" score (according to wikipedia)
            return 100
            
    def flesch_kincaid_grade(self, para_spans_index_start, para_spans_index_end):
        '''
        This is a paragraph level feature that returns the Flesh-Kincaid grade level for
        the given paragraph.
        '''
        # get the span for the paragraph in question
        para_span = self.paragraph_spans[para_spans_index_start:para_spans_index_end][0]
        # get number of sentences in the paragraph
        num_sentences = len(spanutils.slice(self.sentence_spans, para_span[0], para_span[1]))
        word_spans = spanutils.slice(self.word_spans, para_span[0], para_span[1])
        num_words = len(word_spans)
        
        num_syl = 0  # total number of sylables in the paragraph                                                                          
        for start, end in word_spans:
            num_syl += self._syl(start, end)
        try:
            return 0.39*(num_words / float(num_sentences)) + 11.8*(num_syl / float(num_words))-15.59
        except ZeroDivisionError:
            # Either num of word or num of sentences was zero, so return 100, an "easy" score (according to wikipedia)
            return 0
    
    def word_unigram(self, word_spans_index_start, word_spans_index_end, target_word):
        '''
        This is a word level feature that returns the number of occurences of the word <target_word>
        for the given words.
        '''
        total = 0
        for i in range(word_spans_index_start, word_spans_index_end):
            w_start, w_end = self.word_spans[i]
            word = self.text[w_start:w_end]
            if word.strip(".").lower() == target_word:
                total += 1
        return total
    
    def pos_trigram(self, word_spans_index_start, word_spans_index_end, pos):
        '''
        pos is a tuple of parts-of-speech e.g. ("NN", "NN", "NN")
        '''
        #  http://www.aaai.org/Papers/Workshops/1998/WS-98-05/WS98-05-001.pdf
       
        # make sure that we have done PoS tagging
        if not self.pos_tagged:
            self._init_tag_list(self.text)
                
        total = 0
        for i in range(word_spans_index_start, word_spans_index_end-2):
            tag1, tag2, tag3, = self.pos_tags[i][1], self.pos_tags[i+1][1], self.pos_tags[i+2][1]
            if (tag1, tag2, tag3) == pos:
                total += 1
        return total
        
    def vowelness_trigram(self, char_index_start, char_index_end, tri):
        '''
        This feature returns the number of occurences of the given "vowelness" trigram. That is,
        the items of the trigram are either "V" for vowel, "C" for consonant, or "X" for other.
        
        tri is a tuple of character types (consonant or vowel), e.g. ("V", "C", "V")
        '''
        def vowel_or_consonant(char):
            if char in "aeiou":
                return "V"
            elif char in "qwrtypsdfghjklzxcvbnm":
                return "C"
            else:
                return "X"

        total = 0
        for char_i in range(char_index_start, char_index_end-2):
            c1, c2, c3 = self.text[char_i].lower(), self.text[char_i+1].lower(), self.text[char_i+2].lower()
            if vowel_or_consonant(c1) == tri[0] and vowel_or_consonant(c2) == tri[1] and vowel_or_consonant(c3) == tri[2]:
                total += 1
        return total
            

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
    
    #print f.get_feature_vectors(["num_chars"], "sentence")
    if f.get_feature_vectors(["num_chars"], "sentence") == [(15,), (10,), (15,)]:
        print "num_chars test passed"
    else:
        print "num_chars test FAILED"

    #print f.get_feature_vectors(["avg(num_chars)"], "sentence")
    if f.get_feature_vectors(["avg(num_chars)"], "sentence") == [(3.75,), (5.0,), (3.0,)]:
        print "avg(num_chars) test passed"
    else:
        print "avg(num_chars) test FAILED"

    # TODO: ADD TEST FOR INTERNAL WORD FREQ CLASS
    f.get_feature_vectors(["avg_internal_word_freq_class"], "sentence")
    print "avg_internal_word_freq_class test DOES NOT EXIST" 
  
    
    # TODO: ADD TEST FOR EXTERNAL WORD FREQ CLASS
    f.get_feature_vectors(["avg_external_word_freq_class"], "sentence")
    print "avg_external_word_freq_class test DOES NOT EXIST"
    
    f = FeatureExtractor("The brown fox ate. I go to the school? Believe it. I go.")
    #print f.get_feature_vectors(["syntactic_complexity"], "sentence")
    if f.get_feature_vectors(["syntactic_complexity"], "sentence") == [(0,), (1,), (0,), (1,)]:
        print "syntactic_complexity test passed"
    else:
        print "syntactic_complexity test FAILED"

    f = FeatureExtractor("I absolutely go incredibly far. Zach went fast over crab sand land.\n\nThis is a new paragraph. This is the second sentence in that paragraph. This exsquisite utterance is indubitably the third sentence of this fine text.\n\nPlagiarism detection can be operationalized by decomposing a document into natural sections, such as sentences, chapters, or topically related blocks, and analyzing the variance of stylometric features for these sections. In this regard the decision problems in Sect. 1.2 are of decreasing complexity: instances of AVFIND are comprised of both a selection problem (finding suspicious sections) and an AVOUTLIER problem; instances of AVBATCH are a restricted variant of AVOUTLIER since one has the additional knowledge that all elements of a batch are (or are not) outliers at the same time.")
    f.get_feature_vectors(["flesch_reading_ease"], "paragraph")
    print "flesch_reading_ease test DOES NOT EXIST"
    
    f = FeatureExtractor("I absolutely go incredibly far. Zach went fast over crab sand land.\n\nThis is a new paragraph. This is the second sentence in that paragraph. This exsquisite utterance is indubitably the third sentence of this fine text.\n\nPlagiarism detection can be operationalized by decomposing a document into natural sections, such as sentences, chapters, or topically related blocks, and analyzing the variance of stylometric features for these sections. In this regard the decision problems in Sect. 1.2 are of decreasing complexity: instances of AVFIND are comprised of both a selection problem (finding suspicious sections) and an AVOUTLIER problem; instances of AVBATCH are a restricted variant of AVOUTLIER since one has the additional knowledge that all elements of a batch are (or are not) outliers at the same time.")
    f.get_feature_vectors(["flesch_kincaid_grade"], "paragraph")
    print "flesch_kincaid_grade test DOES NOT EXIST"
    
    f = FeatureExtractor("The brown fox ate. I go to the school? Believe it. This is reprehensible.")
    #print f.get_feature_vectors(["average_syllables_per_word"], "sentence")
    if f.get_feature_vectors(["average_syllables_per_word"], "sentence") == [(1/1.0,), (1/1.0,), (3/2.0,), (7/3.0,)]:
        print "average_syllables_per_word test passed"
    else:
        print "average_syllables_per_word test FAILED"
    
    f = FeatureExtractor("The brown fox ate. I go to the school? Believe it. This is reprehensible. How have you been?")
    #print f.get_feature_vectors(["word_unigram,the"], "sentence")
    if f.get_feature_vectors(["word_unigram,the"], "sentence") == [(1,), (1,), (0,), (0,), (0,)]:
        print "word_unigram,the test passed"
    else:
        print "word_unigram,the test FAILED"
          
    f = FeatureExtractor("The mad hatter likes tea and the red queen hates alice.")
    #print f.get_feature_vectors(["pos_trigram,NN,NN,NN"], "sentence")
    if f.get_feature_vectors(["pos_trigram,NN,NN,NN"], "sentence") == [(1,)]:
        print "pos_trigram,NN,NN,NN test passed"
    else:
        print "pos_trigram,NN,NN,NN test FAILED"
    
    f = FeatureExtractor("The mad hatter likes tea and the red queen hates alice.")
    print f.get_feature_vectors(["vowelness_trigram,C,V,C"], "sentence")
    if f.get_feature_vectors(["vowelness_trigram,C,V,C"], "sentence") == [(9,)]:
        print "vowelness_trigram,C,V,C test passed"
    else:
        print "vowelness_trigram,C,V,C test FAILED"

    f = FeatureExtractor("The mad hatter likes tea and the red queen hates alice. Images of the Mandelbrot set display an elaborate boundary that reveals progressively ever-finer recursive detail at increasing magnifications. The style of this repeating detail depends on the region of the set being examined. The set's boundary also incorporates smaller versions of the main shape, so the fractal property of self-similarity applies to the entire set, and not just to its parts.")
    #print "yule_k_characteristic", f.get_feature_vectors(["yule_k_characteristic"], "paragraph")
    vectors =  f.get_feature_vectors(["yule_k_characteristic"], "paragraph")
    if [round(x[0], 4) for x in vectors] == [555.3578]:
        print "yule_k_characteristic test passed"
    else:
        print "yule_k_characteristic test FAILED"

    f = FeatureExtractor("The mad hatter likes tea and the red queen hates alice. Images of the Mandelbrot set display an elaborate boundary that reveals progressively ever-finer recursive detail at increasing magnifications. The style of this repeating detail depends on the region of the set being examined. The set's boundary also incorporates smaller versions of the main shape, so the fractal property of self-similarity applies to the entire set, and not just to its parts.")
    #print f.get_feature_vectors(["gunning_fog_index"], "sentence")
    vectors =  f.get_feature_vectors(["gunning_fog_index"], "sentence")
    if [round(x[0], 4) for x in vectors] == [4.4, 20.5333, 11.3333, 17.5613]:
        print "gunning_fog_index test passed"
    else:
        print "gunning_fog_index test FAILED"

    f = FeatureExtractor("The mad hatter likes tea and the red queen hates alice. Images of the Mandelbrot set display an elaborate boundary that reveals progressively ever-finer recursive detail at increasing magnifications. The style of this repeating detail depends on the region of the set being examined. The set's boundary also incorporates smaller versions of the main shape, so the fractal property of self-similarity applies to the entire set, and not just to its parts.")
    #print "honore_r_measure", f.get_feature_vectors(["honore_r_measure"], "paragraph")
    vectors =  f.get_feature_vectors(["honore_r_measure"], "paragraph")
    if [round(x[0], 4) for x in vectors] == [2331.4436]:
        print "honore_r_measure test passed"
    else:
        print "honore_r_measure test FAILED"
    
if __name__ == "__main__":
    _test()

