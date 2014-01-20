from .. import tokenization
from .. import spanutils
from ..shared.passage import IntrinsicPassage

import nltk
import inspect
import string
import math
import cPickle
import os.path

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

        for func_name, func in all_methods:
            func_args = set(inspect.getargspec(func).args)

            valid_func = len(feature_arg_options.intersection(func_args)) > 0

            # Has some overlap -- may be a nested function
            if include_nested and valid_func:
                feature_function_names.append(func_name)

            # Has some overlap, but we want to ignore nested functions
            # (which include 'subfeatures' as an argument)
            if not include_nested and valid_func and \
                    'subfeatures' not in func_args:
                feature_function_names.append(func_name)

        return feature_function_names

    
    def __init__(self, text):
        self.text = text
        self.word_spans = tokenization.tokenize(text, "word")
        self.sentence_spans = tokenization.tokenize(text, "sentence")
        self.paragraph_spans = tokenization.tokenize(text, "paragraph")
    
        self.features = {}

        self._cmud = None # initialized by self._syl()
        self.pos_tags = None
        self.pos_tagged = False

        #TODO evaluate whether we want to keep this?
        self.pos_frequency_count_table_initialized = False
 
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
        #print "querying syntactic complexity of", word_spans_index_start, "to", word_spans_index_end    
        conjunctions_table = self.pos_frequency_count_table.get("SUB", None)
        if conjunctions_table != None:
            #print "conjunctions", conjunctions_table[word_spans_index_start:word_spans_index_end+1]
            num_conjunctions = conjunctions_table[word_spans_index_end] - conjunctions_table[word_spans_index_start]
        else:
            num_conjunctions = 0

        wh_table = self.pos_frequency_count_table.get("WH", None)
        if wh_table != None:
            #print "wh", wh_table[word_spans_index_start:word_spans_index_end+1]
            num_wh_pronouns = wh_table[word_spans_index_end] - wh_table[word_spans_index_start]
        else:
            num_wh_pronouns = 0

        verb_table = self.pos_frequency_count_table.get("VERBS", None)
        if verb_table != None:
            #print "verbs", verb_table[word_spans_index_start:word_spans_index_end+1]
            num_verb_forms = verb_table[word_spans_index_end] - verb_table[word_spans_index_start]
        else:
            num_verb_forms = 0
        
        return 2 * num_conjunctions + 2 * num_wh_pronouns + num_verb_forms

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

        self.features["avg_internal_word_freq_class"] = table
    
    def avg_internal_word_freq_class(self, word_spans_index_start, word_spans_index_end):
        '''
        Returns a feature like avg_external_word_freq_class except that the frequenecy
        classes of words are calculated based on the occurrences of words within this
        text, not the brown corpus.
        '''
        if "avg_internal_word_freq_class" not in self.features:
            self._init_internal_word_freq_class()

        sum_table = self.features["avg_internal_word_freq_class"]
        total = sum_table[word_spans_index_end] - sum_table[word_spans_index_start]
        return total / float(max(1, word_spans_index_end - word_spans_index_start))
        
    def _init_external_word_freq_class(self):

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
            self._init_external_word_freq_class()
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
    
    def frequency_of_word_of(self, word_spans_index_start, word_spans_index_end):
        '''
        This is a word level feature that returns the number of occurences of the word "of" for the
        given words.
        '''
        return self._frequency_of_word(word_spans_index_start, word_spans_index_end, "of")
        
    def frequency_of_word_is(self, word_spans_index_start, word_spans_index_end):
        '''
        This is a word level feature that returns the number of occurences of the word "is" for the
        given words.
        '''
        return self._frequency_of_word(word_spans_index_start, word_spans_index_end, "is")
        
    def frequency_of_word_the(self, word_spans_index_start, word_spans_index_end):
        '''
        This is a word level feature that returns the number of occurences of the word "the" for the
        given words.
        '''
        return self._frequency_of_word(word_spans_index_start, word_spans_index_end, "the")
        
    def frequency_of_word_been(self, word_spans_index_start, word_spans_index_end):
        '''
        This is a word level feature that returns the number of occurences of the word "been" for the
        given words.
        '''
        return self._frequency_of_word(word_spans_index_start, word_spans_index_end, "been")

    def _frequency_of_word(self, start, end, target_word):
        '''
        This helper function return the number of occurence of target_word in in the section of text
        deliminated by the start and end self.word_spans indices.
        '''
        total = 0
        for i in range(start, end):
            w_start, w_end = self.word_spans[i]
            word = self.text[w_start:w_end]
            if word.strip(".").lower() == target_word:
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
    
    f = FeatureExtractor("The brown fox ate. I go to the school? Believe it. Of mice and men. How have you been?")
    #print f.get_feature_vectors(["average_syllables_per_word"], "sentence")
    if f.get_feature_vectors(["frequency_of_word_of"], "sentence") == [(0,), (0,), (0,), (1,), (0,)]:
        print "frequency_of_word_of test passed"
    else:
        print "frequency_of_word_of test FAILED"
        
    f = FeatureExtractor("The brown fox ate. I go to the school? Believe it. This is reprehensible. How have you been?")
    #print f.get_feature_vectors(["average_syllables_per_word"], "sentence")
    if f.get_feature_vectors(["frequency_of_word_been"], "sentence") == [(0,), (0,), (0,), (0,), (1,)]:
        print "frequency_of_word_been test passed"
    else:
        print "frequency_of_word_been test FAILED"
    
    f = FeatureExtractor("The brown fox ate. I go to the school? Believe it. This is reprehensible. How have you been?")
    #print f.get_feature_vectors(["average_syllables_per_word"], "sentence")
    if f.get_feature_vectors(["frequency_of_word_the"], "sentence") == [(1,), (1,), (0,), (0,), (0,)]:
        print "frequency_of_word_the test passed"
    else:
        print "frequency_of_word_the test FAILED"
    
    f = FeatureExtractor("The brown fox ate. I go to the school? Believe it. This is reprehensible. How have you been?")
    #print f.get_feature_vectors(["average_syllables_per_word"], "sentence")
    if f.get_feature_vectors(["frequency_of_word_is"], "sentence") == [(0,), (0,), (0,), (1,), (0,)]:
        print "frequency_of_word_is test passed"
    else:
        print "frequency_of_word_is test FAILED"
    
    
    
    #f = FeatureExtractor("The brown fox ate. I go to the school. Believe it. I go.")
    ##print f.get_feature_vectors(["syntactic_complexity_average"], "paragraph")
    #if f.get_feature_vectors(["syntactic_complexity_average"], "paragraph") == [(0.5,)]:
    #    print "syntactic_complexity_average test passed"
    #else:
    #    print "syntactic_complexity_average test FAILED"
    
if __name__ == "__main__":
    _test()

    #f = FeatureExtractor("I absolutely go incredibly far. Zach went fast over sand crab land.")
    #print f.get_feature_vectors(["num_chars", "avg(num_chars)", "avg(avg(num_chars))", "std(num_chars)", "avg(std(num_chars))", "std(std(num_chars))"], "paragraph")
    ##print f.get_feature_vectors(["num_chars", "avg(num_chars)", "avg(avg(num_chars))", "std(num_chars)", "avg(std(num_chars))", "std(std(num_chars))"], "paragraph")
    #print f.get_feature_vectors(["punctuation_percentage", "avg(punctuation_percentage)", "std(punctuation_percentage)"], "paragraph")

