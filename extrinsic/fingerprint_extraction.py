# fingerprint_extraction.py
# Module for generating fingerprints from documents.

import nltk
import itertools
import os
import string, random, re, operator
from .. import tokenization
from ..shared.util import ExtrinsicUtility

import psycopg2
from ..dbconstants import username, password, dbname
import fingerprintstorage

# TODO: omit words tokenized by nltk that are just puncuation


class Passage:
    def __init__(self, id, doc_name, doc_path, doc_xml_path, method, n, k, hash_len, atom_type, atom_number, is_source, hash_values):
        self.id = id
        self.doc_name = doc_name
        self._doc_path = doc_path
        self._doc_xml_path = doc_xml_path
        self.method = method
        self.n = n
        self.k = k
        self.hash_len = hash_len
        self.atom_type = atom_type
        self.atom_number = atom_number
        self.is_source = is_source
        self.hash_values = hash_values

class FingerprintExtractor:

    def __init__(self, hash_span=10000):
        self.anchors = ['ul', 'ay', 'oo', 'yo', 'si', 'ca', 'am', 'ie', 'mo', 'rt']
        self.hash_span = hash_span

    def _gen_string_hash(self, in_string):
        '''
        Converts the given string <in_string> to an integer which is
        roughly uniformly distributed over all possible values. The integer is
        between 0 and self.hash_span.

        This method is used in "Methods for Identifying Versioned and 
        Plagiarized Documents".
        '''
        if len(in_string) == 0:
            return 0
        elif len(in_string) == 1:
            return ord(in_string)

        hash_list = [ord(in_string[0])] # TODO: what should the 0th element be?
        for i in xrange(1, len(in_string)):
            cur_char = in_string[i]
            res = hash_list[i-1]^(ord(cur_char) + (hash_list[i-1]<<6) + (hash_list[i-1]>>2))
            hash_list.append(res)
        return hash_list[-1] % self.hash_span

    def get_fingerprint(self, document, n, method, k):
        '''
        Returns a fingerprint, or list of minutiae, for the given document.

        There are several key pieces involved in fingerprinting:
        1: Converting n-grams to integers using a hash function
        2: Granularity, or size of the n-grams (more simply, the size of n)
        3: Resolution, or the number of minutiae used to represent the fingerprint
        4: Selection Strategy, which is the method used to select n-grams from the document
        '''
        if method == "full":
            words = nltk.tokenize.punkt.PunktWordTokenizer().tokenize(document)
            # Strips all punctuation from <words>
            words_stripped = tokenization.strip_punctuation(words)
            fingerprint = self._get_full_fingerprint(words_stripped, n)

        elif method == "kth_in_sent":
            sentences = nltk.tokenize.punkt.PunktSentenceTokenizer().tokenize(document)
            fingerprint = self._get_kth_in_sent_fingerprint(sentences, n, k)

        elif method == "anchor":
            words = nltk.tokenize.punkt.PunktWordTokenizer().tokenize(document)
            # Strips all punctuation from <words>
            words_stripped = tokenization.strip_punctuation(words)
            fingerprint = self._get_anchor_fingerprint_by_word(words_stripped, n)
            
        elif method == "winnow-k":
            fingerprint = self._get_winnow_k(document, n, k)

        return fingerprint
    
    def _get_full_fingerprint(self, words, n):
        fingerprint = []
        for i in xrange(len(words) - n + 1):
            fingerprint.append(self._gen_string_hash(" ".join(words[i:i+n])))
        return fingerprint

    def _get_kth_in_sent_fingerprint(self, sentences, n, k):
        fingerprint = []
        for sent in sentences:
            # Split the sentence the same way we tokenize based on words
            # and strip punctuation
            split_sent = nltk.tokenize.punkt.PunktWordTokenizer().tokenize(sent)
            split_sent = tokenization.strip_punctuation(split_sent)
            
            L = len(split_sent)
            # We want the n-gram beginning at word k, but if k > len(sentence) or n > len(sentence)
            # then we want the longest n-gram we can find (perhaps the whole sentence).
            fingerprint.append(self._gen_string_hash(" ".join(split_sent[min(k, L - min(n, L)) : min(k + n, L)])))
        return fingerprint

    def gen_anchors(anchor_length = 2, num_anchors=10):
        """
        This function should be called whenever we want to generate a new list of
        anchors. Just set self.anchors equal to the result of this function.
        """
        alphabet = string.ascii_lowercase
        # look at all permutations
        anchor_counts = {}
        for anchor in itertools.product(alphabet, repeat=anchor_length):
            anchor = "".join(anchor)
            anchor_counts[anchor] = 0
        print anchor_counts

        corp = nltk.corpus.gutenberg
        for filename in corp.fileids():
            print 'Counting anchors in', filename
            for anchor in anchor_counts.keys():
                results = re.findall(anchor, corp.raw(filename))
                anchor_counts[anchor] += len(results)

        # sort keys in decreasing order
        anchors = anchor_counts.keys()
        anchors = filter(lambda x: anchor_counts[x] != 0, sorted(anchors, key=lambda x: anchor_counts[x], reverse=True))
        for a in anchors:
            print a, anchor_counts[a]

        start_index = int(0.15*len(anchors))
        return anchors[start_index:start_index+num_anchors] 

    def _get_anchor_fingerprint(self, document, n):
        # anchors are start or middle of n-gram?
        fingerprint = []
        for anchor in self.anchors:
            # Our regular expression puts the word containing the anchor in the middle of the n-gram
            # tie is resolved with an extra word at the end
            regex = '\w+\s+' * ((n - 1) / 2) + '\w*' + anchor + '\w*' + '\s+\w+' * ((n - (n%2) )/2)
            for match in re.finditer(regex, document):
                fingerprint.append(self._gen_string_hash(match.group(0)))
        return fingerprint

    def _get_anchor_fingerprint_by_word(self, words, n):
        '''
        Finds all w in <words> such that w contains one of <self.anchors>,
        hashes the <n>gram surrounding each w, and returns the hashes
        of these surrounding <n>grams as the fingerprint of the document
        '''
        fingerprint = []

        for i in range(len(words)):
            cur_word = words[i]
            for anchor in self.anchors:
                if anchor in cur_word:
                    # Place anchor in middle of n-gram
                    start_index = max(0, i - (n - 1) / 2)
                    end_index = min(i + n / 2, len(words)) + 1

                    # Avoid too-short cases at start/end of text
                    if end_index - start_index == n:
                        ngram = ' '.join(words[start_index : end_index])
                        fingerprint.append(self._gen_string_hash(ngram))

        return fingerprint

    def _get_winnow_k(self, document, k, t):
        '''
        Takes as arguements a k to be used for k-grams and a noise threshold, t.

        Produces a full hash using k-grams of the text given and selects the minimum hash from
        each window where window size is t-k+1 if it is a new minimum.

        Example:

        for string: "They can't burn it down if we burn it down first."
        	k: 8
        	t: 14

        produces k grams from theycantburnitdownifweburnitdownfirst of size 8 and hashes them:

        theycant = 77
        heycantb = 64
        eycantbu = 18
        ...

        selects from all hashes the minimum within the window w = t-k+1 = 14-8+1 = 7

        window1:
        [77, 64, 18, 15, 98, 87, 45], 12, 15, 84, 65, 75, 35, ... ; selects 15

        window2:
        77, [64, 18, 15, 98, 87, 45, 12], 15, 84, 65, 75, 35, ... ; selects 12

        window3:
        77, 64, [18, 15, 98, 87, 45, 12, 15], 84, 65, 75, 35, ... ; 12 already selected so nothing selected.

        fingerprint = [15, 12, ... ]

        End Example.
        '''

        if t < k:
            raise Exception("Invalid input--noise threshold <t> must be bigger than <k>")

        document = "".join(self._strip_punctuation(document).lower().split())
        fingerprint = []
        document_hash = []

        # set window size
        w = t-k+1

        # produce k-grams
        for i in xrange(len(document)-k+1):
            document_hash.append(self._gen_string_hash(document[i:i+k]))

        if len(document_hash) == 0:
            return []

        first_min = document_hash[0]

        # select minimums from windows
        for i in xrange(len(document_hash)-w+1):
            window = document_hash[i:i+w]
            second_min = min(window)
            if first_min == second_min:
                if window[w-1] == second_min:
                    fingerprint.append(first_min)
                else:
                    continue
            else:
                first_min = second_min
                fingerprint.append(first_min)

        return fingerprint

    def _strip_punctuation(self, document):
        document = document.translate(string.maketrans("",""), string.punctuation)
        return document

class FingerprintEvaluator:

    def __init__(self, source_filenames, fingerprint_method, n, k):
        self.extractor = FingerprintExtractor()
        self.n = n
        self.k = k
        self.fingerprint_method = fingerprint_method
        self.source_filenames = source_filenames

    #def _get_fingerprint(self, filename, atom_type, session, base_path):
    #    fp = extrinsic_processing.query_fingerprint(filename, self.fingerprint_method, self.n, self.k, atom_type, session, base_path)
    #    return fp

    def classify_passage(self, filename, atom_type, atom_index, fingerprint_method, n, k, hash_len, confidence_method, mid, dids=None, passage_atom_type=None, passage_mid=None):
        '''
        Returns a list of (source_filename, similarity) tuples sorted in decreasing similarity to the 
        input document.
        '''
         # Get the full path from the filename and base_path
        full_path = ExtrinsicUtility.CORPUS_SUSPECT_LOC + filename + ".txt"
        
        # allow for different atom_types to be compared
        if not passage_atom_type:
            passage_atom_type = atom_type

        if not passage_mid:
            passage_mid = mid

        # Get the fingerprint of the passage in question
        fingerprint = fingerprintstorage.get_passage_fingerprint(full_path, atom_index, passage_atom_type, passage_mid)
        
        with psycopg2.connect(user = username, password = password, database = dbname.split("/")[1], host="localhost", port = 5432) as conn:
            conn.autocommit = True
            
            source_passages = {}
            for hash_value in fingerprint:
                if hash_value == 0: # There may be a bug with fingerprinting whereby most passages have 0 in their fingerprint.
                    continue
                matching_source_passages = fingerprintstorage.get_matching_passages(hash_value, mid, conn, dids=dids)
                for passage in matching_source_passages:
                    if (passage["doc_name"], passage["atom_number"]) not in source_passages:
                        source_passage_fp = fingerprintstorage.get_passage_fingerprint_by_id(passage["pid"], mid, conn)
                        if len(source_passage_fp):# make sure its not an empty fingerprint
                            source_passages[(passage["doc_name"], passage["atom_number"], passage["did"], filename, atom_index)] = get_plagiarism_confidence(fingerprint, source_passage_fp, confidence_method)
                        else:
                            source_passages[(passage["doc_name"], passage["atom_number"], passage["did"], filename, atom_index)] = 0
                            
        if not len(source_passages):
            source_passages[('dummy', 0, 0, filename, atom_index)] = 0
            
        return sorted(source_passages.items() , key = operator.itemgetter(1), reverse=True)

    def classify_and_display(self, doc):
        '''
        '''
        result = self.classify_passage(doc)
        print 'Using', self.fingerprint_method

        for src, sim in result:
            print os.path.basename(src), sim
        print 


def get_plagiarism_confidence(suspect_fingerprint, source_fingerprint, confidence_method):
    '''
    Wrapper function for set similarity measures.  Returns the similarity between the input
    fingerprint sets and uses the appropriate function as given by confidence_method.
    '''
    if confidence_method == "jaccard":
        return jaccard_similarity(suspect_fingerprint, source_fingerprint)
    elif confidence_method == "containment":
        return containment_similarity(suspect_fingerprint, source_fingerprint)
    else:
        raise Exception("Invalid plagiarism confidence method: " + confidence_method)

def jaccard_similarity(a, b):
    '''
    Measures the jaccard similarity of input sets a and b
    '''
    intersection_size = len(set(a).intersection(set(b)))
    # len([k for k in a if k in b])
    union_size = len(a) + len(b) - intersection_size
    if union_size > 0:
        return float(intersection_size) / union_size
    else:
        return 0

def containment_similarity(a, b):
    '''
    Measures the percent of elements in set a that are also contained in set b.
    '''
    intersection_size = len(set(a).intersection(set(b)))
    if len(a) > 0:
        return float(intersection_size) / len(a)
    else:
        return 0


def anchor_test():
    text = 'good, should be caught. also am should be, as well as cat. end of doc is here.'
    ex = FingerprintExtractor()
    print ex.get_fingerprint(text, 4, method='anchor')


if __name__ == '__main__':
    '''ex = FingerprintExtractor()
    corp = nltk.corpus.gutenberg

    util = ExtrinsicUtility()
    sources = util.get_sample_source_paths()
    
    full = FingerprintEvaluator(sources, "full")
    kth = FingerprintEvaluator(sources, "kth_in_sent")
    anchor = FingerprintEvaluator(sources, "anchor")
    '''
    ex = FingerprintExtractor()
    #text = "Hi my name is Marcus and I'm working in the CMC. Why does our project have to be so ridiculous."
    f = open("/copyCats/itty-bitty-corpus/suspicious/einstein.txt","r")
    text = f.read()
    f.close()
    print ex.get_fingerprint(text, 5, "winnow-k", 8)
