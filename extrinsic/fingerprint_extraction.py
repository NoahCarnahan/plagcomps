# fingerprint_extraction.py
# Module for generating fingerprints from documents.

import nltk
import itertools
import os
import string, random, re, operator
from .. import tokenization
from ..shared.util import ExtrinsicUtility
import extrinsic_processing

# TODO: omit words tokenized by nltk that are just puncuation

class FingerprintExtractor:

    def __init__(self):
        self.anchors = ['ul', 'ay', 'oo', 'yo', 'si', 'ca', 'am', 'ie', 'mo', 'rt']
        self.hash_span = 100000

    def _gen_string_hash(self, in_string):
        '''
        Converts the given string <in_string> to an integer which is
        roughly uniformly distributed over all possible values.

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

    def get_fingerprint(self, document, n, method="full", k=5):
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
            fingerprint = self._get_winnow_k(document, noise_threshold, guarantee_threshold)

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
        document = "".join(self._strip_punctuation(document).lower().split())
        print document
        fingerprint = []
        document_hash = []
        w = t-k+1

        for i in xrange(len(document)-k+1):
            document_hash.append(self._gen_string_hash(document[i:i+k]))

        first_min = document_hash[0]
        print document_hash

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

class FingerprintEvaluator:

    def __init__(self, source_filenames, fingerprint_method="full", n=3):
        self.extractor = FingerprintExtractor()
        self.n = n
        self.fingerprint_method = fingerprint_method
        self.source_filenames = source_filenames

    def _get_fingerprint(self, filename, atom_type, session, base_path):
        fp = extrinsic_processing._query_fingerprint(filename, self.fingerprint_method, self.n, 5, atom_type, session, base_path)
        return fp

    def classify_document(self, filename, atom_type, atom_index, session):
        '''
        Returns a list of (source_filename, similarity) tuples sorted in decreasing similarity to the 
        input document.
        '''
        fp = self._get_fingerprint(filename, atom_type, session, ExtrinsicUtility.CORPUS_SUSPECT_LOC)
        if atom_type == "full":
            fingerprint = fp.get_fingerprints(session)
        else:
            fingerprint = fp.get_fingerprints(session)[atom_index]

        source_documents = {}
        for source in self.source_filenames:
            print 'source:', source
            source_fp = self._get_fingerprint(source, atom_type, session, ExtrinsicUtility.CORPUS_SRC_LOC)
            source_fingerprints = source_fp.get_fingerprints(session)
            if atom_type == "full":
                if len(source_fingerprints): # make sure it's not an empty fingerprint
                    source_documents[(source, 0)] = jaccard_similarity(fingerprint, source_fingerprints)
                else:
                    source_documents[(source, 0)] = 0
            else:
                for i in xrange(len(source_fingerprints)):
                    if len(source_fingerprints[i]): # make sure it's not an empty fingerprint
                        source_documents[(source, i)] = jaccard_similarity(fingerprint, source_fingerprints[i])
                    else:
                        source_documents[(source, i)] = 0
        return sorted(source_documents.items() , key = operator.itemgetter(1), reverse=True)

    def classify_and_display(self, doc):
        '''
        '''
        result = self.classify_document(doc)
        print 'Using', self.fingerprint_method

        for src, sim in result:
            print os.path.basename(src), sim
        print 

def jaccard_similarity(a, b):
    '''a and b are lists (multisets) of fingerprints of which we want to find the similarity'''
    intersection_size = len(set(a).intersection(set(b)))
    # len([k for k in a if k in b])
    union_size = len(a) + len(b) - intersection_size
    if union_size > 0:
        return float(intersection_size) / union_size
    else:
        return 0

def anchor_test():
    text = 'good, should be caught. also am should be, as well as cat. end of doc is here.'
    ex = FingerprintExtractor()
    print ex.get_fingerprint(text, 4, method='anchor')

if __name__ == '__main__':
    ex = FingerprintExtractor()
    corp = nltk.corpus.gutenberg

    util = ExtrinsicUtility()
    sources = util.get_sample_source_paths()
    
    full = FingerprintEvaluator(sources, "full")
    kth = FingerprintEvaluator(sources, "kth_in_sent")
    anchor = FingerprintEvaluator(sources, "anchor")

    # TODO this won't work right now. 
    # is there anything being tested here that isn't done in extrinsic_testing?
# # ev = FingerprintEvaluator(["sample_corpus/easy_source.txt"])
# # ev.classify_document("sample_corpus/easy_test.txt")

#   test_files = util.get_sample_test_paths()

#   for test_doc in test_files:
#       print "testing", os.path.basename(test_doc)
#       full_doc = open(test_doc, 'r')
#       text = full_doc.read()
#       full_doc.close()

#       full.classify_and_display(text)
#       kth.classify_and_display(text)
#       anchor.classify_and_display(text)
#       print '-'*20

#   #print ex.get_fingerprint(corp.raw("austen-sense.txt"), 3, "anchor")
#   #print ex.get_fingerprint(corp.raw("austen-emma.txt"), 3, "anchor")
 
