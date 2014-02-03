import os
import glob
import xml
import xml.etree.ElementTree as ET
import json

from plagcomps import tokenization
from plagcomps.shared.passage import PassageWithGroundTruth

import sklearn.metrics
import matplotlib.pyplot as pyplot
import time

UTIL_LOC = os.path.abspath(os.path.dirname(__file__))

class BaseUtility:
    '''
    Utility functions common to both extrinsic and intrinsic
    '''

    SAMPLE_CORPUS_LOC = os.path.join(UTIL_LOC, '..', 'sample_corpus/')

    @staticmethod
    def draw_roc(actuals, confidences, save_figure=True, **metadata):
        '''
        Draws an ROC curve based on <actuals> and <confidences> and saves
        the figure to figures/roc<timestamp>

        The optional <metadata> are written in the title of the figure

        The path to the figure, and area under the curve are returned 
        '''
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(actuals, confidences, pos_label=1)
        roc_auc = sklearn.metrics.auc(fpr, tpr)
        
        # The following code is from http://scikit-learn.org/stable/auto_examples/plot_roc.html
        pyplot.clf()
        pyplot.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        pyplot.plot([0, 1], [0, 1], 'k--')
        pyplot.xlim([0.0, 1.0])
        pyplot.ylim([0.0, 1.0])
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')

        title = 'ROC'

        for arg_name, arg_val in metadata.iteritems():
            title += ', ' + str(arg_val)

        pyplot.title(title)
        pyplot.legend(loc="lower right")
        
        figure_path = os.path.join(os.path.dirname(__file__), "../figures/roc"+str(time.time())+".pdf")
        json_path = figure_path.replace('pdf', 'json')
        pyplot.savefig(figure_path)

        # Save a JSON file of metadata about figure
        metadata['auc'] = roc_auc
        with open(json_path, 'wb') as f:
            json.dump(metadata, f, indent=4)

        return figure_path, roc_auc

    @staticmethod
    def get_corpus_name(full_path):
        '''
        Returns the name of the corpus from which <full_path> came (i.e. intrinsic or
        extrinsic)
        '''
        if 'intrinsic-detection-corpus' in full_path:
            return 'intrinsic'
        elif 'external-detection-corpus' in full_path:
            return 'extrinsic'
        else:
            print "%s didn't come from either the intrinsic or extrinsic corpus!" % full_path
            return 'unknown_corpus'

    def read_file_list(self, file_name, base_location_path, include_txt_extension=True, min_len=None):
        '''
        Return list of absolute paths to files in <file_name> whose
        location is relative to <base_location_path>
        '''
        relative_paths = self.get_relative_training_set(file_name, include_txt_extension)
        training_list = [base_location_path + r for r in relative_paths]

        return training_list

    def get_relative_training_set(self, file_name, include_txt_extension=True):
        '''
        Return list of relative paths to files in <file_name>
        '''
        f = file(file_name, 'r')
        if include_txt_extension:
            relative_paths = [l.strip() + '.txt' for l in f.readlines()]
        else:
            relative_paths = [l.strip() for l in f.readlines()]

        f.close()

        return relative_paths

    def get_plagiarized_spans(self, xml_path):
        '''
        Using the ground truth, return a list of spans representing the passages of the
        text that are plagiarized. Note, this method was plagiarized from Noah's intrinsic
        testing code.
        '''
        spans = [] 
        tree = xml.etree.ElementTree.parse(xml_path)

        for feature in tree.iter("feature"):
            if feature.get("name") == "artificial-plagiarism":
                start = int(feature.get("this_offset"))
                end = start + int(feature.get("this_length"))
                spans.append((start, end))
        return spans

    def add_ground_truth_to_passages(self, passages, xml_path):
        plag_spans = self.get_plagiarized_spans(xml_path)
        
        for p in passages:
            for s in plag_spans:
                # A tuple if there is any overlap, otherwise None
                overlap = self.overlap((p.char_index_start, p.char_index_end), s, return_length=False)

                if overlap:
                    p.add_plag_span(overlap)

    def overlap(self, interval1, interval2, return_length=True):
        '''
        If <return_length>,
        returns the length of the overlap between <interval1> and <interval2>,
        both of which are tuples. 

        Otherwise returns a tuple of the overlapping character indices, if there 
        is any overlap. If there is no overlap, returns None 
        '''
        start_overlap = max(interval1[0], interval2[0])
        end_overlap = min(interval1[1], interval2[1])
        diff = end_overlap - start_overlap - 1

        overlap_length = max(0, diff)

        if return_length:
            return overlap_length
        elif overlap_length > 0:
            # Overlap and expecting a tuple returned
            return (start_overlap, end_overlap)
        else:
            # No overlap 
            return None

class IntrinsicUtility(BaseUtility):

    TRAINING_LOC = os.path.join(UTIL_LOC, '..', 'corpus_partition/training_set_files.txt')
    CORPUS_LOC = '/copyCats/pan-plagiarism-corpus-2009/intrinsic-detection-corpus/suspicious-documents'

    def read_corpus_file(self, rel_path):
        '''
        <rel_path> should be like 'part1/suspicious-document0000
        Returns the text stored in <rel_path>'s file
        '''
        full_path = os.path.join(IntrinsicUtility.CORPUS_LOC, rel_path + '.txt')

        f = file(full_path, 'rb')
        text = f.read()
        f.close()

        return text

    def get_n_training_files(self, n=None, include_txt_extension=True, min_len=None, first_doc_num=0, pct_plag=None):
        '''
        Returns first <n> training files, or all of them if <n> is not specified

        <first_doc_num> defaults to 0, and indicates where the <n> documents should
        start counting from

        If <min_len> is specified, only return files which contain at least <min_len>
        characters. 

        If <pct_plag> is specified, then <pct_plag> of the <n> returned files will contain plagiarism 
        '''
        all_training_files = self.read_file_list(IntrinsicUtility.TRAINING_LOC, 
                                                 IntrinsicUtility.CORPUS_LOC,
                                                 include_txt_extension=include_txt_extension)
        
         # Default to using all training files if <n> isn't specified
        n = len(all_training_files) if n is None else n

        # Skip complicated processing if making a simple query
        if not min_len and not pct_plag:
            return all_training_files[first_doc_num : first_doc_num + n]

        # Otherwise, begin processing based on other restrictions
        if pct_plag:
            n_plag = n*pct_plag
            n_no_plag = n - n_plag

        plag_docs = []
        no_plag_docs = []
            
        for fname in all_training_files[first_doc_num:]:
            # Accumulated <n> files already
            if len(plag_docs) + len(no_plag_docs) >= n:
                break

            xml_path = fname.replace('txt', 'xml')
            fvalid = True

            # Check for plag in document
            plag_spans = self.get_plagiarized_spans(xml_path)
            contains_plag = len(plag_spans) > 0

            # Filter out short files, if min_len is specified
            if min_len:
                f = file(fname, 'rb')
                text = f.read()
                f.close()

                if len(text) < min_len:
                    fvalid = False

            # fname is NOT VALID if
            # it contains plag but we already have n_plag plagiarized docs OR
            # it doesn't contain plag, but we already have n_no_plag non-plagiarized docs
            if pct_plag:
                if not ((contains_plag and len(plag_docs) < n_plag) or \
                        (not contains_plag and len(no_plag_docs) < n_no_plag)):
                    fvalid = False

            if fvalid and contains_plag:
                plag_docs.append(fname)
            elif fvalid and not contains_plag:
                no_plag_docs.append(fname)

        return plag_docs + no_plag_docs


class ExtrinsicUtility(BaseUtility):

    # TRAINING_SRC_LOC = os.path.join(UTIL_LOC, '..', 'extrinsic_corpus_partition/extrinsic_training_source_files.txt')
    # TRAINING_SUSPECT_LOC = os.path.join(UTIL_LOC, '..', 'extrinsic_corpus_partition/extrinsic_training_suspect_files.txt')
    TRAINING_SRC_LOC = os.path.join(UTIL_LOC, '..', 'extrinsic_corpus_partition/small_sample_corpus/sample_source_listing2.txt')
    TRAINING_SUSPECT_LOC = os.path.join(UTIL_LOC, '..', 'extrinsic_corpus_partition/small_sample_corpus/sample_suspect_listing2.txt')

    CORPUS_SRC_LOC = '/copyCats/pan-plagiarism-corpus-2009/external-detection-corpus/source-documents'
    CORPUS_SUSPECT_LOC = '/copyCats/pan-plagiarism-corpus-2009/external-detection-corpus/suspicious-documents'

    def get_n_training_files(self, n=None, file_type='both', include_txt_extension=True):
        '''
        Returns first <n> training files, or all of them if <n> is not specified
        <file_type> should be 'source', 'suspect', or 'both'.
        If 'both', return both lists (source files first, suspect files second)

        NOTE that all source documents are returned, regardless of the size
        of <n>. We always need to know all source documents in order
        to detect plagiarism in any of the suspicious ones!
        '''
        all_src_files = self.read_file_list(ExtrinsicUtility.TRAINING_SRC_LOC,
                                            ExtrinsicUtility.CORPUS_SRC_LOC,
                                            include_txt_extension=include_txt_extension)

        all_suspect_files = self.read_file_list(ExtrinsicUtility.TRAINING_SUSPECT_LOC,
                                                ExtrinsicUtility.CORPUS_SUSPECT_LOC,
                                                include_txt_extension=include_txt_extension)
        
        n = len(all_suspect_files) if n is None else n

        if file_type == 'source':
            return all_src_files
        elif file_type == 'suspect':
            return all_suspect_files[:n]
        elif file_type == 'both':
            return all_src_files, all_suspect_files[:n]
        else:
            raise Exception("Argument <file_type> must be 'source', 'suspect' or 'both'")

    def get_sample_source_paths(self):
        sample_sources = glob.glob(ExtrinsicUtility.SAMPLE_CORPUS_LOC + 'source*txt')

        return sample_sources

    def get_sample_test_paths(self):
        sample_tests = glob.glob(ExtrinsicUtility.SAMPLE_CORPUS_LOC + 'test*txt')

        return sample_tests

    def get_source_plag(self, suspect_path):
        '''
        <suspect_path> is a full path to an .xml file

        Returns the full paths of all source documents that are plagiarized in
        <suspect_path>'s corresponding document
        '''
        tree = ET.parse(suspect_path)
        sources = set()

        for feature in tree.iter('feature'):
            if feature.get('name') == 'artificial-plagiarism':
                src = feature.get('source_reference')
                full_path = self.get_src_abs_path(src)
                sources.add(full_path)

        return sources    

    def get_src_abs_path(self, doc_name):
        '''
        <doc_name> is like "source-document10172.txt" -- find which part of 
        the corpus the source document is in and return an absolute path 
        (i.e.) /copyCats/......./part6/source-document10172.txt
        '''
        possible_dirs = os.listdir(ExtrinsicUtility.CORPUS_SRC_LOC)
        for candidate in possible_dirs:
            full_path = os.path.join(ExtrinsicUtility.CORPUS_SRC_LOC, candidate, doc_name)
            if os.path.exists(full_path):
                return full_path

if __name__ == '__main__':
    # To test gen_n_training_files:
    # python -m plagcomps.shared.util | xargs grep -m 1 "artificial-plagiarism" | wc -l
    # which should output n*pct_plag 
    util = IntrinsicUtility()
    trainers = util.get_n_training_files(n=200, first_doc_num=0, pct_plag=.5)
    xmls = [x.replace('txt', 'xml') for x in trainers]
    for x in xmls:
        print x
