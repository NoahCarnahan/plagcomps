import os
import glob
import xml
import xml.etree.ElementTree as ET
from plagcomps import tokenization
from plagcomps.shared.passage import PassageWithGroundTruth

UTIL_LOC = os.path.abspath(os.path.dirname(__file__))

class BaseUtility:
    '''
    Utility functions common to both extrinsic and intrinsic
    '''
    SAMPLE_CORPUS_LOC = os.path.join(UTIL_LOC, '..', 'sample_corpus/')


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


    def get_bare_passages_and_plagiarized_spans(self, doc_path, xml_path, atom_type):
        '''
        TODO finish this: the idea is to create passage objects that contain information
        about whether or not each passage contains a plagiarized chunk of text
        '''
        f = file(doc_path, 'rb')
        text = f.read()
        f.close()

        plag_spans = self.get_plagiarized_spans(xml_path)

        spans = tokenization.tokenize(text, atom_type)
        all_passages = []

        for span in spans:
            overlap_plag = None

            start, end = span
            for pspan in plag_spans:
                # Note that a passage's <pspan> only holds the last 
                # plagiarized span that overlaps
                if self.overlap(pspan, span) > 0:
                    overlap_plag = pspan

            all_passages.append(PassageWithGroundTruth(start, end, text[start : end], overlap_plag))

        return all_passages

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

    def get_n_training_files(self, n=None, include_txt_extension=True, min_len=None):
        '''
        Returns first <n> training files, or all of them if <n> is not specified

        If <min_len> is specified, only return files which contain at least <min_len>
        characters. 
        '''
        all_training_files = self.read_file_list(IntrinsicUtility.TRAINING_LOC, 
                                                 IntrinsicUtility.CORPUS_LOC,
                                                 include_txt_extension=include_txt_extension)
        
        to_return = []
         # Default to using all training files if <n> isn't specified
        n = len(all_training_files) if n is None else n

        if min_len:
            # Read through files until finding <n> documents of 
            # length >= min_len
            for fname in all_training_files:
                if len(to_return) >= n:
                    break
                else:
                    f = file(fname, 'rb')
                    text = f.read()
                    f.close()

                    if len(text) >= min_len:
                        to_return.append(fname)

        else:
            to_return = all_training_files[:n]

        return to_return

class ExtrinsicUtility(BaseUtility):
    
    CORPUS_SRC_LOC = '/copyCats/pan-plagiarism-corpus-2009/external-detection-corpus/source-documents'
    CORPUS_SUSPECT_LOC = '/copyCats/pan-plagiarism-corpus-2009/external-detection-corpus/suspicious-documents'

    def get_training_files(self, n="all", path_type="absolute", file_type='both', include_txt_extension=True):
        '''
        Returns first <n> training files, or all of them if <n> is not specified
        <file_type> should be 'source', 'suspect', or 'both'.
        If 'both', return both lists (source files first, suspect files second)
        
        If path_type is "name" then just /part1/suspicious-file-XXXXX is returned.
        '''
        loc = os.path.join(os.path.dirname(__file__), "..", "extrinsic_corpus_partition/var_corp.txt")
        f = open(loc, "r")
        lines = f.readlines()
        f.close()
        suspicious_files = eval(lines[0])
        source_files = eval(lines[1])
        source_cutoff = eval(lines[2])
        
        if n == "all":
            n = len(suspicious_files)
            
        if path_type == "name":
            #strip base path
            for path in suspicious_files:
                path.replace(CORPUS_SUSPECT_LOC, "")
            for path in source_files:
                path.replace(CORPUS_SRC_LOC, "")
        if include_txt_extension == False:
            suspicious_files = [s[:-4]for s in suspicious_files]
            source_files = [s[:-4] for s in source_files]
                        
        if file_type == "source":
            return source_files[:source_cutoff[n-1]]
        elif file_type == "suspect" or file_type == "suspicious":
            return suspicious_files[:n]
        else:
            return source_files[:source_cutoff[n-1]], suspicious_files[:n]
    
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





