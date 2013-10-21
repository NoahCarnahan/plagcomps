from controller import Controller

import xml.etree.ElementTree as ET
from collections import Counter

DEBUG = True

#TODO:
# Come up with a name for the other metric

#NOTES:
# The metric which I am calling the atom-to-atom metric compares every atom to every
# other atom and returns:
# |{pair (x, y) : x and y in same ground truth group AND x and y in same predicted groups}| / (total number of pairs)

class ToolTester:
    
    def __init__(self, suspect, atom_type):
        #atom_type = "word", "sentence", or "paragraph"
        #atom_type specifies what atoms the clusterer should use
    
        path_start = "/copyCats/pan-plagiarism-corpus-2009/intrinsic-detection-corpus/suspicious-documents/part1/"
        self.text_file_path = path_start + suspect + ".txt"
        self.xml_file_path = path_start + suspect + ".xml"

        self.c = Controller(self.text_file_path)
        
        self.atom_spans = self.c.feature_evaluator.getAllByAtom(atom_type)
        self.atom_features = self.c.extractFeatures(atom_type)
        self.atom_clusters = self.c.clusterFeatures(self.atom_features, "kmeans", 2)
        
        self._smallest_cluster = None #The number of the smallest cluster.
        self._plagiarized_spans = None
        
        self._is_char_plagiarized_cache = None
        
    def get_plagiarized_spans(self):
        '''
        Using the ground truth, return a list of spans representing the passages of the
        text that are plagiarized. 
        '''
        if self._plagiarized_spans == None:
            spans = []
            tree = ET.parse(self.xml_file_path)
            for feature in tree.iter("feature"):
                if feature.get("name") == "artificial-plagiarism": # are there others?
                    start = int(feature.get("this_offset"))
                    end = start + int(feature.get("this_length"))
                    spans.append((start, end))
            self._plagiarized_spans = spans
            
        return self._plagiarized_spans
    
    def _get_smallest_cluster(self):
        '''
        Return the number of the smallest of the two clusters.
        '''
        if self._smallest_cluster == None:
            self._smallest_cluster = Counter(self.atom_clusters).most_common()[-1][0]
        return self._smallest_cluster
    
    def _is_plagiarized(self, a):
        '''
        a is an index into self.atom_spans. self.atom_spans[a] is a (start index, end index)
        span representing the location of the atom in the suspicious document.
        Return True if the first character of the atom is in a plagiarized span.
        '''
        # TODO: Consider other ways to judge if an atom is plagiarized or not. For example, look to see if the WHOLE atom in a plagiarized segment (?)
        for i in self.get_plagiarized_spans():
            if i[0] <= self.atom_spans[a][0] < i[1]:
                return True
        return False

    def _is_in_smallest_cluster(self, a):
        '''
        a is an index into self.atom_spans
        Returns True if the given atom a is in the smaller of the two clusters.
        '''
        return self.atom_clusters[a] == self._get_smallest_cluster()
    
    def main(self):
        '''
        Returns (number of correctly classified atoms) / (number of atoms)
        Assumes that the smaller of the two predicted clusters is the cluster of plagiarized passages
        '''
        #TODO: Name this metric and rename this function appropriately
        
        if DEBUG:
            if len(self.get_plagiarized_spans()) == 0:
                print "NOTE: no plagiarzed passages in this document"
            print "Total sentences:", len(self.c.feature_evaluator.sentence_spans)
            amount_done = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            amount_done_i = 0
            
        correct = 0
        total = 0
        
        for a in range(len(self.atom_spans)):
            if DEBUG and float(total) / len(self.atom_spans) > amount_done[amount_done_i] / 100.0:
                print str(amount_done[amount_done_i])+"% done..."
                amount_done_i += 1

            if self._is_plagiarized(a) == self._is_in_smallest_cluster(a):
                correct +=1
            total += 1
        
        return float(correct) / total
    
    #####
    # The following methods are used for the atom-to-atom metric and may no longer work
    # given the changes that were made for the new metric.
    #####

    def _is_char_plagiarized(self, char):
        #NOTE: This can be replaced by _is_plagiarized once feature_evaluator is completed.
        '''
        Returns True if the given character index is part of a passage that has been plagiarized
        (based on the group truth)
        '''
        # Cache would probably be faster if it was a list instead of a dict...
        if self._is_char_plagiarized_cache == None:
            self._is_char_plagiarized_cache = {}
        if char in self._is_char_plagiarized_cache:
            return self._is_char_plagiarized_cache[char]
        else:
            # linear bad!
            for i in self.get_plagiarized_spans():
                if i[0] <= char < i[1]:
                    self._is_char_plagiarized_cache[char] = True
                    return True
            self._is_char_plagiarized_cache[char] = False
            return False

    def _get_cluster(self, char):
        '''
        Return the number of the cluster created by our tool that the given character (index) is a part of.
        '''
        # linear search is bad! We should probably use the binary search that marcus and I
        # wrote. Perhaps we should pull it out into a utilities file.
        for atom_number in range(len(self.sentence_spans)):
            if self.sentence_spans[atom_number][0] <= char < self.sentence_spans[atom_number][1]:
                return self.sentence_clusters[atom_number]
        return False
    
    def _in_same_ground_truth_group(self, char1, char2):
        '''
        Returns True if both given characters are plagiarized or if both given characters
        are not plagiarized according to the ground truth.
        If one of the characters is part of a plagiarized passage and one is not, returns
        False.
        char1 and char2 are character indicies into the document.
        '''
        return self._is_char_plagiarized(char1) == self._is_char_plagiarized(char2)

    def _in_same_ground_truth_sentence_group(self, s1, s2):
        '''
        Returns True if both given sentences are plagiarized or if both given sentences are not
        plagiarized according to the ground truth.
        Returns False otherwise.
        '''
        return self._is_sentence_plagiarized(s1) == self._is_sentence_plagiarized(s2)

    def _in_same_predicted_group(self, char1, char2):
        '''
        If our tool puts char1 and char2 in the same group, return True. Otherwise return 
        False.
        '''
        return self._get_cluster(char1) == self._get_cluster(char2)

    def _in_same_predicted_sentence_group(self, s1, s2):
        '''
        If our tool puts s1 and s2 (both are sentnces) in the same group, return True. Otherwise,
        Return False.
        '''
        return self._get_sentence_cluster(s1) == self._get_sentence_cluster(s2)

    def main_2(self):
        '''
        Returns the value of the atom-to-atom metric where the atoms are characters
        '''
        matching = 0
        total = 0
        print "Total characters =", len(self.c.feature_evaluator.input_file)
        for char1 in range(len(self.c.feature_evaluator.input_file)):
            print "On character", char1, "..."
            for char2 in range(len(self.c.feature_evaluator.input_file)):
                try:
                    #self._in_same_ground_truth_group(char1, char2) # This shows that even just looking at the baseline is too slow...
                    if self._in_same_ground_truth_group(char1, char2) and self._in_same_predicted_group(char1, char2):
                        matching += 1
                    total += 1
                except IndexError:
                    pass
        return float(matching) / total

    def main_3(self):
        '''
        Returns the value of the atom-to-atom metric where the atoms are sentences
        '''
        matching = 0
        total = 0
        for sentence1 in range(len(self.c.feature_evaluator.sentence_spans)):
            print "On sentence", sentence1, "of", len(self.c.feature_evaluator.sentence_spans)
            for sentence2 in range(len(self.c.feature_evaluator.sentence_spans)):
                try:
                    if self._in_same_ground_truth_sentence_group(sentence1, sentence2) and self._in_same_predicted_sentence_group(sentence1, sentence2):
                        matching += 1
                    total += 1
                except IndexError:
                    pass
        return float(matching) / total
        
if __name__ == "__main__":
    t = ToolTester("suspicious-document00969", "paragraph")
    print t.main()
