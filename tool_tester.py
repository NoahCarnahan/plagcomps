from controller import Controller
import xml.etree.ElementTree as ET

class ToolTester:
    
    def __init__(self, suspect):
    
        path_start = "/copyCats/pan-plagiarism-corpus-2009/intrinsic-detection-corpus/suspicious-documents/part1/"
        self.text_file_path = path_start + suspect + ".txt"
        self.xml_file_path = path_start + suspect + ".xml"

        self.c = Controller(self.text_file_path)
        self.sentence_spans = self.c.feature_evaluator.getAllByAtom("sentence")
        self.sentence_features = self.c.extractFeatures("sentence")
        self.sentence_clusters = self.c.clusterFeatures(self.sentence_features, "kmeans", 2)
        
        self._smaller_cluster = None #The name (which is an int) of the smallest cluster.
        self._plagiarized_spans = None
        
        self._is_plagiarized_cache = None
        
        
     
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
    
    def _get_smaller_cluster(self):
        '''
        Return the name of (its an int) the smallest of the two clusters. Assumes only two clusters!
        '''
        if self._smaller_cluster == None:
            size_0 = 0
            size_1 = 0
            for i in self.sentence_clusters:
                if i == 0:
                    size_0 += 1
                elif i == 1:
                    size_1 += 1
            if size_0 < size_1:
                self._smaller_cluster = 0
            elif size_1 < size_0:
                self._smaller_cluster = 1
        return self._smaller_cluster
        
        
    def _is_plagiarized(self, char):
        '''
        Returns True if the given character index is part of a passage that has been plagiarized
        (based on the group truth)
        '''
        # Cache would probably be faster if it was a list instead of a dict...
        if self._is_plagiarized_cache == None:
            self._is_plagiarized_cache = {}
        if char in self._is_plagiarized_cache:
            return self._is_plagiarized_cache[char]
        else:
            # linear bad!
            for i in self.get_plagiarized_spans():
                if i[0] <= char < i[1]:
                    self._is_plagiarized_cache[char] = True
                    return True
            self._is_plagiarized_cache[char] = False
            return False
    
    def _is_sentence_plagiarized(self, s):
        '''
        Returns true if the first character of the sentence is in a plagiarized span. This may not
        be the best way to judge if a sentence is plagiarized or not, but it is easy!
        '''
        for i in self.get_plagiarized_spans():
            if i[0] <= self.sentence_spans[s][0] < i[1]:
                return True
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
    
    def _get_sentence_cluster(self, s):
        '''
        Return the number of the cluster created by our tool that the given sentence span is a part of.
        '''
        for atom_index in range(len(self.sentence_spans)):
            if self.sentence_spans[atom_index] == s:
                return self.sentence_clusters[atom_index]
        return False
    
    def _in_same_ground_truth_group(self, char1, char2):
        '''
        Returns True if both given characters are plagiarized or if both given characters
        are not plagiarized according to the ground truth.
        If one of the characters is part of a plagiarized passage and one is not, returns
        False.
        char1 and char2 are character indicies into the document.
        '''
        return self._is_plagiarized(char1) == self._is_plagiarized(char2)
    
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
    
    def _is_in_smaller_cluster(self, s):
        '''
        Returns True if the given sentnce s is in the smaller of the two clusters.
        '''
        return self._get_sentence_cluster(2) == self._get_smaller_cluster()
    
    
    def main_2(self):
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
    
    def main(self):
        '''
        Returns (number of correctly classified sentnces) / (number of sentences)
        Assumes that the smaller of the two predicted clusters is the cluster of plagiarized passages.
        '''
        
        right = 0
        total = 0
        
        print "Total sentences:", len(self.c.feature_evaluator.sentence_spans)
        amount_done = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        amount_done_i = 0
        
        if len(self.get_plagiarized_spans()) == 0:
            print "NOTE: no plagiarzed passages in this document"
            
        for s in range(len(self.c.feature_evaluator.sentence_spans)):
        
            # debug code:
            #print "On sentece", s            
            if float(total) / len(self.c.feature_evaluator.sentence_spans) > amount_done[amount_done_i] / 100.0:
                print str(amount_done[amount_done_i])+"% done..."
                amount_done_i += 1
            
            if self._is_sentence_plagiarized(s) == self._is_in_smaller_cluster(s):
                right += 1
            total += 1
        return float(right) / total
        
                        
                    
            

if __name__ == "__main__":
    t = ToolTester("suspicious-document00999")

    # Why are these different lengths sometimes?
    #print len(t.sentence_spans)
    #print len(t.sentence_clusters)

    print t.main()