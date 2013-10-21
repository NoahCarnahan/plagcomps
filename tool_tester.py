from controller import Controller

import xml.etree.ElementTree as ET
from collections import Counter

DEBUG = True



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
        
        #self.atom_spans = self.c.feature_evaluator.getAllByAtom(atom_type)
        #self.atom_features = self.c.extractFeatures(atom_type)
        #self.atom_clusters = self.c.clusterFeatures(self.atom_features, "kmeans", 2)
        self.passages = self.c.get_passages(atom_type, ["averageWordLength", "averageSentenceLength"], "kmeans", 2)
        
        self._smallest_cluster = None #The number of the smallest cluster.
        self._plagiarized_spans = None
        
        #self._is_char_plagiarized_cache = None
        
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
            cluster_nums = [x for p.cluster_num in self.passages]
            self._smallest_cluster = Counter(cluster_nums).most_common()[-1][0]
        return self._smallest_cluster
    
    def _is_plagiarized(self, p):
        '''
        p is a passage. Return True if the first character of the passage is in a plagiarized span.
        '''
        # TODO: Consider other ways to judge if an atom is plagiarized or not. For example, look to see if the WHOLE atom in a plagiarized segment (?)
        for i in self.get_plagiarized_spans():
            if i[0] <= p.start < i[1]:
                return True
        return False

    def _is_in_smallest_cluster(self, p):
        '''
        p is a passage
        Returns True if the given passage p is in the smaller of the two clusters.
        '''
        return p.cluster_num == self._get_smallest_cluster()
    
    def main(self):
        '''
        Returns (number of correctly classified passages) / (number of passages)
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
        
        for p in passages:
            if DEBUG and float(total) / len(self.passages) > amount_done[amount_done_i] / 100.0:
                print str(amount_done[amount_done_i])+"% done..."
                amount_done_i += 1

            if self._is_plagiarized(p) == self._is_in_smallest_cluster(p):
                correct +=1
            total += 1
        
        return float(correct) / total
    
    def atom-to-atom(self):
        '''
        Returns the value of the atom-to-atom metric.
        '''
        matching = 0
        total = 0
        for p1 in self.passages:
            for p2 in self.passages:
                if (self._is_plagiarized(p1) == self._is_plagiarized(p2)) and (p1.cluster_num == p2.cluster_num):
                    matching += 1
                total += 1
        return float(matching) / total
        
if __name__ == "__main__":
    t = ToolTester("suspicious-document00969", "paragraph")
    print t.atom-to-atom()
