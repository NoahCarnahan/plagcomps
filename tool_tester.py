from controller import Controller
import xml.etree.ElementTree as ET

class ToolTester:
    
    def __init__(self, suspect):
    
        path_start = "/copyCats/pan-plagiarism-corpus-2009/intrinsic-detection-corpus/suspicious-documents/part1/"
        self.text_file_path = path_start + suspect + ".txt"
        self.xml_file_path = path_start + suspect + ".xml"

        self.c = Controller(self.text_file_path)
        self.atoms = self.c.feature_evaluator.getAllByAtom("sentence")
        self.features = self.c.extractFeatures("sentence")
        self.clusters = self.c.clusterFeatures(self.features, "kmeans", 2)
        
        self.plagiarised_spans = None
        self._is_plagiarised_cache = None
    
    def get_plagiarised_spans(self):
        '''
        Using the ground truth, return a list of spans representing the passages of the
        text that are plagiarized. 
        '''
        if self.plagiarised_spans == None:
            spans = []
            tree = ET.parse(self.xml_file_path)
            for feature in tree.iter("feature"):
                if feature.get("name") == "artificial-plagiarism": # are there others?
                    start = int(feature.get("this_offset"))
                    end = start + int(feature.get("this_length"))
                    spans.append((start, end))
            self.plagiarised_spans = spans
            
        return self.plagiarised_spans
    
    def _is_plagiarised(self, char):
        '''
        Returns True if the given character index is part of a passage that has been plagiarized
        (based on the group truth)
        '''
        # Cache would probably be faster if it was a list instead of a dict...
        if self._is_plagiarised_cache == None:
            self._is_plagiarised_cache = {}
        if char in self._is_plagiarised_cache:
            return self._is_plagiarised_cache[char]
        else:
            # linear bad!
            for i in self.get_plagiarised_spans():
                if i[0] <= char < i[1]:
                    self._is_plagiarised_cache[char] = True
                    return True
            self._is_plagiarised_cache[char] = False
            return False

    def _get_cluster(self, char):
        '''
        Return the number of the cluster created by our tool that the given character (index) is a part of.
        '''
        # linear search is bad! We should probably use the binary search that marcus and I
        # wrote. Perhaps we should pull it out into a utilities file.
        for atom_number in range(len(self.atoms)):
            if self.atoms[atom_number][0] <= char < self.atoms[atom_number][1]:
                return self.clusters[atom_number]
        return False
    
    def _in_same_baseline_group(self, char1, char2):
        '''
        Returns True if both given characters are plagiarized or if both given characters
        are not plagiarized according to the baseline.
        If one of the characters is part of a plagiarized passage and one is not, returns
        False.
        char1 and char2 are character indicies into the document.
        '''
        return self._is_plagiarised(char1) == self._is_plagiarised(char2)
    
    #TODO: REPLACE x WITH SOME WORD!! should it be "custom", "test", "classified", "tool", what?
    def _in_same_x_group(self, char1, char2):
        '''
        If our tool puts char1 and char2 in the same group, return True. Otherwise return 
        False.
        '''
        return self._get_cluster(char1) == self._get_cluster(char2)
    
    def main(self):
        matching = 0
        total = 0
        print "Total characters =", len(self.c.feature_evaluator.input_file)
        for char1 in range(len(self.c.feature_evaluator.input_file)):
            try:
                print "On character", char1, "..."
                for char2 in range(len(self.c.feature_evaluator.input_file)):
                    self._in_same_baseline_group(char1, char2) # This shows that even just looking at the baseline is too slow...
                    #if self._in_same_baseline_group(char1, char2) and self._in_same_x_group(char1, char2):
                    #    matching += 1
                    total += 1
            except IndexError:
                pass
        return float(matching) / total
            

if __name__ == "__main__":
    t = ToolTester("suspicious-document00997")
    print t.atoms
    print t.clusters
    # Why are these different lengths?
    print len(t.atoms)
    print len(t.clusters)
    #
    print t._get_cluster(6)
    print t._get_cluster(33890)
    
    print t.get_plagiarised_spans()

    print t._in_same_baseline_group(10, 11475)
    print t.main()