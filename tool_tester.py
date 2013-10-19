from controller import Controller
import xml.etree.ElementTree as ET

class ToolTester:
    
    def __init__(self, suspect):
    
        path_start = "/copyCats/pan-plagiarism-corpus-2009/intrinsic-detection-corpus/suspicious-documents/part1/"
        self.text_file_path = path_start + suspect + ".txt"
        self.xml_file_path = path_start + suspect + ".xml"

        c = Controller(self.text_file_path)
        self.atoms = c.feature_evaluator.getAllByAtom("sentence")
        self.features = c.extractFeatures("sentence")
        self.clusters = c.clusterFeatures(self.features, "kmeans", 2)
        
        self.plagiarised_spans = None
    
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
        # linear bad!
        for i in self.get_plagiarised_spans:
            if i[0] <= char < i[1]:
                return True
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

if __name__ == "__main__":
    t = ToolTester("suspicious-document00999")
    print t.atoms
    print t.clusters
    # Why are these different lengths?
    print len(t.atoms)
    print len(t.clusters)
    #
    print t._get_cluster(6)
    print t._get_cluster(33890)
    
    print t.get_plagiarised_spans()