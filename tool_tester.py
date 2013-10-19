from controller import Controller

class ToolTester:
    
    def __init__(self, suspect):
    
        path_start = "/copyCats/pan-plagiarism-corpus-2009/intrinsic-detection-corpus/suspicious-documents/part1/"
        self.text_file_path = path_start + suspect + ".txt"
        self.xml_file_path = path_start + suspect + ".xml"

        c = Controller(self.text_file_path)
        features = c.extractFeatures("sentence")
        self.clusters = c.clusterFeatures(features, "kmeans", 2)

    def _get_cluster(self, char):
        '''
        Return the number of the cluster that the given character (index) is a part of.
        '''
        pass

if __name__ == "__main__":
    ToolTester("suspicious-document00999")
    print ToolTester.clusters