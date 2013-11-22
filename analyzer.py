import test_framework, operator

class Analyzer:
	'''
    This class allows for the combined intrinsic/extrinsic analysis of a document.
    Given a document, analyze(...) will run intrinsic analysis on that document, and
    identify the most suspicious passages. Then, it will run extrinsic detection on
    each of the suspicious passages. It will return a % likelihood of plagiarism
    for each suspicious paragraph in terms of intrinsic and extrinsic analysis.
    '''

    def __init__(self, features, cluster_type, k, atom_type, fingerprint_type):
        self.features = features
        self.cluster_type = cluster_type
        self.k = k
        self.atom_type = atom_type
        self.fingerprint_type = fingerprint_type

        # what confidence must we have in the plagiarization of a passage 
        # (based on intrinsic) to consider a passage suspicious?
        self.intrinsic_suspicion_threshold = .85

    def analyze(self, doc):
        '''Run intrinsic and extrinsic detection on the document'''
        
        intrinsic_results = test_framework.run_intrinsic(self.features, self.cluster_type, self.k, self.atom_type, doc)
        intrinsic_results = sorted(intrinsic_results, key=operator.itemgetter(2), reverse=True)
        intrinsic_results = [x for x in intrinsic_results if x[2] > self.intrinsic_suspicion_threshold]
        # intrinsic_results now holds a list of tuples in which the tuple holds start and stop indices
        #   of a span within the reduced doc. Right now, the only way to get at the reduced doc is
        #   through the test_framework class, and it's supposed to be a private method, which is a problem

        for intrinsic_result in intrinsic_results:
            pass
            # 1: Find the fingerprint for this passage           
                # So get it from the database? or recreate it. This means that we need a way to interface
                # with fingerprint extractor with a span or a paragraph

            # 2: Look through all the source documents' fingerprints, and run jaccard similarity on each
                # Now we should have a confidence-of-plagiarism from the extrinsic world
                # for each of our passages, and we can list potential sources of plagiarism.


if __name__ == "__main__":

    analyzer = Analyzer(['averageSentenceLength', 'averageWordLength'], 'kmeans', 2, 'paragraph', 'anchor')
    analyzer.analyze('/part4/suspicious-document06242')
