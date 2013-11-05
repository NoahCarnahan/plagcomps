import time
import numpy
import scipy

class ROCTrial:

    def __init__(self, doc_name, features, cluster_strat, atom_type, plag_cluster, non_plag_cluster, passages):
        self.doc_name = doc_name
        self.features = features
        self.cluster_strategy = cluster_strat
        self.atom_type = atom_type
        self.plag_cluster = plag_cluster
        self.non_plag_cluster = non_plag_cluster
        self.passages = passages
        self.time_stamp = time.time()
            
    
    def get_success_statistics(self, threshold = None):
        false_positives = 0
        false_negatives = 0
        true_positives = 0
        true_negatives = 0
        
        for p in self.passages:
            is_plag = p["is_gt_plagiarism"]
            #if is_plag:
                #print "got a plag!"
            classified_as_plag = self._is_plagiarism(p, threshold)
            if classified_as_plag and is_plag:
                true_positives += 1
            elif classified_as_plag and not is_plag:
                false_positives += 1
            elif not classified_as_plag and is_plag:
                false_negatives += 1
            elif not classified_as_plag and not is_plag:
                true_negatives += 1
        
        return {"tp":true_positives,"fp":false_positives,"tn":true_negatives,"fn":false_negatives}
    
          

    def get_false_positive_rate(self, threshold = None):
        stats = self.get_success_statistics(threshold)
        fp = stats["fp"]
        tn = stats["tn"]
        try:
            return  fp / float(fp+tn)
        except ZeroDivisionError:
            return 0
    
    def get_true_positive_rate(self, threshold = None):
        stats = self.get_success_statistics(threshold)
        tp = stats["tp"]
        fn = stats["fn"]
        try:
            return tp / float(tp + fn)
        except ZeroDivisionError:
            return 0
    
    def _is_plagiarism(self, p, t = None):
        '''
        Returns true if we classified passage <p> as plagiarism. If a threshold <t> is given,
        returns true if we classified this passage <p> as plagiarism with above <t> certainty.
        '''
        if t == None:
            return p["cluster"] == self.plag_cluster["num"]
        else:
            return self._get_confidence(p) >= t
    
    def _get_confidence(self, p):
        '''
        Returns how confident we are that the given passage is plagiarized.
        '''
        d_from_plag_c = float(scipy.spatial.distance.pdist(numpy.matrix([self.plag_cluster["centroid"], p["feature_vector"].values()]), "euclidean")[0])
        d_from_notplag_c = float(scipy.spatial.distance.pdist(numpy.matrix([self.non_plag_cluster["centroid"], p["feature_vector"].values()]), "euclidean")[0])
        conf = 1 - d_from_plag_c / (d_from_plag_c + d_from_notplag_c)
        return conf
        
    def get_dictionary_representation(self):
        '''
        Return a dictionary representation of this ROCTrial in the following format:
        
        {
            "doc_name": test_file,
            "features": features,
            "time_stamp": 1383422140.466412,
            "cluster_strategy": "kmeans",
            "atom_type": "paragraph",
            "plag_cluster": {"num": <0 or 1> , "centroid": <numpy array?>},
            "non_plag_cluster": {"num": <0 or 1> , "centroid": <numpy array?>},
            "passages": [
                {"cluster": <0 or 1>, "feature_vector": [1,2,4], "is_gt_plagiarism": True},
                {"cluster": <0 or 1>, "feature_vector": [5,6,7], "is_gt_plagiarism": False},
                ...
            ]
        }
        '''
        
        d = {
            "doc_name": self.doc_name,
            "features": self.features,
            "time_stamp": self.time_stamp,
            "cluster_strategy": self.cluster_strategy,
            "atom_type": self.atom_type,
            "plag_cluster": self.plag_cluster,
            "non_plag_cluster": self.non_plag_cluster,
            "passages": self.passages
        }
        
        return d