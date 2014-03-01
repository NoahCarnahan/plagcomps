from ..intrinsic.featureextraction import FeatureExtractor
from ..extrinsic.extrinsic_testing import ExtrinsicTester
from intrinsic import get_confidences_actuals
from ..shared.util import ExtrinsicUtility, BaseUtility, IntrinsicUtility

import sqlalchemy
from sqlalchemy.orm import sessionmaker
from ..dbconstants import username, password, dbname
url = "postgresql://%s:%s@%s" % (username, password, dbname)
engine = sqlalchemy.create_engine(url)
Session = sqlalchemy.orm.sessionmaker(bind=engine)

class CombinationTester:

    def __init__(self, session, suspect_documents, source_documents, **kwargs):
        
        self.ex_suspect_documents = []
        self.ex_source_documents = []
        self.in_suspect_documents = []
        self.in_source_documents = []

        # we want intrinsic documents to end with .txt, and extrinsic documents not to.
        for document in suspect_documents:
            if document[-4:] == ".txt":
                no_txt = document[:-4]
            else:
                no_txt = document
            self.ex_suspect_documents.append(no_txt)
            self.in_suspect_documents.append(no_txt + ".txt")
        for document in source_documents:
            if document[-4:] == ".txt":
                no_txt = document[:-4]
            else:
                no_txt = document
            self.ex_source_documents.append(no_txt)
            self.in_source_documents.append(no_txt + ".txt")

        self.session = session
            
        self.atom_type = kwargs.get("atom_type", "nchars")
        self.cluster_type = kwargs.get("cluster_type", "outlier")
        self.fingerprint_method = kwargs.get("fingerprint_method", "full")
        self.similarity_measure = kwargs.get("similarity_measure", "jaccard")
        self.search_method = kwargs.get("search_method", "normal")
        self.search_n = kwargs.get("search_n", "1")
        self.combination_method = kwargs.get("combination_method", "geo_mean")
        self.combination_parameter = kwargs.get("combination_parameter", 2)
        self.fingerprint_n = kwargs.get("fingerprint_n", 5)
        self.fingerprint_k = kwargs.get("fingerprint_k", 0)

    def test_intrinsic(self):
        '''
        (features, cluster_type, k, atom_type, docs, corpus='intrinsic', save_roc_figure=True, reduced_docs=None, feature_vector_weights=None, 
        '''

        features = FeatureExtractor.get_all_feature_function_names(include_nested=True)
        features = [f for f in features if "evolved" not in f ]
        return get_confidences_actuals(self.session, features, self.cluster_type, 2, self.atom_type, self.in_suspect_documents, corpus="extrinsic") #, corpus_type="testing")

    def test_extrinsic(self):
        
        #print "fingerprint_method=", self.fingerprint_method
        ex_tester = ExtrinsicTester(self.atom_type, self.fingerprint_method, self.fingerprint_n, self.fingerprint_k, 10000000, self.similarity_measure, self.ex_suspect_documents, self.ex_source_documents, search_method=self.search_method, search_n=self.search_n)

        #ex_tester = ExtrinsicTester(self.atom_type, self.fingerprint_method, 5, 0, 10000000, "containment", self.suspect_documents, self.source_documents, search_method="normal", search_n=1 )
        #trials, ground_truths =
        conf_tuples, act_tuples, conf_dict, act_dict = ex_tester.get_trials(self.session)
        #a, b, c = ex_tester.evaluate(self.session)
        #print "Extrinisc AUC is", a
        confidences = [tup[1] for tup in conf_tuples]
        actuals = [tup[0] for tup in act_tuples]
        return confidences, actuals

    def _geo_mean(self, x, y):
        #smoothing constant e
        e = .001
        return [((x[i] + e) * (y[i] + e)) ** 0.5 for i in range(len(x))]

    def _arith_mean(self, x, y):
        return [(x[i] + y[i])/2.0 for i in range(len(x))]

    def _power_mean(self, x, y, power = None):
        #smoothing constant e
        e = .001
        if power == None:
            power = self.combination_parameter
        return [(((x[i] + e) ** power + (y[i] + e) ** power)/2) ** (float(1) / power) for i in range(len(x))]

    def _max(self, x, y):
        return [max(x[i], y[i]) for i in range(len(x))]

    def _min(self, x, y):
        return [min(x[i], y[i]) for i in range(len(x))]

    def _sum(self, x, y):
        return [x[i] + y[i] for i in range(len(x))]

    def combine(self):
        ex_confidences, ex_actuals = self.test_extrinsic()
        in_confidences, in_actuals = self.test_intrinsic()
        
        #print "ex confidences, ex_actuals", ex_confidences, ex_actuals
        #print "in_confidences, in_actuals", in_confidences, in_actuals
        assert ex_actuals == in_actuals
        print "ex_actuals == in_actuals"
        actuals = in_actuals

        #if self.combination_method == "geo_mean":
        #   combined_confidences = self._geo_mean(ex_confidences, in_confidences)
        #elif self.combination_method == "power_mean":
        #   combined_confidences = self._power_mean(ex_confidences, in_confidences)

        combos = {}
        combos["geo"] = self._geo_mean(ex_confidences, in_confidences)
        combos["pow-1"] = self._power_mean(ex_confidences, in_confidences, power=-1)
        combos["pow1"] = self._power_mean(ex_confidences, in_confidences, power=1)
        combos["pow2"] = self._power_mean(ex_confidences, in_confidences, power=2)
        combos["pow10"] = self._power_mean(ex_confidences, in_confidences, power=10)
        combos["max"] = self._max(ex_confidences, in_confidences)
        combos["min"] = self._min(ex_confidences, in_confidences)
        combos["arith"] = self._arith_mean(ex_confidences, in_confidences)
        combos["sum"] = self._sum(ex_confidences, in_confidences)

        paths = {}
        metadata = {'n': len(actuals)}
        for key in combos:
                paths[key], combos[key] = BaseUtility.draw_roc(actuals, combos[key], save_figure=True, **metadata)
        #combined_path, combined_roc_auc = BaseUtility.draw_roc(actuals, combined_confidences, save_figure=True, **metadata)
        in_path, in_roc_auc = BaseUtility.draw_roc(actuals, in_confidences, save_figure=True, **metadata)
        ex_path, ex_roc_auc = BaseUtility.draw_roc(actuals, ex_confidences, save_figure=True, **metadata)

        print "Intrinsic AUC:", in_roc_auc, in_path
        print "Extrinsic AUC:", ex_roc_auc, ex_path
        #print "Combined AUC:", combined_roc_auc
        for key in combos:
            print key, "AUC:", combos[key], paths[key]
    
if __name__ == "__main__":
    print "glajsfdldasjflka"
    session = Session()
    num_files = 64

    #args = {"search_method":"two_level_ff", "search_n":4}
    args1 = {"fingerprint_method":"full", "similarity_measure":"jaccard"}
    args2 = {"fingerprint_method":"kth_in_sent", "fingerprint_k":5, "similarity_measure":"jaccard"}

    source_file_list, suspect_file_list = ExtrinsicUtility().get_corpus_files(n = num_files, include_txt_extension = True)

    #tester1 = CombinationTester(session, suspect_file_list, source_file_list, **args1)
    tester2 = CombinationTester(session, suspect_file_list, source_file_list, **args2)

    #tester1.combine()
    tester2.combine()
    
    session.close()
