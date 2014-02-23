from ..intrinsic.featureextraction import FeatureExtractor
from ..extrinsic.extrinsic_testing import ExtrinsicTester
from intrinsic import get_confidences_actuals
from ..shared.util import ExtrinsicUtility, BaseUtility

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
        self.fingerprint_method = kwargs.get("fingerprint_method", "anchor")
        self.search_method = kwargs.get("search_method", "normal")
        self.search_n = kwargs.get("search_n", "1")
        self.combination_method = kwargs.get("combination_method", "geo_mean")
        self.combination_parameter = kwargs.get("combination_parameter", 2)

    def test_intrinsic(self):
        '''
        (features, cluster_type, k, atom_type, docs, corpus='intrinsic', save_roc_figure=True, reduced_docs=None, feature_vector_weights=None, 
        '''

        features = FeatureExtractor.get_all_feature_function_names(include_nested=True)
        features = [f for f in features if "evolved" not in f ]
        # evolved features are currently not populated
        return get_confidences_actuals(self.session, features, self.cluster_type, 2, self.atom_type, self.in_suspect_documents, corpus="extrinsic")

    def test_extrinsic(self):
        
        #print "fingerprint_method=", self.fingerprint_method
        ex_tester = ExtrinsicTester(self.atom_type, self.fingerprint_method, 5, 0, 10000001, "containment", self.ex_suspect_documents, self.ex_source_documents, search_method=self.search_method, search_n=self.search_n)

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

    def _power_mean(self, x, y):
        #smoothing constant e
        e = .001
        power = self.combination_parameter
        return [(((x[i] + e) ** power + (y[i] + e) ** power)/2) ** (float(1) / power) for i in range(len(x))]

    def _max(self, x, y):
        return [max(x[i], y[i]) for i in range(len(x))]

    def combine(self):
        ex_confidences, ex_actuals = self.test_extrinsic()
        in_confidences, in_actuals = self.test_intrinsic()
        
        #print "ex confdiences, ex_actuals", ex_confidences, ex_actuals
        #print "in_confidences, in_actuals", in_confidences, in_actuals
        assert ex_actuals == in_actuals
        actuals = ex_actuals

        if self.combination_method == "geo_mean":
           combined_confidences = self._geo_mean(ex_confidences, in_confidences)
        elif self.combination_method == "power_mean":
           combined_confidences = self._power_mean(ex_confidences, in_confidences)
        elif self.combination_method == "max":
            combined_confidences = self._max(ex_confidences, in_confidences)

        metadata = {'n': len(actuals)}
        combined_path, combined_roc_auc = BaseUtility.draw_roc(actuals, combined_confidences, save_figure=False, **metadata)
        in_path, in_roc_auc = BaseUtility.draw_roc(actuals, in_confidences, save_figure=False, **metadata)
        ex_path, ex_roc_auc = BaseUtility.draw_roc(actuals, ex_confidences, save_figure=False, **metadata)

        print "Intrinsic AUC:", in_roc_auc
        print "Extrinsic AUC:", ex_roc_auc
        print "Combined AUC:", combined_roc_auc
    
if __name__ == "__main__":
    session = Session()
    num_files = 250

    #args = {"search_method":"two_level_ff", "search_n":4}
    args = {"combination_method": "max", "combination_parameter":10}

    source_file_list, suspect_file_list = ExtrinsicUtility().get_training_files(n = num_files, include_txt_extension = True)

    tester = CombinationTester(session, suspect_file_list, source_file_list) #, **args)

    tester.combine()
    
    session.close()
