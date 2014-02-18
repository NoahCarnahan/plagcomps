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

    def __init__(self, session, suspect_documents, source_documents, atom_type, fingerprint_method, cluster_type):
        self.suspect_documents = suspect_documents
        self.source_documents = source_documents
        self.atom_type = atom_type
        self.fingerprint_method = fingerprint_method
        self.cluster_type = cluster_type
        self.session = session

    def test_intrinsic(self):
        '''
        (features, cluster_type, k, atom_type, docs, corpus='intrinsic', save_roc_figure=True, reduced_docs=None, feature_vector_weights=None, 
        '''

        #suspect_txt_documents = [doc + ".txt"  for doc in self.suspect_documents]

        return get_confidences_actuals(self.session, ["num_chars"], self.cluster_type, 2, self.atom_type, self.suspect_documents, corpus="extrinsic")

    def test_extrinsic(self):
        '''
        def __init__(self, atom_type, fingerprint_method, n, k, hash_len, confidence_method, suspect_file_list, source_file_list, search_method, search_n=1
        '''

        ex_tester = ExtrinsicTester(self.atom_type, self.fingerprint_method, 5, 3, 10000, "containment", self.suspect_documents, self.source_documents, "normal")
        trials, ground_truths = ex_tester.get_trials(self.session)
        confidences = [tup[1] for tup in trials]
        actuals = [tup[0] for tup in ground_truths]
        return confidences, actuals

    def _geo_mean(self, x, y):
        #smoothing constant e
        e = .001
        return [((x[i] + e) * (y[i] + e)) ** 0.5 for i in range(len(x))]

    def _power_mean(self, x, y, power):
        #smoothing constant e
        e = .001
        return [(((x[i] + e) ** power + (y[i] + e) ** power)/2) ** (float(1) / power) for i in range(len(x))]

    def combine(self, method="geo_mean", **kwargs):
        ex_confidences, ex_actuals = self.test_extrinsic()
        in_confidences, in_actuals = self.test_intrinsic()
        
        print "ex confdiences, ex_actuals", ex_confidences, ex_actuals
        print "in_confidences, in_actuals", in_confidences, in_actuals
        assert ex_actuals == in_actuals
        actuals = ex_actuals

        if method == "geo_mean":
           combined_confidences = self._geo_mean(ex_confidences, in_confidences)
        elif method == "power_mean":
           combined_confidences = self._power_mean(ex_confidences, in_confidences, kwargs.get("power", 2))

        metadata = {'n': len(actuals)}
        combined_path, combined_roc_auc = BaseUtility.draw_roc(actuals, combined_confidences, save_figure=False, **metadata)
        in_path, in_roc_auc = BaseUtility.draw_roc(actuals, in_confidences, save_figure=False, **metadata)
        ex_path, ex_roc_auc = BaseUtility.draw_roc(actuals, ex_confidences, save_figure=False, **metadata)

        print "Intrinsic AUC:", in_roc_auc
        print "Extrinsic AUC:", ex_roc_auc
        print "Combined AUC:", combined_roc_auc
    
if __name__ == "__main__":
    session = Session()
    num_files = 50
    source_file_list, suspect_file_list = ExtrinsicUtility().get_training_files(n = num_files, include_txt_extension = True)

    tester = CombinationTester(session, suspect_file_list, source_file_list, "nchars", "anchors", "outlier")

    tester.combine()
    session.close()
