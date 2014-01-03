# extrinsic_testing.py

import xml
import random
import scipy
import sklearn.metrics
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as pyplot
import os
import time
import nltk
import fingerprint_extraction
from ..shared.util import ExtrinsicUtility

class ExtrinsicTester:

    def __init__(self, atom_type, fingerprint_method, suspect_file_list, source_file_list):
        self.source_file_paths, self.suspect_file_paths = ExtrinsicUtility().get_n_training_files()

        # uncomment these two lines to test on reasonable sized corpus
        # self.source_file_paths = ["sample_corpus/source1", "sample_corpus/source2", "sample_corpus/source3"]
        # self.suspect_file_paths = ["sample_corpus/test1", "sample_corpus/test2", "sample_corpus/test3", "sample_corpus/test4"]

        self.atom_type = atom_type
        self.fingerprint_method = fingerprint_method
        self.suspect_file_list = suspect_file_list
        self.evaluator = fingerprint_extraction.FingerprintEvaluator(self.source_file_paths, fingerprint_method, 3)

    def _get_trials(self):
        '''
        For each testing document, split the document into atoms and classify each atom
        as plagiarized or not-plagiarized. Build a list of classifications and a list
        of the ground truths for each atom of each document.
        '''
        classifications = []
        actuals = []
        for f in self.suspect_file_paths:
            suspicious_document = open(f+'.txt')
            doc = suspicious_document.read()
            atoms = [a for a in doc.splitlines()]
            suspicious_document.close()
            atom_classifications = self.evaluator.classify_document(doc)
            # just take the most similar source document's similarity as the confidence of plagiarism for now.
            similarity = atom_classifications[0][1]

            acts = self._get_actuals(doc, f+'.xml')
            actuals += acts
            classifications += [similarity for i in xrange(len(acts))] # just classify each paragraph in a document as the same similarity
        return classifications, actuals

    def _get_actuals(self, document_text, xml_filepath):
        '''
        Returns a list of the true plagiarism status for each atom of
        the given document.
        '''
        if self.atom_type == "sentence":
            tokenizer = nltk.PunktSentenceTokenizer()
            spans = tokenizer.span_tokenize(document_text)
        elif self.atom_type == "paragraph":
            paragraph_texts = document_text.splitlines()
            s = []
            start_index = 0
            for paragraph in paragraph_texts:
                start = document_text.find(paragraph, start_index)
                s.append((start, start + len(paragraph)))
                start_index = start + len(paragraph)
            spans = s

        plag_spans = self._get_plagiarized_spans(xml_filepath)
        actuals = []
        for s in spans:
            act = 0
            for ps in plag_spans:
                if s[0] >= ps[0] and s[0] < ps[1]:
                    act = 1
                    break
            actuals.append(act)
        return actuals

    def _get_plagiarized_spans(self, xml_file_path):
        '''
        Using the ground truth, return a list of spans representing the passages of the
        text that are plagiarized. Note, this method was plagiarized from Noah's intrinsic
        testing code.
        '''
        spans = []
        tree = xml.etree.ElementTree.parse(xml_file_path)
        for feature in tree.iter("feature"):
            if feature.get("name") == "artificial-plagiarism":
                start = int(feature.get("this_offset"))
                end = start + int(feature.get("this_length"))
                spans.append((start, end))
        return spans

    def plot_ROC_curve(self):
        '''
        Outputs an ROC figure based on our plagiarism classifications and the 
        ground truth of each atom.
        '''
        trials, actuals = self._get_trials()

        # actuals is a list of ground truth classifications for passages
        # trials is a list consisting of 0s and 1s. 1 means we think the atom is plagiarized
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(actuals, trials, pos_label=1)
        roc_auc = sklearn.metrics.auc(fpr, tpr)

        # The following code is from http://scikit-learn.org/stable/auto_examples/plot_roc.html
        pyplot.clf()
        pyplot.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        pyplot.plot([0, 1], [0, 1], 'k--')
        pyplot.xlim([0.0, 1.0])
        pyplot.ylim([0.0, 1.0])
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        pyplot.title('Receiver Operating Characteristic -- Extrinsic w/ '+self.fingerprint_method+' fingerprinting')
        pyplot.legend(loc="lower right")
        
        path = "figures/roc_extrinsic_"+str(time.time())+"_"+self.fingerprint_method+".pdf"
        pyplot.savefig(path)


if __name__ == "__main__":
    util = ExtrinsicUtility()
    num_files = 5

    source_file_list, suspect_file_list = util.get_n_training_files(n=num_files)

    print 'Testing first', num_files, ' suspect files using a corpus of ', len(source_file_list),' source documents:'
    print 'Suspect filenames:', suspect_file_list

    tester = ExtrinsicTester("paragraph", "full", suspect_file_list, source_file_list)
    tester.plot_ROC_curve()

    tester = ExtrinsicTester("paragraph", "kth_in_sent", suspect_file_list, source_file_list)
    tester.plot_ROC_curve()

    tester = ExtrinsicTester("paragraph", "anchor", suspect_file_list, source_file_list)
    tester.plot_ROC_curve()
