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

class ExtrinsicTester:

    def __init__(self, atom_type, fingerprint_method, file_list):
        self.suspicious_path_start = "/copyCats/pan-plagiarism-corpus-2009/external-detection-corpus/suspicious-documents"
        self.corpus_path_start = "/copyCats/pan-plagiarism-corpus-2009/external-detection-corpus/source-documents/"
        source_dirs = os.listdir(self.corpus_path_start)
        self.source_file_paths = []
        for d in source_dirs:
            p = os.path.join(self.corpus_path_start, d)
            for f in os.listdir(p):
                self.source_file_paths.append(os.path.join(p, f))

        self.base_file_paths = [self.suspicious_path_start + f for f in file_list]
        self.atom_type = atom_type
        self.fingerprint_method = fingerprint_method
        self.file_list = file_list

    def _get_trials(self):
        '''
        For each testing document, split the document into atoms and classify each atom
        as plagiarized or not-plagiarized. Build a list of classifications and a list
        of the ground truths for each atom of each document.
        '''
        classifications = []
        actuals = []
        for f in self.base_file_paths:
            suspicious_document = open(f+'.txt')
            doc = suspicious_document.read()
            atoms = [a for a in doc.splitlines()]
            suspicious_document.close()
            atom_classifications = self.classify_document(atoms, self.atom_type, self.source_file_paths, self.fingerprint_method)
            acts = self._get_actuals(doc, f+'.xml')
            actuals += acts
            classifications += atom_classifications
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


    def classify_document(self, atoms, atom_type, corpus_filenames, fingerprint_method):
        '''
        Split document into atoms and return a list of TRUE/FALSEs 
        corresponding to whether or not each chunk was plagiarized.
        '''
        return [1 if random.random() < .5 else 0 for i in xrange(len(atoms))]


if __name__ == "__main__":
    test_file_listing = file('extrinsic_corpus_partition/extrinsic_training_set_files.txt')
    all_test_files = [f.strip() for f in test_file_listing.readlines()]
    test_file_listing.close()
    first_test_files = all_test_files[0:25]
    print first_test_files

    tester = ExtrinsicTester("paragraph", "anchor", first_test_files)
    tester.plot_ROC_curve()
