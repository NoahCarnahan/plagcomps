# extrinsic_testing.py

import scipy
import sklearn
import sklearn.metrics
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as pyplot
import os
import time
import fingerprint_extraction
import fingerprintstorage
import ground_truth
from ..shared.util import ExtrinsicUtility
from ..tokenization import *

class ExtrinsicTester:

    def __init__(self, atom_type, fingerprint_method, n, k, confidence_method, suspect_file_list, source_file_list):
        self.suspicious_path_start = ExtrinsicUtility.CORPUS_SUSPECT_LOC
        self.corpus_path_start = ExtrinsicUtility.CORPUS_SRC_LOC
        source_dirs = os.listdir(self.corpus_path_start)
            
        self.atom_type = atom_type
        self.fingerprint_method = fingerprint_method
        self.n = n
        self.k = k
        self.confidence_method = confidence_method
        self.suspect_file_list = suspect_file_list
        self.source_file_list = source_file_list
        self.evaluator = fingerprint_extraction.FingerprintEvaluator(source_file_list, fingerprint_method, self.n, self.k)

    def _get_trials(self, session):
        '''
        For each suspect document, split the document into atoms and classify each atom
        as plagiarized or not-plagiarized. Build a list of classifications and a list
        of the ground truths for each atom of each document.
        '''
        classifications = []
        classificationsDocNames = {}
        actualConfidences = []
        actualDocNames = {}
        for f in self.suspect_file_list:
            doc_classifications = []
            suspicious_document = open(f + '.txt')
            doc = suspicious_document.read()
            suspicious_document.close()
            
            doc_name = f.replace(self.suspicious_path_start, "")

            acts = ground_truth._query_ground_truth(f, self.atom_type, session, self.suspicious_path_start).get_ground_truth(session)
            actualConfidences += acts
            actualDocNames[f] = actualConfidences

            print f
            print 'Classifying', doc_name
            
            for atom_index in xrange(len(acts)):    
                atom_classifications = self.evaluator.classify_document(doc_name, self.atom_type, atom_index, self.fingerprint_method, self.n, self.k, self.confidence_method, session)
                #print atom_classifications
                # just take the most similar source document's similarity as the confidence of plagiarism for now.
                classifications.append(atom_classifications[0][1])
                
                print 'atom index:', str(atom_index+1) + '/' + str(len(acts))
                print 'confidence (actual, guess):', acts[atom_index], atom_classifications[0][1]

        return classifications, actuals


    def plot_ROC_curve(self, sess):
        '''
        Outputs an ROC figure based on our plagiarism classifications and the 
        ground truth of each atom.
        '''
        trials, Treference, actuals, Areference = self._get_trials()
    
        self.analyze_fpr_fnr(Treference, Areference)

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
        pyplot.title('Extrinsic: method=' + self.fingerprint_method+' n=' + str(self.n) + ' confidence_method=' + self.confidence_method)
        pyplot.legend(loc="lower right")
        
        path = os.path.join(os.path.dirname(__file__), "../figures/roc_extrinsic_"+str(time.time())+"_"+self.fingerprint_method+".pdf")
        pyplot.savefig(path)

        return roc_auc, path

def evaluate(method, n, k, atom_type, confidence_method, num_files="all"):
    '''
    Run our tool with the given parameters and return the area under the roc.
    If a num_files is given, only run on the first num_file suspicious documents,
    otherwise run on all of them.
    '''
    
    session = fingerprintstorage.Session()
    source_file_list, suspect_file_list = ExtrinsicUtility().get_training_files(n = num_files, include_txt_extension = False)
    #print suspect_file_list
    #print "Populating first", len(suspect_file_list), "suspect files and first", len(source_file_list), "source documents."
    #fingerprintstorage.populate_db(source_file_list+suspect_file_list, method, n, k, atom_type)
    
    print "Testing first", len(suspect_file_list), "suspect files and first", len(source_file_list), "source documents."
   

    tester = ExtrinsicTester(atom_type, method, n, k, confidence_method, suspect_file_list, source_file_list)
    print tester.plot_ROC_curve(session)


    def analyze_fpr_fnr(self, trials, actuals):

        falsePositives = {}
        falseNegatives = {}

        for key in trials.keys():
            for i in xrange(len(trials[key])):
                if trials[key][i] >=  0.50:
                    if actuals[key][i] == 0:
                        try:
                            falsePositives[key].append(i)
                        except:
                            falsePositives[key] = [i]
                else:
                    if actuals[key][i] == 1:
                        try:
                            falseNegatives[key].append(i)
                        except:
                            falseNegatives[key] = [i]

        fileFPR = open("plagcomps/extrinsic/FPR_FNR/falsePositives" + str(time.time()) + "-" + self.fingerprint_method + ".txt", "w")
        
        for f in falsePositives.keys():
            file = open(f + ".txt")
            text = file.read()
            file.close()
            paragraph_spans = tokenize(text, self.atom_type)

            print "These are the spans: " , paragraph_spans
        
            print falsePositives[f]    

            for index in falsePositives[f]:
                print "Index Number: ", index
                print "On span start: ", paragraph_spans[index][0]
                print "On span end: ", paragraph_spans[index][1]
                paragraph = text[paragraph_spans[index][0]:paragraph_spans[index][1]]
                fileFPR.write("Document Name: " + f + "\n")
                fileFPR.write("Paragraph Index: " + str(index) + "\n")
                fileFPR.write("Detected Confidence: " + str(trials[f][index]) + "\n")
                fileFPR.write("Fingerprint Technique: " + self.fingerprint_method + str(self.n) + "\n")
                fileFPR.write("\n")
                fileFPR.write(paragraph + "\n")
                fileFPR.write("--"*20 + "\n")

        fileFPR.close()
        fileFNR = open("plagcomps/extrinsic/FPR_FNR/falseNegatives" + str(time.time()) + "-" + self.fingerprint_method + ".txt", "w")

        for f in falseNegatives.keys():
            file = open(f + ".txt")
            text = file.read()
            file.close()
            paragraph_spans = tokenize(text, self.atom_type)

            for index in falseNegatives.keys():
                paragraph = text[paragraph_spans[index][0]:paragraph_spans[index][1]]
                fileFNR.write("Document Name: " + f + "\n")
                fileFNR.write("Paragraph Index: " + str(index) + "\n")
                fileFNR.write("Detected Confidence: " + str(trials[f][index]) + "\n")
                fileFNR.write("Fingerprint Technique: " + self.fingerprint_method + "-" + str(self.n) + "\n")
                fileFNR.write("\n")
                fileFNR.write(paragraph + "\n")
                fileFNR.write("--"*20 + "\n")

        fileFNR.close()
        
    
if __name__ == "__main__":
    evaluate("kth_in_sent", 5, 5, "paragraph", "jaccard", num_files=15)
