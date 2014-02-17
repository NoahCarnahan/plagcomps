# extrinsic_testing.py
import re
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

from ..dbconstants import username, password, dbname
import psycopg2
import sqlalchemy

url = "postgresql://%s:%s@%s" % (username, password, dbname)
engine = sqlalchemy.create_engine(url)
Session = sqlalchemy.orm.sessionmaker(bind=engine)

class ExtrinsicTester:

    def __init__(self, atom_type, fingerprint_method, n, k, hash_len, confidence_method, suspect_file_list, source_file_list, log_search, log_search_n):
        self.suspicious_path_start = ExtrinsicUtility.CORPUS_SUSPECT_LOC
        self.corpus_path_start = ExtrinsicUtility.CORPUS_SRC_LOC
        source_dirs = os.listdir(self.corpus_path_start)
        self.mid = fingerprintstorage.get_mid(fingerprint_method, n, k, atom_type, hash_len)
        self.base_atom_type = atom_type
        self.fingerprint_method = fingerprint_method
        self.n = n
        self.k = k
        self.hash_len = hash_len
        self.confidence_method = confidence_method
        self.suspect_file_list = suspect_file_list
        self.source_file_list = source_file_list
        self.evaluator = fingerprint_extraction.FingerprintEvaluator(source_file_list, fingerprint_method, self.n, self.k)
        self.log_search = log_search
        self.log_search_n = log_search_n


    def get_trials(self, session):
        '''
        For each suspect document, split the document into atoms and classify each atom
        as plagiarized or not-plagiarized. Build a list of classifications and a list
        of the ground truths for each atom of each document.
        '''
        classifications = []
        actuals = []
        
        for fi, f in enumerate(self.suspect_file_list, 1):
            print
            if self.log_search:
                doc_name = f.replace(self.suspicious_path_start, "")
                print '%d/%d Classifying %s (log search)' % (fi, len(self.suspect_file_list), doc_name)

                acts = ground_truth._query_ground_truth(f, "paragraph", session, self.suspicious_path_start).get_ground_truth(session)
                actuals += acts

                # first, get a list of the most similar full documents to this document
                atom_classifications = self.evaluator.classify_passage(doc_name, "full", 0, self.fingerprint_method, 
                    self.n, self.k, self.hash_len, "containment", 
                    fingerprintstorage.get_mid(self.fingerprint_method, self.n, self.k, "full", self.hash_len))

                top_docs = atom_classifications[:self.log_search_n]
                dids = [x[0][2] for x in top_docs]
                
                # now, compare all paragraphs in the most similar documents to this paragraph
                for atom_index in xrange(len(acts)):
                    atom_classifications = self.evaluator.classify_passage(doc_name, "paragraph", atom_index, 
                        self.fingerprint_method, self.n, self.k, self.hash_len, self.confidence_method, self.mid, dids=dids)
                    # print 'atom_classifications:', atom_classifications
                    # top_source is a tuple with the form ((source_doc_name, atom_index), confidence, suspect_filename)
                    top_source = atom_classifications[0]
                    source_filename, source_atom_index, did, suspect_filename = top_source[0]
                    confidence = top_source[1]

                    classifications.append(top_source)

                    print 'atom index:', str(atom_index+1) + '/' + str(len(acts))
                    print 'confidence (actual, guess):', acts[atom_index], (confidence, source_filename, source_atom_index)
            else:
                doc_name = f.replace(self.suspicious_path_start, "")

                acts = ground_truth._query_ground_truth(f, self.base_atom_type, session, self.suspicious_path_start).get_ground_truth(session)
                actuals += acts

                print f
                print '%d/%d Classifying %s' % (fi, len(self.suspect_file_list), doc_name)
                
                for atom_index in xrange(len(acts)):
                    atom_classifications = self.evaluator.classify_passage(doc_name, self.base_atom_type, atom_index, self.fingerprint_method, self.n, self.k, self.hash_len, self.confidence_method, self.mid)
                    # print atom_classifications
                    # top_source is a tuple with the form ((source_doc_name, atom_index), confidence)
                    top_source = atom_classifications[0]
                    source_filename, source_atom_index, did, suspect_filename = top_source[0]
                    confidence = top_source[1]

                    classifications.append(top_source)
                    
                    print 'atom index:', str(atom_index+1) + '/' + str(len(acts))
                    print 'confidence (actual, guess):', acts[atom_index][0], (confidence, source_filename, source_atom_index)

        return classifications, actuals


    def plot_ROC_curve(self, confidences, actuals):
        '''
        Outputs an ROC figure based on our plagiarism classifications and the 
        ground truth of each atom.
        '''
        # actuals is a list of ground truth classifications for passages

        # trials is a list consisting of 0s and 1s. 1 means we think the atom is plagiarized
        # print "actuals:"
        # print actuals
        # print "confidences:"
        # print confidences
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(actuals, confidences, pos_label=1)
        print "fpr:"
        print fpr
        print "tpr:"
        print tpr
        print "thresholds:"
        print thresholds
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


    def evaluate(self, session):
        '''
        Run our tool with the given parameters and return the area under the roc.
        If a num_files is given, only run on the first num_file suspicious documents,
        otherwise run on all of them.
        '''
        trials, ground_truths = self.get_trials(session)

        print 'Computing source accuracy...'
        num_plagiarized = 0
        num_called_plagiarized = 0
        num_correctly_identified = 0
        incorrectly_identified = []

        for trial, ground_truth in zip(trials, ground_truths):
            guessed_doc_name = trial[0][0]
            if ground_truth[0] == 1:
                num_plagiarized += 1
                if guessed_doc_name in ground_truth[1]:
                    num_correctly_identified += 1
                else:
                    incorrectly_identified.append([trial, ground_truth])
                if guessed_doc_name != 'dummy':
                    num_called_plagiarized += 1

        source_accuracy = float(num_correctly_identified) / num_called_plagiarized
        true_source_accuracy = float(num_correctly_identified) / num_plagiarized
        # uncomment the following lines to print the incorrectly identified passages
        print num_plagiarized, num_correctly_identified, source_accuracy
        # print
        # print 'Incorrect Guesses'
        # print '================='
        # for x in incorrectly_identified:
        #     susppect_name = re.sub(r'/part\d*/', '', x[1][1][0]) + '.txt'
        #     susupect_path = ExtrinsicUtility().get_src_abs_path(source_name)
        #     print source_path
        #     span = x[1][2][0]
        #     print span
        #     f = open(source_path, 'r')
        #     text = f.read()
        #     f.close()
        #     print x
        #     print 'text[%d : %d]:' % (span[0], span[1])
        #     print text[span[0] : span[1]]
        #     print

        # build list of plain confidences and actuals values
        confidences = [x[1] for x in trials]
        actuals = [x[0] for x in ground_truths]

        roc_auc, path = self.plot_ROC_curve(confidences, actuals)
        return roc_auc, source_accuracy, true_source_accuracy


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
        paragraph_spans = tokenize(text, self.base_atom_type)

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
        paragraph_spans = tokenize(text, self.base_atom_type)

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


def test(method, n, k, atom_type, hash_size, confidence_method, num_files="all", log_search=True, log_search_n=5):
    session = Session()
        
    source_file_list, suspect_file_list = ExtrinsicUtility().get_training_files(n = num_files, include_txt_extension = False)
    # TODO: get rid of this...
    # suspect_file_list = ['/copyCats/pan-plagiarism-corpus-2009/external-detection-corpus/suspicious-documents/part5/suspicious-document09634']
    print suspect_file_list    
    print "Testing first", len(suspect_file_list), "suspect files against how ever many source documents have been populated."
       
    # # Save the reult
    # with psycopg2.connect(user = username, password = password, database = dbname.split("/")[1], host="localhost", port = 5432) as conn:
    #     conn.autocommit = True    
    #     with conn.cursor() as cur:
    #         num_sources = fingerprintstorage.get_number_sources(fingerprintstorage.get_mid(method, n, k, atom_type, hash_size))
    #         query = "INSERT INTO extrinsic_results (method_name, n, k, atom_type, hash_size, simmilarity_method, suspect_files, source_files, auc, source_accuracy) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"
    #         args = (method, n, k, atom_type, hash_size, confidence_method, num_files, num_sources, auc, source_accuracy)
    #         cur.execute(query, args)
    
    tester = ExtrinsicTester(atom_type, method, n, k, hash_size, confidence_method, suspect_file_list, source_file_list, log_search, log_search_n)
    roc_auc, source_accuracy, true_source_accuracy = tester.evaluate(session)
    print 'ROC auc:', roc_auc
    print 'Source Accuracy:', source_accuracy
    print 'True Source Accuracy:', true_source_accuracy

        
if __name__ == "__main__":
    test("anchor", 5, 0, "paragraph", 10000000, "containment", num_files=20, log_search=False, log_search_n=1)
    #evaluate("kth_in_sent", 5, 3, "full", 10000000, "jaccard", num_files=10)
