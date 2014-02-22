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

    def __init__(self, atom_type, fingerprint_method, n, k, hash_len, confidence_method, suspect_file_list, source_file_list, search_method, search_n=1):
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
        self.search_method = search_method
        self.search_n = search_n


    def get_trials(self, session):
        '''
        For each suspect document, split the document into atoms and classify each atom
        as plagiarized or not-plagiarized. Build a list of classifications and a list
        of the ground truths for each atom of each document.
        '''
        classifications = []
        actuals = []
        classifications_dict = {}
        actuals_dict = {}

        outer_search_level_mid = fingerprintstorage.get_mid(self.fingerprint_method, self.n, self.k, "full", self.hash_len)

        for fi, f in enumerate(self.suspect_file_list, 1):
            print
            doc_name = f.replace(self.suspicious_path_start, "")
            if ".txt" in doc_name:
                doc_name = doc_name.replace(".txt", "")
            if self.search_method == 'two_level_ff':
                print '%d/%d Classifying %s (%s)' % (fi, len(self.suspect_file_list), doc_name, self.search_method)

                acts = ground_truth._query_ground_truth(doc_name, self.base_atom_type, session, self.suspicious_path_start).get_ground_truth(session)
                actuals += acts

                actuals_dict[f] = acts
                doc_classifications = []

                # first, get a list of the most similar full documents to this document
                full_atom_classifications = self.evaluator.classify_passage(doc_name, "full", 0, self.fingerprint_method,
                    self.n, self.k, self.hash_len, "containment", outer_search_level_mid)

                top_docs = full_atom_classifications[:self.search_n]
                dids = [x[0][2] for x in top_docs]
                
                # now, compare all paragraphs in the most similar documents to this paragraph
                for atom_index in xrange(len(acts)):
                    atom_classifications = self.evaluator.classify_passage(doc_name, "paragraph", atom_index, 
                        self.fingerprint_method, self.n, self.k, self.hash_len, self.confidence_method, self.mid, dids=dids)
                    # print 'atom_classifications:', atom_classifications
                    # top_source is a tuple with the form ((source_doc_name, atom_index), confidence, suspect_filename)
                    top_source = atom_classifications[0]
                    source_filename, source_atom_index, did, suspect_filename, atom_index = top_source[0]
                    confidence = top_source[1]

                    classifications.append(top_source)
                    doc_classifications.append(top_source)

                    print 'atom index:', str(atom_index+1) + '/' + str(len(acts))
                    print 'confidence (actual, guess):', acts[atom_index], (confidence, source_filename, source_atom_index)

                classifications_dict[f] = doc_classifications

            elif self.search_method == 'two_level_pf':
                print '%d/%d Classifying %s (%s)' % (fi, len(self.suspect_file_list), doc_name, self.search_method)
                acts = ground_truth._query_ground_truth(doc_name, self.base_atom_type, session, self.suspicious_path_start).get_ground_truth(session)
                actuals += acts

                actuals_dict[f] = acts
                doc_classifications = []

                for atom_index in xrange(len(acts)):
                    # first, find most similar documents to this paragraph
                    full_atom_classifications = self.evaluator.classify_passage(doc_name, "full", atom_index,
                        self.fingerprint_method, self.n, self.k, self.hash_len, "containment",
                        fingerprintstorage.get_mid(self.fingerprint_method, self.n, self.k, "full", self.hash_len),
                        passage_atom_type="paragraph",
                        passage_mid=fingerprintstorage.get_mid(self.fingerprint_method, self.n, self.k, "paragraph", self.hash_len))

                    top_docs = full_atom_classifications[:self.search_n]
                    dids = [x[0][2] for x in top_docs]

                    # don't compare at the paragraph level if no full documents had any similarity to the paragraph
                    if top_docs[0][1] == 0:
                        top_source = top_docs[0]
                    else:
                        # now, compare this paragraph to all paragraphs in <top_docs>
                        atom_classifications = self.evaluator.classify_passage(doc_name, "paragraph", atom_index, 
                            self.fingerprint_method, self.n, self.k, self.hash_len, self.confidence_method, self.mid, dids=dids)
                        # print 'atom_classifications:', atom_classifications
                        # top_source is a tuple with the form ((source_doc_name, atom_index), confidence, suspect_filename)
                        top_source = atom_classifications[0]
                    
                    source_filename, source_atom_index, did, suspect_filename, atom_index = top_source[0]
                    confidence = top_source[1]

                    classifications.append(top_source)
                    doc_classifications.append(top_source)

                    print 'atom index:', str(atom_index+1) + '/' + str(len(acts))
                    print 'confidence (actual, guess):', acts[atom_index], (confidence, source_filename, source_atom_index)

                classifications_dict[f] = doc_classifications
                
            else:
                acts = ground_truth._query_ground_truth(f, self.base_atom_type, session, self.suspicious_path_start).get_ground_truth(session)
                actuals += acts

                actuals_dict[f] = acts
                doc_classifications = []

                print f
                print '%d/%d Classifying %s' % (fi, len(self.suspect_file_list), doc_name)

                for atom_index in xrange(len(acts)):
                    atom_classifications = self.evaluator.classify_passage(doc_name, self.base_atom_type, atom_index, self.fingerprint_method, self.n, self.k, self.hash_len, self.confidence_method, self.mid)
                    # print atom_classifications
                    # top_source is a tuple with the form ((source_doc_name, atom_index), confidence)
                    top_source = atom_classifications[0]
                    source_filename, source_atom_index, did, suspect_filename, atom_index = top_source[0]
                    confidence = top_source[1]

                    classifications.append(top_source)
                    doc_classifications.append(top_source)
                    
                    print 'atom index:', str(atom_index+1) + '/' + str(len(acts))
                    print 'confidence (actual, guess):', acts[atom_index][0], (confidence, source_filename, source_atom_index)

                classifications_dict[f] = doc_classifications

        return classifications, actuals, classifications_dict, actuals_dict


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

    def evaluate(self, session, ignore_high_obfuscation=False, show_false_negpos_info=False):
        '''
        Run our tool with the given parameters and return the area under the roc.
        If a num_files is given, only run on the first num_file suspicious documents,
        otherwise run on all of them.
        '''
        trials, ground_truths, trials_dict, actuals_dict = self.get_trials(session)

        print 'Computing source accuracy...'
        num_plagiarized = 0
        num_called_plagiarized = 0
        num_correctly_identified = 0
        false_negatives = []
        false_positives = []

        for trial, ground_truth in zip(trials, ground_truths):
            guessed_doc_name = trial[0][0]
            if ground_truth[0] == 1:
                num_plagiarized += 1
                if guessed_doc_name in ground_truth[1]:
                    num_correctly_identified += 1
                else:
                    false_negatives.append([trial, ground_truth])
                if guessed_doc_name != 'dummy':
                    num_called_plagiarized += 1
            else:
                if guessed_doc_name != 'dummy':
                    false_positives.append([trial, ground_truth])

        source_accuracy = float(num_correctly_identified) / num_called_plagiarized
        true_source_accuracy = float(num_correctly_identified) / num_plagiarized
        
        print num_plagiarized, num_correctly_identified, source_accuracy
        
        if show_false_negpos_info:
            self.display_false_negative_info(false_negatives)
            self.display_false_positive_info(false_positives)

        confidences = []
        actuals = []
        for trial, ground_truth in zip(trials, ground_truths):
            if not ignore_high_obfuscation or 'high' not in ground_truth[4]:
                confidences.append(trial[1])
                actuals.append(ground_truth[0])

        # UNCOMMENT NEXT LINE TO GET FALSEPOSITIVES AND FALSENEGATIVES
        # self.analyze_fpr_fnr(trials_dict, actuals_dict, 0.50)
        roc_auc, path = self.plot_ROC_curve(confidences, actuals)
        return roc_auc, source_accuracy, true_source_accuracy


    def display_false_positive_info(self, false_positives):
        print
        print 'False Positives'
        print '================='
        for x in false_positives:
            print 'FALSE POSITIVE'
            print x
            suspect_name = re.sub(r'/part\d*/', '', x[0][0][3]) + '.txt'
            suspect_path = ExtrinsicUtility().get_suspect_abs_path(suspect_name)
            if x[0][0][0] != 'dummy':
                guessed_name = re.sub(r'/part\d*/', '', x[0][0][0]) + '.txt'
                guessed_path = ExtrinsicUtility().get_src_abs_path(guessed_name)
            else:
                guessed_name = 'dummy'
                guessed_path = 'no-path'
            
            f = open(suspect_path, 'r')
            text = f.read()
            f.close()

            f = open(guessed_path, 'r')
            guessed_text = f.read()
            f.close()

            atom_number = x[0][0][4]
            guessed_atom_number = x[0][0][1]
            span = tokenize(text, self.base_atom_type, n=5000)[atom_number]
            guessed_span = tokenize(guessed_text, self.base_atom_type, n=5000)[guessed_atom_number]
            print 'suspect atom_number:', atom_number
            print 'suspect span:', span
            print 'actual text from suspect document that WE were wrong about:'
            print
            print text[span[0] : span[1]]
            print 
            print '***********************************'
            print 'guessed source atom_number:', guessed_atom_number
            print 'guessed source span:', guessed_span
            print 'actual text from guessed source:'
            print 
            print guessed_text[guessed_span[0] : guessed_span[1]]
            print 
            print '=========================================='
            print


    def display_false_negative_info(self, false_negatives):
        '''
        Print info about the false positives from the given list of <false_negatives>.
        Each entry in <false_negatives> is a tuple of the ugly form that nobody should
        ever try and use... Sorry guys.
        '''
        print 'False Negatives'
        print '================='
        for x in false_negatives:
            print 'FALSE NEGATIVE'
            print x
            suspect_name = re.sub(r'/part\d*/', '', x[0][0][3]) + '.txt'
            suspect_path = ExtrinsicUtility().get_suspect_abs_path(suspect_name)
            if x[0][0][0] != 'dummy':
                guessed_name = re.sub(r'/part\d*/', '', x[0][0][0]) + '.txt'
                guessed_path = ExtrinsicUtility().get_src_abs_path(guessed_name)
            else:
                guessed_name = 'dummy'
                guessed_path = 'no-path'

            source_name = re.sub(r'/part\d*/', '', x[1][1][0]) + '.txt'
            source_path = ExtrinsicUtility().get_src_abs_path(source_name)
            source_span = x[1][3][0]
            obfuscation = x[1][4][0]
            
            print 'actual source:', source_name
            print 'obfuscation level:', obfuscation
            span = x[1][2][0]
            print 'actual span:', span
            print 'actual source span:', source_span
            print 'actual text from suspect document that WE were wrong about:', span
            print
            f = open(suspect_path, 'r')
            text = f.read()
            f.close()
            print text[span[0] : span[1]]
            print
            print '*******************************'
            print 'actual text from source document that WE were wrong about:', source_span
            f = open(source_path, 'r')
            text = f.read()
            f.close()
            print text[source_span[0] : source_span[1]]
            print
            print
            print
            print '=========================================='


    def analyze_fpr_fnr(self, trials, actuals, threshold):
        '''
        Takes as arguements two dictionaries of format:

        trials = {doc_name: [list of atoms with reported confidence ... an element looks like: (0, 'dummy', 0)]}
        actuals = {doc_name: [list of 0's and 1's where index specifies atom index]}

        Creates two files in plagcomps/extrinsic/FPR_FNR/... that reports the falsePositives and falseNegatives
        based on a given threshold.
        '''
        falsePositives = {}
        falseNegatives = {}

        if threshold > 1.0 or threshold < 0.0:
            print "INVALID THREHOLD VALUE. THRESHOLD MUST BE BETWEEN 0.0 and 1.0"

        # Gets atom indexes for falsePositives and falseNegatives and maps them to the appropriate document in the 
        # appropriate dictionary
        for key in trials.keys():
            for i in xrange(len(trials[key])):
                confidence = trials[key][i]
                actual = actuals[key][i][0]
                if confidence > threshold:
                    if actual == 0:
                        try:
                            falsePositives[key].append(i)
                        except:
                            falsePositives[key] = [i]
                else:
                    if actual == 1:
                        try:
                            falseNegatives[key].append(i)
                        except:
                            falseNegatives[key] = [i]


        if not len(falsePositives):
            print "Found no False Positives!"
        else:
            print "Beginning to print False Positives to file."
            filename = "plagcomps/extrinsic/FPR_FNR/falsePositives" + str(time.time()) + "-" + self.fingerprint_method + ".txt"
            fileFPR = open(filename, "w")
            
            for f in falsePositives.keys():
                file = open(f + ".txt")
                text = file.read()
                file.close()
                paragraph_spans = tokenize(text, self.base_atom_type)

                for index in falsePositives[f]:
                    paragraph = text[paragraph_spans[index][0]:paragraph_spans[index][1]]
                    fileFPR.write("Document Name: " + f + "\n")
                    fileFPR.write("Paragraph Index: " + str(index) + "\n")
                    fileFPR.write("Detected Confidence: " + str(trials[f][index]) + "\n")
                    fileFPR.write("Fingerprint Technique: " + self.fingerprint_method + str(self.n) + "\n")
                    fileFPR.write("\n")
                    fileFPR.write(paragraph + "\n\n")
                    fileFPR.write("--"*20 + "\n\n")

            print "Output for falsePositives is in:" + filename
            fileFPR.close()

        if not falseNegatives:
            print "Found no False Negatives!"
        else:
            print "Beginning to print False Negatives to file."
            filename = "plagcomps/extrinsic/FPR_FNR/falseNegatives" + str(time.time()) + "-" + self.fingerprint_method + ".txt"
            fileFNR = open(filename, "w")

            for f in falseNegatives.keys():
                file = open(f + ".txt")
                text = file.read()
                file.close()
                paragraph_spans = tokenize(text, self.base_atom_type)

                for index in falseNegatives[f]: 
                    paragraph = text[paragraph_spans[index][0]:paragraph_spans[index][1]]
                    fileFNR.write("Document Name: " + f + "\n")
                    fileFNR.write("Paragraph Index: " + str(index) + "\n")
                    fileFNR.write("Detected Confidence: " + str(trials[f][index]) + "\n")
                    fileFNR.write("Fingerprint Technique: " + self.fingerprint_method + str(self.n) + "\n")
                    fileFNR.write("\n")
                    fileFNR.write(paragraph + "\n\n")
                    fileFNR.write("--"*20 + "\n\n")

            print "Output for falseNegatives is in:" + filename
            fileFNR.close()


def test(method, n, k, atom_type, hash_size, confidence_method, num_files="all", search_method='normal', search_n=5, save_to_db=True, ignore_high_obfuscation=False, show_false_negpos_info=False):
    session = Session()
    
    # Get the list of suspect files to test on
    source_file_list, suspect_file_list = ExtrinsicUtility().get_training_files(n = num_files, include_txt_extension = False)
    
    # Confirm that these suspects and enough source documents have been populated
    num_suspect_documents = len(suspect_file_list)
    num_source_documents = len(source_file_list)
    
    mid = fingerprintstorage.get_mid(method, n, k, atom_type, hash_size)
    num_populated_suspects = fingerprintstorage.get_number_suspects(mid)
    num_populated_sources = fingerprintstorage.get_number_sources(mid)
    
    if num_populated_suspects < num_suspect_documents or num_populated_sources < num_source_documents:
        raise ValueError("Not all of the documents used in this test have been populated (only "+str(num_populated_sources)+" sources, "+str(num_populated_suspects)+" suspects have been populated). Populate them first with fingerprintstorage.")
    
    # If the search method is two level, we need to check that additional things are in the database
    if search_method == "two_level_ff" or search_method == "two_level_pf":
        full_mid = fingerprintstorage.get_mid(method, n, k, "full", hash_size)
        para_mid = fingerprintstorage.get_mid(method, n, k, "paragraph", hash_size)
        
        num_populated_full_suspects = fingerprintstorage.get_number_suspects(full_mid)
        num_populated_para_suspects = fingerprintstorage.get_number_suspects(para_mid)
        
        num_populated_full_sources = fingerprintstorage.get_number_sources(full_mid)
        num_populated_para_sources = fingerprintstorage.get_number_sources(para_mid)
        
        num_populated_sources = num_populated_full_sources
        num_populated_suspects = num_populated_full_suspects
        
        if num_populated_full_suspects < num_suspect_documents or num_populated_para_suspects < num_suspect_documents \
            or num_populated_full_sources < num_source_documents or num_populated_para_sources < num_source_documents \
            or num_populated_para_sources < num_populated_full_sources \
            or num_populated_para_suspects < num_populated_full_suspects:
            raise ValueError("Not all of the documents used in this test have been populated (only "+str(num_populated_sources)+" sources, "+str(num_populated_suspects)+" suspects have been populated). Populate them first with fingerprintstorage.")
    
    
    print suspect_file_list    
    print "Testing first", suspect_file_list, "suspect files against", num_populated_sources, "source documents."
    
    tester = ExtrinsicTester(atom_type, method, n, k, hash_size, confidence_method, suspect_file_list, source_file_list, search_method, search_n)

    roc_auc, source_accuracy, true_source_accuracy = tester.evaluate(session, ignore_high_obfuscation, show_false_negpos_info)

    # Save the result
    if save_to_db:
        with psycopg2.connect(user = username, password = password, database = dbname.split("/")[1], host="localhost", port = 5432) as conn:
            conn.autocommit = True    
            with conn.cursor() as cur:
                query = "INSERT INTO extrinsic_results (method_name, n, k, atom_type, hash_size, simmilarity_method, suspect_files, source_files, auc, true_source_accuracy, source_accuracy, search_method, search_n, ignore_high_obfuscation) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"
                args = (method, n, k, atom_type, hash_size, confidence_method, num_files, num_populated_sources, roc_auc, true_source_accuracy, source_accuracy, search_method, search_n, ignore_high_obfuscation)
                cur.execute(query, args)
    
    print 'ROC auc:', roc_auc
    print 'Source Accuracy:', source_accuracy
    print 'True Source Accuracy:', true_source_accuracy

        
if __name__ == "__main__":
    test("full", 5, 0, "paragraph", 10000000, "jaccard", num_files=20, search_method='normal', search_n=1, 
        save_to_db=True, ignore_high_obfuscation=False, show_false_negpos_info=False)
