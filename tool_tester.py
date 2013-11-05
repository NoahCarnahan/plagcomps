from controller import Controller
from trial import Trial
from ROCTrial import ROCTrial

import xml.etree.ElementTree as ET
from collections import Counter
import cProfile
import numpy
import scipy
import time
import pickle
import os
import trial_store
import sklearn.metrics

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as pyplot

DEBUG = True



#NOTES:
# The metric which I am calling the atom-to-atom metric compares every atom to every
# other atom and returns:
# |{pair (x, y) : x and y in same ground truth group AND x and y in same predicted groups}| / (total number of pairs)

class ToolTester:
    
    def __init__(self, atom_type, feature_list, file_list):
        '''
        <file_list> should be relative to suspicious-documents/
        '''
        # atom_type = "word", "sentence", or "paragraph"
        # atom_type specifies what atoms the clusterer should use
    
        path_start = "/copyCats/pan-plagiarism-corpus-2009/intrinsic-detection-corpus/suspicious-documents/"
        self.base_file_paths = [path_start + f for f in file_list]
        self.atom_type = atom_type
        self.feature_list = feature_list
        self.file_list = file_list
        
    def get_plagiarized_spans(self, xml_file_path):
        '''
        Using the ground truth, return a list of spans representing the passages of the
        text that are plagiarized. 
        '''
        spans = []
        tree = ET.parse(xml_file_path)
        for feature in tree.iter("feature"):
            if feature.get("name") == "artificial-plagiarism": # are there others?
                start = int(feature.get("this_offset"))
                end = start + int(feature.get("this_length"))
                spans.append((start, end))
            
        return spans
    
    def _get_smallest_cluster(self, passages):
        '''
        Return the number of the smallest of the two clusters.
        '''
        
        cluster_nums = [p.cluster_num for p in passages]
        smallest_cluster = Counter(cluster_nums).most_common()[-1][0]
        
        return smallest_cluster
    
    def _is_plagiarized(self, p, plagiarized_spans):
        '''
        p is a passage. Return True if the first character of the passage is in a plagiarized span.
        '''
        # TODO: Consider other ways to judge if an atom is plagiarized or not. 
        # For example, look to see if the WHOLE atom in a plagiarized segment (?)
        for s in plagiarized_spans:
            if s[0] <= p.start_word_index < s[1]:
                return True
        return False

    def _is_in_smallest_cluster(self, p):
        '''
        p is a passage
        Returns True if the given passage p is in the smaller of the two clusters.
        '''
        return p.cluster_num == self._get_smallest_cluster()
    
    def test_one_file(self, file_base, features = 'all'):
        '''
        <features> should either be a list of features to test, or 'all'
        if we want to use all of <self.feature_list>

        Returns a Trial object which holds the document's name, features used,
        number of correctly classified passages, and total number of passages
        Assumes that the smaller of the two predicted clusters is the cluster of plagiarized passages
        '''
        #TODO: Name this metric and rename this function appropriately
            
        c = Controller(file_base + '.txt')
        if features == 'all':
            features = self.feature_list

        passages = c.get_passages(self.atom_type, features, "kmeans", 2)
        plagiarzed_spans = self.get_plagiarized_spans(file_base + '.xml')
        smallest_cluster = self._get_smallest_cluster(passages)

        if DEBUG:
            print "Total passages:", len(passages)
            amount_done = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            amount_done_i = 0

        tp = 0
        fp = 0
        tn = 0
        fn = 0
        
        for p in passages:
            if DEBUG and float(tp+fp+tn+fn) / len(passages) > amount_done[amount_done_i] / 100.0:
                print str(amount_done[amount_done_i])+"% done..."
                amount_done_i += 1

            actually_plagiarized = self._is_plagiarized(p, plagiarzed_spans)
            
            if (actually_plagiarized and p.cluster_num == smallest_cluster):
                tp += 1
            elif (not actually_plagiarized and p.cluster_num != smallest_cluster):
                tn += 1
            elif (actually_plagiarized and p.cluster_num != smallest_cluster):
                fn += 1
            elif (not actually_plagiarized and p.cluster_num == smallest_cluster):
                fp += 1
        
        trial = Trial(file_base, features, tp, fp, tn, fn)
        return trial

    def test_all_files(self, features = 'all'):
        '''
        Using 
        '''
        all_trials = []
        for base_path in self.base_file_paths:
            print 'Testing', base_path
            result_trial = self.test_one_file(base_path, features)
            all_trials.append(result_trial)

        return all_trials

    def _is_above_plagiarism_threshold(self, passage, threshold, plag_centroid, notplag_centroid):
        if threshold == None:
            return p.cluster_num == smallest_cluster
        return self._get_certainty(passage, plag_centroid, notplag_centroid) >= threshold
    
    def _get_certainty(self, passage, plag_centroid, notplag_centroid):
        #print "plagcentroid type:", type(plag_centroid)
        #print "plagcentroid:", plag_centroid
        #print "p.features type:", type(passage.features)
        #print "p.features:", passage.features
        #print "p.features.values():", passage.features.values()
        
        d_from_plag_c = float(scipy.spatial.distance.pdist(numpy.matrix([plag_centroid, passage.features.values()]), "euclidean")[0])
        d_from_notplag_c = float(scipy.spatial.distance.pdist(numpy.matrix([notplag_centroid, passage.features.values()]), "euclidean")[0])
        cert = 1 - d_from_plag_c / (d_from_plag_c + d_from_notplag_c)
        #print cert
        return cert
        #  1 - [ (distance from plag_centroid) / (distance from plag_centroid + distance from notplag_centroid) ]

    def test_one_feature_set(self, controller_obj, feature_set, file_base, threshold = None):
        all_trials = []

        
        passages = controller_obj.get_passages(self.atom_type, feature_set, "kmeans", 2)
        print len(passages), 'total passages'
        plagiarzed_spans = self.get_plagiarized_spans(file_base + '.xml')
        smallest_cluster = self._get_smallest_cluster(passages)
        
        
        centroids = controller_obj.get_centroids()
        plag_centroid = centroids[smallest_cluster]
        notplag_centroid = centroids[1 if smallest_cluster == 0 else 0] #assumes k=2!

        tp = 0
        fp = 0
        tn = 0
        fn = 0
        
        for p in passages:
            actually_plagiarized = self._is_plagiarized(p, plagiarzed_spans)
            
            if (actually_plagiarized and self._is_above_plagiarism_threshold(p, threshold, plag_centroid, notplag_centroid)):
                tp += 1
            elif (not actually_plagiarized and not self._is_above_plagiarism_threshold(p, threshold, plag_centroid, notplag_centroid)):
                tn += 1
            elif (actually_plagiarized and not self._is_above_plagiarism_threshold(p, threshold, plag_centroid, notplag_centroid)):
                fn += 1
            elif (not actually_plagiarized and self._is_above_plagiarism_threshold(p, threshold, plag_centroid, notplag_centroid)):
                fp += 1
        
        trial = Trial(file_base, feature_set, tp, fp, tn, fn)

        return trial

    def test_features_individually(self):
        '''
        For each feature in <self.feature_list>, use ONLY that feature 
        and test every file in <self.file_list>

        Output the results of each trial using <self.write_out_trials>
        '''
        # feature => trial objects for that (those) feature(s)
        all_trials = []
        for single_file in self.base_file_paths:
            print 'Working on', single_file
            c = Controller(single_file + '.txt')
            for single_feature in self.feature_list:
                print 'Using this feature:', single_feature
                all_trials.append(self.test_one_feature_set(c, [single_feature], single_file))

        self.write_out_trials(all_trials)

    def test_full_feature_set(self, threshold = None):
        '''
       
        '''
        # feature => trial objects for that (those) feature(s)
        all_trials = []
        for single_file in self.base_file_paths:
            print 'Working on', single_file
            c = Controller(single_file + '.txt')
            
            all_trials.append(self.test_one_feature_set(c, self.feature_list, single_file, threshold))

        self.write_out_trials(all_trials, 'results/noah_test_'+str(threshold)+'.csv')


    # get rid of
    def write_out_trials(self, trials, outfile = 'test_trial.csv'):
        '''
        For each trial in <trials>, write a CSV representation of the trial's
        results to <outfile>
        '''
        f = open(outfile, 'w')
        header = Trial.format_header(self.feature_list)
        #header = ', '.join(self.feature_list + ['docname', 'correct', 'total']) + '\n'
        f.write(header)
        
        for t in trials:
            trial_as_csv = t.format_csv(self.feature_list)
            print 'here is a trial'
            print trial_as_csv
            f.write(trial_as_csv)

        print 'Wrote', outfile
        f.close()
    
    # get rid of
    def atom_to_atom(self):
        '''
        Returns the value of the atom-to-atom metric.
        '''
        # NOTE: This is waaaaaaay slow. Maybe one day we can figure out how to optimize it.
        
        if DEBUG:
            if len(self.get_plagiarized_spans()) == 0:
                print "NOTE: no plagiarzed passages in this document"
            print "Total passages:", len(self.passages)
            p_count = 0    
                
        matching = 0
        total = 0
        for p1 in self.passages:
            if DEBUG:
                p_count  += 1
                print "On passage", p_count
            for p2 in self.passages:
                if (self._is_plagiarized(p1) == self._is_plagiarized(p2)) and (p1.cluster_num == p2.cluster_num):
                    matching += 1
                total += 1
        return float(matching) / total
    
    def _perform_classification(self, test_file, features):
        '''
        Runs our classifier on the given <test_file> using the given list of <features>,
        self.atom_type, and kmeans clustering and returns a ROCTrial object.
        '''
        
        c = Controller(test_file + '.txt')
        
        passages = c.get_passages(self.atom_type, features, "kmeans", 2)
        plagiarzed_spans = self.get_plagiarized_spans(test_file + '.xml')
        
        passage_dicts = []
        for p in passages:
            d = {"cluster": p.cluster_num, "feature_vector": p.features, "is_gt_plagiarism": self._is_plagiarized(p, plagiarzed_spans)}
            passage_dicts.append(d)
        
        centroids = c.get_centroids()
        smallest_cluster = self._get_smallest_cluster(passages)
        plag_cluster = {"num": smallest_cluster, "centroid": centroids[smallest_cluster]}
        notplag_cluster_num = 1 if smallest_cluster == 0 else 0 #assumes k=2!
        notplag_cluster = {"num": notplag_cluster_num, "centroid": centroids[notplag_cluster_num]}
        
        
        return ROCTrial(test_file, features, "kmeans", self.atom_type, plag_cluster, notplag_cluster, passage_dicts)

    # refactor
    def perform_tests(self, composite = True, bust_cache = False):
        '''
        Runs our classifier on each file in self.base_file_paths. Tests each feature from
        self.feature_list individually, and all together if composite is True.
        
        Returns a list of ROCTrial objects generated as a result of each of these tests.
        '''
        trials = {}
        for i in self.feature_list:
            trials[i] = []
        
        for f in self.base_file_paths:
        
            if composite:
                trials[str(self.feature_list)] = [] # Add a dictionary key
                t = None
                if not bust_cache:
                    t = self._load_trial(f, self.feature_list)
                if not t:
                    t = self.perform_clasification(f, self.feature_list)
                    self._save_trial(t)
                trials[str(self.feature_list)].append(t)
                
            for feat in self.feature_list:
                t = None
                if not bust_cache:
                    t = self._load_trial(f, [feat])
                if not t:
                    t = self.perform_clasification(f, [feat])
                    self._save_trial(t)
                trials[feat].append(t)
                
        return trials
    
    def _get_trials(self):
        '''
        Returns ROCTrial objects for each file in self.base_file_paths tested with the
        features from self.feature_list, the passage size of self.atom_type, and kmeans
        clustering
        '''
        #TODO: support other clustering methods
        trials = []
        for f in self.base_file_paths:
            if DEBUG: print "On file "+str(len(trials)+1)+" of "+str(len(self.base_file_paths))
            t = self._load_trial(f, self.feature_list, self.atom_type, "kmeans")
            if not t:
                t = self._perform_classification(f, self.feature_list)
                self._save_trial(t)
            trials.append(t)
        return trials
    
    def _save_trial(self, trial):
        # TODO: The file name is not unique enough. Figure out what to do about this...
        # parts = trial.doc_name.split("/")
        # path = 'trials/' + parts[-2] + "/" + parts[-1] + str(trial.features[0])
        # self._ensure_dir(path)
        # f = open(path, "wb")
        # pickle.dump(trial, f)
        # f.close()
        # print "Just saved a new file at path", path
        
        trial_store.store_trial(trial)
        if DEBUG:
            print "Saved new trial for", trial.doc_name
    
    def _load_trial(self, doc_name, features, atom_type, cluster_strategy):
        return trial_store.load_trial(doc_name, features, atom_type, cluster_strategy)
        
    #def _load_trial(self, f, features):
    #    '''
    #    Returns None if the trial is not saved, otherwise returns the saved trial.
    #    '''
    #    parts = f.split("/")
    #    path = 'trials/' + parts[-2] + "/" + parts[-1] + str(features[0])
    #    try:
    #        f = open(path)
    #    except:
    #        print "Failed to load file at path", path
    #        return None
    #    t = pickle.load(f)
    #    f.close()
    #    print "Successfully loaded file at path", path
    #    return t
    
    #def _ensure_dir(self, f):
    #    # This method copied directly from:
    #    # http://stackoverflow.com/questions/273192/create-directory-if-it-doesnt-exist-for-file-write
    #    d = os.path.dirname(f)
    #    if not os.path.exists(d):
    #        os.makedirs(d)
    
    def generate_roc_plot(self):
        '''
        Generates an ROC plot from the average TP rate and FP rate from the the documents in
        self.base_file_paths using all features in self.feature_list in the feature vector
        
        Returns the path of the ROC plot
        '''
        # TODO: Check if the figure already exists in DB
        # TODO: Save figures to DB
        
        trials = self._get_trials()
        
        actuals = []
        confidences = []
        
        for trial in trials:
            for p in trial.passages:
                actuals.append(1 if p["is_gt_plagiarism"] else 0)
                confidences.append(trial._get_confidence(p))
        
        # actuals is a list of ground truth classifications for passages
        # confidences is a list of confidence scores for passages
        # So, if confidences[i] = .3 and actuals[i] = 1 then passage i is plagiarised and
        # we are .3 certain that it is plagiarism (So its in the non-plag cluster).
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(actuals, confidences, pos_label=1)
        roc_auc = sklearn.metrics.auc(fpr, tpr)
        
        # The following code is from http://scikit-learn.org/stable/auto_examples/plot_roc.html
        pyplot.clf()
        pyplot.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        pyplot.plot([0, 1], [0, 1], 'k--')
        pyplot.xlim([0.0, 1.0])
        pyplot.ylim([0.0, 1.0])
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        pyplot.title('Receiver operating characteristic')
        pyplot.legend(loc="lower right")
        
        path = "results/roc"+str(time.time())+".pdf"
        pyplot.savefig(path)
        return path
        
        
    # Axe this
    def generate_roc_plots(self, composite = True, bust_cache = False):
        '''
        Generates an ROC plots from the AVERAGE TP rates and FP rates from the documents in
        self.base_file_paths for each feature in self.feature_list and for all features in
        the list together if composite = True.
        
        Saves the plots to ...
        '''
        #http://scikit-learn.org/0.11/auto_examples/plot_roc.html
        #http://stackoverflow.com/questions/8192455/ntlk-python-plotting-roc-curve
        #http://gim.unmc.edu/dxtests/roc2.htm
        
        
        # TODO: Check for saved trials before generating new ones. Alternatively there
        #       could be a way to pass this function trials instead...
        # TODO: Come up with a legitimate naming scheme for the plots
        trials = self.perform_tests(composite, bust_cache)
        if DEBUG:
            for l in trials.values():
                for t in l:
                    pass
                    #print t.get_dictionary_representation()
        
        for feature_set in trials.keys():
            
            fpr_avgs = []
            tpr_avgs = []
            thresholds = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
        
            for t in thresholds:
                total = 0
                fpr_sum = 0.0
                tpr_sum = 0.0
            
                for trial in trials[feature_set]:
                    if DEBUG:
                        print trial.get_success_statistics(t), t
                    fpr_sum += trial.get_false_positive_rate(t)
                    tpr_sum += trial.get_true_positive_rate(t)
                    total += 1
                
                fpr_avgs.append(fpr_sum/float(total))
                tpr_avgs.append(tpr_sum/float(total))
        
            print feature_set
            print "points:", zip(fpr_avgs, tpr_avgs, thresholds)
            pyplot.clf()
            pyplot.plot(fpr_avgs, tpr_avgs, marker='o', color='r', ls='')
            pyplot.plot([0, 1], [0, 1], 'k--')
            pyplot.xlim([0.0, 1.0])
            pyplot.ylim([0.0, 1.0])
            pyplot.xlabel('False Positive Rate')
            pyplot.ylabel('True Positive Rate')
            pyplot.title('ROC curve '+feature_set)
            path = "results/roc"+str(time.time())+".pdf"
            pyplot.savefig(path)
            print "Just wrote chart to", path
   
    
def test():

    features = ['averageWordLength']
    test_file_listing = file('corpus_partition/training_set_files.txt')
    all_test_files = [f.strip() for f in test_file_listing.readlines()]
    test_file_listing.close()
    first_test_files = all_test_files[0:25]
    
    t = ToolTester('paragraph', features, first_test_files)
    t._get_trials() # Generates a trial for each file in the ToolTester using <features> for the feature vector
    print t.generate_roc_plot() # Generates a single roc plot that averages fpr and tpr from each file in the ToolTester
    
    # TODO: Add functions for accuracy and recall charts
    # TODO: Add the option to bust the cache
     
if __name__ == "__main__":
   test()

# all_features = [
#     'averageWordLength',
#     'averageSentenceLength',
#     'getPosPercentageVector',
#     'get_avg_word_frequency_class',
#     'get_punctuation_percentage',
#     'get_stopword_percentage'
# ]