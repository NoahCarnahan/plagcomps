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
import data_store
import sklearn.metrics

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as pyplot

DEBUG = True

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
     
    def _get_plagiarized_spans(self, xml_file_path):
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
        
    #Refactor...
    #def test_features_individually(self):
    #    '''
    #    For each feature in <self.feature_list>, use ONLY that feature 
    #    and test every file in <self.file_list>
    #
    #    Output the results of each trial using <self.write_out_trials>
    #    '''
    #    # feature => trial objects for that (those) feature(s)
    #    all_trials = []
    #    for single_file in self.base_file_paths:
    #        print 'Working on', single_file
    #        c = Controller(single_file + '.txt')
    #        for single_feature in self.feature_list:
    #            print 'Using this feature:', single_feature
    #            all_trials.append(self.test_one_feature_set(c, [single_feature], single_file))
    #
    #    self.write_out_trials(all_trials)

    #Wut?
    #def write_out_trials(self, trials, outfile = 'test_trial.csv'):
    #    '''
    #    For each trial in <trials>, write a CSV representation of the trial's
    #    results to <outfile>
    #    '''
    #    f = open(outfile, 'w')
    #    header = Trial.format_header(self.feature_list)
    #    #header = ', '.join(self.feature_list + ['docname', 'correct', 'total']) + '\n'
    #    f.write(header)
    #    
    #    for t in trials:
    #        trial_as_csv = t.format_csv(self.feature_list)
    #        print 'here is a trial'
    #        print trial_as_csv
    #        f.write(trial_as_csv)
    #
    #    print 'Wrote', outfile
    #    f.close()
    
    def _perform_classification(self, test_file, features):
        '''
        Runs our classifier on the given <test_file> using the given list of <features>,
        self.atom_type, and kmeans clustering and returns a ROCTrial object.
        '''
        
        c = Controller(test_file + '.txt')
        
        passages = c.get_passages(self.atom_type, features, "kmeans", 2)
        plagiarzed_spans = self._get_plagiarized_spans(test_file + '.xml')
        
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
        data_store.store_trial(trial)
        if DEBUG:
            print "Saved new trial for", trial.doc_name
    
    def _load_trial(self, doc_name, features, atom_type, cluster_strategy):
        return data_store.load_trial(doc_name, features, atom_type, cluster_strategy)

    def _save_figure(self, fig):
        pass
    
    def _get_figure_path(self, stuff):
        pass
    
    def generate_roc_plot(self):
        '''
        Generates an ROC plot from the average TP rate and FP rate from the the documents in
        self.base_file_paths using all features in self.feature_list in the feature vector
        
        Returns the path of the ROC plot and the area under the curve
        '''
        trials = self._get_trials()
        
        #Check if the figure already exists in DB
        roc = data_store.load_roc(trials)
        if roc:
            return roc
        
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
        
        path = "figures/roc"+str(time.time())+".pdf"
        pyplot.savefig(path)
        data_store.store_roc(trials, path, roc_auc)
        return path, roc_auc
    
def test():

    features = ['averageSentenceLength', 'averageWordLength', 'get_avg_word_frequency_class','get_punctuation_percentage','get_stopword_percentage']
    #features = ['get_avg_word_frequency_class']
    test_file_listing = file('corpus_partition/training_set_files.txt')
    all_test_files = [f.strip() for f in test_file_listing.readlines()]
    test_file_listing.close()
    first_test_files = all_test_files[0:25]
    
    t = ToolTester('paragraph', features, first_test_files)
    t._get_trials() # Generates a trial for each file in the ToolTester using <features> for the feature vector
    print t.generate_roc_plot() # Generates a single roc plot that averages fpr and tpr from each file in the ToolTester
    
    # TODO: Add functions for accuracy and recall charts
    # TODO: Add the option to bust the cache

def foo():
    # cluster strat, docs, atom, features
    pass

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