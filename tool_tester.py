from controller import Controller
from trial import Trial

import xml.etree.ElementTree as ET
from collections import Counter
import cProfile
import numpy
import scipy

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
        
if __name__ == "__main__":
    # all_features = [
    #     'averageWordLength',
    #     'averageSentenceLength',
    #     'getPosPercentageVector',
    #     'get_avg_word_frequency_class',
    #     'get_punctuation_percentage',
    #     'get_stopword_percentage'
    # ]
    
    all_features = [
        'averageSentenceLength',
        'averageWordLength'
    ]
    test_file_listing = file('corpus_partition/training_set_files.txt')
    all_test_files = [f.strip() for f in test_file_listing.readlines()]
    test_file_listing.close()

    # Just try the first 50 for the moment
    first_test_files = all_test_files[:15]
    
    t = ToolTester('paragraph', all_features, first_test_files)
    for thresh in (.4, .6):
        t.test_full_feature_set(thresh)
