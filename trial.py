class Trial:

    def __init__(self, docname, features, true_positives, false_positives, true_negatives, false_negatives):
        self.docname = docname
        self.features = features
        self.true_positives = true_positives
        self.false_positives = false_positives
        self.true_negatives = true_negatives
        self.false_negatives = false_negatives
        self.num_correct = self.true_positives+self.true_negatives
        self.total = self.true_positives + self.false_positives + self.true_negatives + self.false_negatives
        
    def format_csv(self, all_possible_features):
        '''
        Returns a csv representation of this trial. If <all_possible_features> has 
        <n> features, the first <n> fields are binary indicators (1 if the
        feature was used in this trial). The following 3 fields are the document's 
        name, the number of correctly classified passages, and the total number
        of passages
        '''
        # feature_present_vec[i] = 0 iff all_possible_features[i] 
        # was used in this trial; 1 otherwise
        feature_present_vec = ['1' if f in self.features else '0' for f in all_possible_features ]

        output = feature_present_vec + [self.docname, str(self.num_correct), str(self.total), str(self.get_percision()), str(self.get_recall())]
        print 'output in format', output
        # Gives a csv representation of this trial
        return ', '.join(output) + '\n'


    def get_file_part_and_name(self, full_path):
        # TODO write this. Output would be nicer if it wasn't the full path
        # to the file used in this trial
        pass

    def get_pct_correct(self):
        return float(self.num_correct) / self.total
    
    def get_percision(self):
        return float(self.true_positives) / (self.true_positives + self.false_positives)
    
    def get_recall(self):
        return float(self.true_positives) / (self.true_positives + self.false_negatives)

