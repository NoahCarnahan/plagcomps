'''
Run this script on the comps machine to get a random partitioning of the intrinsic suspect files
(plus their corresponding xml files) into a test and training set. The first command line argument
specifies waht proportion of the files should be in the training set.

Usage:
python random_corpus_partitioner.py .7
Outputs a dictionary with keys "test" and "train" where each value is a list consisting of tuples.
Each tuple is the absolute path of a suspect text document followed by the absolute path of the
corresponding xml file.
'''

import os
import random
import sys

random.seed(8)

def partition_corpus(training_percent = .7):
    intrinsic_corpus_path = "/copyCats/pan-plagiarism-corpus-2009/intrinsic-detection-corpus/suspicious-documents"
    intrinsic_dirs = ["part1/","part2/","part3/","part4/"]

    # Without extensions
    all_base_files = []
    all_files = [] # list of tuples where tuple[0] is the absolute path of the text document and tuple[1] is the absolute path of the xml file

    # Put all the suspect files in a list
    for d in intrinsic_dirs:
        p = os.path.join(intrinsic_corpus_path, d)
        for f in os.listdir(p):
            all_base_files.append(os.path.splitext(f)[0])

            if f[-4:] == ".txt":
                all_files.append((p+f, (p+f)[:-4]+".xml"))

    # Make sure all of these files actually exist
    worked = True
    for suspect in all_files:
        if not os.path.exists(suspect[0]):
            worked = False
            print suspect[0]
        if not os.path.exists(suspect[1]):
            worked = False
    assert(worked)

    # Shuffle the list
    random.shuffle(all_files)

    # Partition it
    cut_off = int(round( len(all_files)*training_percent ))
    training_set = all_files[:cut_off]
    test_set = all_files[cut_off:]
    print len(training_set)
    print len(test_set)

    training_set_file = file('training_set_files.txt', 'w')
    for trainer in training_set:
        rel_path_start = trainer[1].index('/part')
        training_set_file.write(trainer[1][rel_path_start:-4] + '\n')
    training_set_file.close()

    test_and_tuning_set_file = file('test_and_tuning_set_files.txt', 'w')
    for test_and_tuning in test_set:
        rel_path_start = test_and_tuning[1].index('/part')
        test_and_tuning_set_file.write(test_and_tuning[1][rel_path_start:-4] + '\n')
    test_and_tuning_set_file.close()

if __name__ == '__main__':
    training_pct = .7 if len(sys.argv) != 2 else float(sys.argv[1])

    partition_corpus(training_pct)

        
    #print {"train":training_set, "test":test_set}