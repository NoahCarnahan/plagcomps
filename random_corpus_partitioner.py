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

training_percent = .5
if len(sys.argv) == 2:
    training_percent = float(sys.argv[1])

intrinsic_corpus_path = "/copyCats/pan-plagiarism-corpus-2009/intrinsic-detection-corpus/suspicious-documents"
intrinsic_dirs = ["part1/","part2/","part3/","part4/"]
all_files = [] # list of tuples where tuple[0] is the absolute path of the text document and tuple[1] is the absolute path of the xml file

# Put all the suspect files in a list
for d in intrinsic_dirs:
    p = os.path.join(intrinsic_corpus_path, d)
    for f in os.listdir(p):
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

print {"train":training_set, "test":test_set}