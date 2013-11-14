# create_extrinsic_partition.py
# by Marcus Huderle
# Creates a random partition of the extrinsic corpus that ensures the source files
# of plagiarised documents are also included in the partition.
#
# There are 14,428 total suspicious documents in the test pan-plagiarism-corpus-2009 directory

import os
import random
import xml.etree.ElementTree as ET
import sys

def main(training_percent = 0.7):
    random.seed(1337)

    suspects_base_path = "/copyCats/pan-plagiarism-corpus-2009/external-detection-corpus/suspicious-documents/"
    suspects_dirs = ["part1/", "part2/", "part3/", "part4/", "part5/", "part6/", "part7/", "part8/"]
    sources_base_path = "/copyCats/pan-plagiarism-corpus-2009/external-detection-corpus/source-documents/"
    sources_dirs = ["part1/", "part2/", "part3/", "part4/", "part5/", "part6/", "part7/", "part8/"]

     # Without extensions
    all_base_files = []
    all_files = [] # list of tuples where tuple[0] is the absolute path of the text document and tuple[1] is the absolute path of the xml file

    # Put all the suspect files in a list
    for d in suspects_dirs:
        p = os.path.join(suspects_base_path, d)
        for f in os.listdir(p):
            all_base_files.append(os.path.splitext(f)[0])

            if f[-4:] == ".txt":
                all_files.append((p+f, (p+f)[:-4]+".xml"))
    
    # Make sure all of these files actually exist
    worked = True
    for suspect in all_files:
        if not os.path.exists(suspect[0]):
            worked = False
            print ".txt file does not exist:", suspect[0]
        if not os.path.exists(suspect[1]):
            worked = False
            print ".xml file does not exist:", suspect[1]
    assert(worked)

    # shuffle and take files from the front of the list
    print 'Shuffling ', len(all_files), 'suspect files...'
    random.shuffle(all_files)

    cutoff = int(len(all_files)*training_percent)
    print 'Splitting suspect files into', cutoff, 'training files and', len(all_files)-cutoff, 'testing files...'
    training_suspect_partition = all_files[:cutoff]
    testing_suspect_partition = all_files[cutoff:]

    print 'Writing partitions to disk...'
    suspect_training_file = file("extrinsic_training_suspect_files.txt", 'w')
    for suspect in training_suspect_partition:
        rel_path_start = suspect[0].index('/part')
        suspect_training_file.write(suspect[0][rel_path_start:-4] + '\n')
    suspect_training_file.close()

    suspect_testing_file = file("extrinsic_testing_suspect_files.txt", 'w')
    for suspect in testing_suspect_partition:
        rel_path_start = suspect[0].index('/part')
        suspect_testing_file.write(suspect[0][rel_path_start:-4] + '\n')
    suspect_testing_file.close()

    print 'Determining source documents for training partition...'
    training_sources = {}
    num_files = 0
    for filenames in training_suspect_partition:
        tree = ET.parse(filenames[1])
        for feature in tree.iter("feature"):
            if feature.get("name") == "artificial-plagiarism" and feature.get("source_reference") and feature.get("source_reference")[:-4] not in training_sources:
                # figure out which partX the doc is in...so annoying...
                for p in sources_dirs:
                    if os.path.exists(sources_base_path + p + feature.get("source_reference")):
                        training_sources["/" + p + feature.get("source_reference")[:-4]] = 1
        num_files += 1
        if num_files%100 == 0:
            print num_files,
            sys.stdout.flush()
    print
    print len(training_sources.keys()), 'sources for the training partition were found...'

    print 'Determining source documents for testing partition...'
    testing_sources = {}
    num_files = 0
    for filenames in testing_suspect_partition:
        tree = ET.parse(filenames[1])
        for feature in tree.iter("feature"):
            if feature.get("name") == "artificial-plagiarism" and feature.get("source_reference") and feature.get("source_reference")[:-4] not in training_sources:
                # figure out which partX the doc is in...so annoying...
                for p in sources_dirs:
                    if os.path.exists(sources_base_path + p + feature.get("source_reference")):
                        testing_sources["/" + p + feature.get("source_reference")[:-4]] = 1
        num_files += 1
        if num_files%100 == 0:
            print num_files,
            sys.stdout.flush()
    print
    print len(testing_sources.keys()), 'sources for the testing partition were found...'

    print 'Writing source documents to disk...'
    source_training_file = file("extrinsic_training_source_files.txt", 'w')
    for filename in training_sources.keys():
        source_training_file.write(filename + '\n')
    source_training_file.close()

    source_testing_file = file("extrinsic_testing_source_files.txt", 'w')
    for filename in testing_sources.keys():
        source_testing_file.write(filename + '\n')
    source_testing_file.close()
    

if __name__ == '__main__':
    main()