# create_extrinsic_partition.py
# by Marcus Huderle
# Creates a random partition of the extrinsic corpus that ensures the source files
# of plagiarised documents are also included in the partition.  Also ensure that every suspect
# contains plagiarism.
#
# There are 14,428 total suspicious documents in the test pan-plagiarism-corpus-2009 directory

import os
import random
import xml.etree.ElementTree as ET
import sys
import re

from plagcomps.shared.util import IntrinsicUtility
from plagcomps.tokenization import tokenize

# m is the max number of paragraphs in a file
def main(m, training_percent = 0.7):
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

    print 'Grabbing all valid suspects...'
    # grab n files with plagiarism
    training_suspect_partition = [] 
    for filepaths in all_files:
        plag_spans = IntrinsicUtility.get_plagiarized_spans(filepaths[1])
        if len(plag_spans) > 0:
            # make sure it's at least m paragraphs
            f = open(filepaths[0], 'r')
            text = f.read()
            f.close()
            paragraphs = tokenize(text, 'paragraph')
            if len(paragraphs) > m:
                continue

            training_suspect_partition.append(filepaths)
            if len(training_suspect_partition) % 10 == 0:
                print len(training_suspect_partition)

    print len(training_suspect_partition)

    # print 'Writing partitions to disk...'
    # suspect_training_file = file("crisp_extrinsic_training_suspect_files.txt", 'w')
    # for suspect in training_suspect_partition:
    #     rel_path_start = suspect[0].index('/part')
    #     suspect_training_file.write(suspect[0][rel_path_start:-4] + '\n')
    # suspect_training_file.close()


    print 'Determining source documents for training partition...'
    training_sources = {}
    training_sources_suspects = {}
    num_files = 0
    for filenames in training_suspect_partition:
        tree = ET.parse(filenames[1])
        for feature in tree.iter("feature"):
            if feature.get("name") == "artificial-plagiarism" and feature.get("source_reference") and feature.get("source_reference")[:-4] not in training_sources:
                # figure out which partX the doc is in...so annoying...
                for p in sources_dirs:
                    if os.path.exists(sources_base_path + p + feature.get("source_reference")):
                        short_name = "/" + p + feature.get("source_reference")[:-4]
                        long_name = sources_base_path + p + feature.get("source_reference")
                        training_sources[short_name] = 1
                        if filenames[1] not in training_sources_suspects:
                            training_sources_suspects[filenames[1]] = [long_name]
                        else:
                            training_sources_suspects[filenames[1]].append(long_name)

        num_files += 1
        if num_files%100 == 0:
            print num_files,
            sys.stdout.flush()
    print
    print len(training_sources.keys()), 'sources for the training partition were found...'

    print 'Removing invalid suspects because of short sources...'
    # get rid of the ones that are too long...
    final_training_suspect_partition = []
    for _, xml in training_suspect_partition:
        # are all of its sources < m paragraphs?
        short_enough = True
        for source_filename in training_sources_suspects[xml]:
            f = open(source_filename, 'r')
            text = f.read()
            f.close()
            paragraphs = tokenize(text, 'paragraph')
            if len(paragraphs) > m:
                short_enough = False
                break
        if short_enough:
            final_training_suspect_partition.append(xml)

    print 'Constructing final source partition...'
    final_training_source_partition = []
    for suspect in final_training_suspect_partition:
        for long_name in training_sources_suspects[suspect]:
            short_name = '/' + re.sub(sources_base_path, '', long_name)
            if short_name not in final_training_source_partition:
                final_training_source_partition.append(short_name)

    print 'Converting suspects names.......'
    final_training_suspect_partition = ['/' + re.sub('.xml', '', re.sub(suspects_base_path, '', xml)) for xml in final_training_suspect_partition]

    print len(final_training_suspect_partition), final_training_suspect_partition
    print len(final_training_source_partition), final_training_source_partition

    print 'Writing suspect documents to disk...'
    suspects_training_file = file("crisp_corpus_suspect_files.txt", 'w')
    for filename in final_training_suspect_partition:
        suspects_training_file.write(filename + '\n')
    suspects_training_file.close()

    print 'Writing source documents to disk...'
    sources_training_file = file("crisp_corpus_source_files.txt", 'w')
    for filename in final_training_source_partition:
        sources_training_file.write(filename + '\n')
    sources_training_file.close()


if __name__ == '__main__':
    main(1000)