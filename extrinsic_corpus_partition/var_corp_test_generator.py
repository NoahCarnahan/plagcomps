'''
This module will generate what I am calling the "variable length corpus" or "var corp."
The extrinsic utility will provide a function that allows a user to get n suspect documents
from the corpus and a list of the corresponding source documents for those n suspects.
The nice thing about this corpus is that for all n, the n-1 suspect docs are a subset of
the n suspect docs and the source docs that correspond to the n-1 suspect docs are also
a subset of the source docs that correspond to the n suspect docs. This means that if one
populates the database with the first 50 documents, then later wants to populate the database
with the first 100 documents, that about half the population will have already occurred.
Increasing the size of the part of the var corp that you use will result in a minimal increase
in the number of new files that need to be populated.
'''

from plagcomps.shared.util import ExtrinsicUtility
import os.path

loc = os.path.join(os.path.dirname(__file__), "crisp_TRAIN_var_corp.txt")
f = open(loc, "r")
lines = f.readlines()
f.close()

train_suspects = lines[0]
train_sources = lines[1]

SUSPECT_FILE_NAME = "crisp_corpus_suspect_files.txt"
SOURCE_FILE_NAME = "crisp_corpus_source_files.txt"
OUT_FILE_NAME = "crisp_TEST_var_corp.txt"

def get_source_files(suspect):
    '''
    Returns a set containing all of the documents that this suspect has plagiarism from.
    Items in the set are absolute paths.
    '''
    xml_path = suspect[:-3]+"xml"
    all_sources = ExtrinsicUtility().get_source_plag(xml_path)
    return all_sources

# These lists will contain all the information needed to grab a portion of the var corp.
# These will be written to a file (var_corp.txt) that the extrinsic utility will read from
# when giving the user the first n documents from the corpus. suspicious_files and 
# source_files will contain absolute paths!!
suspicious_files = []
source_files = []
source_cutoff = []

# Build the suspicious_files list by reading in extrinsic_training_suspect_files.txt
loc = os.path.join(os.path.dirname(__file__), SUSPECT_FILE_NAME)
f = open(loc, "r")
for line in f:
    full_name = "/copyCats/pan-plagiarism-corpus-2009/external-detection-corpus/suspicious-documents"+line.strip()+".txt"
    suspicious_files.append("/copyCats/pan-plagiarism-corpus-2009/external-detection-corpus/suspicious-documents"+line.strip()+".txt")
f.close()

print suspicious_files

# Initialize the source_files list by reading in extrinsic_training_source_files.txt
loc = os.path.join(os.path.dirname(__file__), SOURCE_FILE_NAME)
f = open(loc, "r")
for line in f:
    source_files.append("/copyCats/pan-plagiarism-corpus-2009/external-detection-corpus/source-documents"+line.strip())
f.close()

#print source_files

# Sanitize source and suspect files
clean_suspicious_files = []
for suspect in suspicious_files:
    print suspect
    if suspect not in train_suspects:
        cur_source_files = get_source_files(suspect)
        if len(set(cur_source_files).intersection(set(train_sources))) == 0:
            clean_suspicious_files.append(suspect)


# Rearrange source_files and build the source_cutoff list
first_unarranged_spot = 0
i = 1
total = float(len(clean_suspicious_files))
last = 0
for suspect in clean_suspicious_files:
    if int((i/total)*100)/1 == last+5:
        print last+5 , "%"
        last += 5
    cur_source_files = get_source_files(suspect)
    for source in cur_source_files:
        cur_pos = source_files.index(source)
        if cur_pos > first_unarranged_spot:
            # swap the source file and the file currently at first_unaranged_spot
            source_files[first_unarranged_spot], source_files[cur_pos] = source_files[cur_pos], source_files[first_unarranged_spot]
            first_unarranged_spot += 1
    source_cutoff.append(first_unarranged_spot)
    i += 1
    

# Write the lists to var_corp.txt
loc = os.path.join(os.path.dirname(__file__), OUT_FILE_NAME)
f = open(loc, "w")
f.write(str(clean_suspicious_files)+"\n"+str(source_files)+"\n"+str(source_cutoff)+"\n")
f.close()

