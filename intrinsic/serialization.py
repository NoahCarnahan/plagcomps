import csv
import os

from plagcomps.intrinsic.featureextraction import FeatureExtractor
from plagcomps.shared.passage import IntrinsicPassage
from plagcomps.shared.util import IntrinsicUtility
from plagcomps.intrinsic.cluster import cluster


def batch_serialize(n=100):
    '''
    Writes csv files ('serializations') of the passages parsed from first <n>
    training files 
    '''
    out_dir = os.path.join(os.path.dirname(__file__), 'serialized')
    util = IntrinsicUtility()
    training_files = util.get_n_training_files(n, include_txt_extension=False)

    text_files = [t + '.txt' for t in training_files]
    xml_files = [t + '.xml' for t in training_files]
    out_files = [os.path.join(out_dir, os.path.basename(t) + '.csv') for t in training_files]

    for tf, xf, of in zip(text_files, xml_files, out_files):
        # Only populate if outfile doesn't already exist
        if not os.path.exists(of):
            print of, 'did not exist. Working on it now.'
            extract_and_serialize(tf, xf, of)


def extract_and_serialize(txt_file, xml_file, out_file, atom_type='paragraph',
                          cluster_method='kmeans', k=2):
    '''
    Performs all of intrinsic (feature extraction, clustering etc.) and creates
    Passage objects for each passage in <txt_file>. Writes a CSV file out
    to <out_file> containing all the features of <txt_file>

    The CSV files can be read easily by R in order to create plots
    '''
    f = file(txt_file, 'r')
    text = f.read()
    f.close()

    util = IntrinsicUtility() 

    feature_names = [
        'average_word_length',
        'average_sentence_length',
        'stopword_percentage',
        'punctuation_percentage',
        'syntactic_complexity',
        'avg_internal_word_freq_class',
        'avg_external_word_freq_class'
    ]
   

    ext = FeatureExtractor(text)
    print 'Initialized extractor'
    # Note that passages don't know their ground truths yet
    passages = ext.get_passages(feature_names, atom_type)
    print 'Extracted passages'
    util.add_ground_truth_to_passages(passages, xml_file)

    feature_vecs = [p.features.values() for p in passages]

    # If just testing feature extraction, don't cluster passages
    if cluster_method != 'none':
        # Cluster the passages and set their confidences
        confidences = cluster(cluster_method, k, feature_vecs)
        for psg, conf in zip(passages, confidences):
            psg.set_plag_confidence(conf)

    f = file(out_file, 'wb')
    csv_writer = csv.writer(f)

    # Writes out the header for corresponding CSV
    csv_writer.writerow(IntrinsicPassage.serialization_header(feature_names))
    for p in passages:
        csv_writer.writerow(p.to_list(feature_names))
    f.close()
    print 'Finished writing', out_file

if __name__ == '__main__':
    batch_serialize()