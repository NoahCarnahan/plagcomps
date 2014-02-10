'''
This module allows for the combined intrinsic/extrinsic analysis of text.
Given a text, analyze(...) will run intrinsic analysis on that document, and
identify the most suspicious passages. Then, it will run extrinsic detection on
each of the suspicious passages. It will return a % likelihood of plagiarism
for each suspicious paragraph in terms of intrinsic and extrinsic analysis.
'''
import intrinsic
import operator

INTRINSIC_SUSPICION_THRESHOLD = .85

def analyze(text, atom_type, features, cluster_type, k, fingerprint_type):
    '''Run intrinsic and extrinsic detection on the given text'''
    
    intrinsic_results = intrinsic.get_plagiarism(text, atom_type, features, cluster_type, k)
    intrinsic_results = sorted(intrinsic_results, key=operator.itemgetter(1), reverse=True)
    intrinsic_results = [x for x in intrinsic_results if x[1] > INTRINSIC_SUSPICION_THRESHOLD]
    
    # intrinsic_results now holds a list of tuples in which the tuple holds start and stop indices
    #   of a span within the text.

    for intrinsic_result in intrinsic_results:
        pass
        # 1: Find the fingerprint for this passage           
            # So get it from the database? or recreate it. This means that we need a way to interface
            # with fingerprint extractor with a span or a paragraph

        # 2: Look through all the source documents' fingerprints, and run jaccard similarity on each
            # Now we should have a confidence-of-plagiarism from the extrinsic world
            # for each of our passages, and we can list potential sources of plagiarism.

def _test():    
    f = open('/copyCats/pan-plagiarism-corpus-2009/external-detection-corpus/suspicious-documents/part4/suspicious-document06242.txt')
    text = f.read()
    f.close
    
    analyze(text, "paragraph", ['average_sentence_length', 'average_word_length'], "kmeans", 2, "anchor")
    
if __name__ == "__main__":
    _test()
