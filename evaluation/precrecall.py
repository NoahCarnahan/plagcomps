from plagcomps.shared.util import BaseUtility, IntrinsicUtility
from plagcomps.intrinsic.cluster import cluster

import matplotlib.pyplot as plt
import os
import time

import math
DEBUG = False

def prec_recall_evaluate(reduced_docs, session, features, cluster_type, k, atom_type, corpus='intrinsic', feature_vector_weights=None, 
            metadata={}, cheating=False, cheating_min_len=5000, **clusterargs):
    '''
   
    '''
    thresholds = [.05 * i for i in range(20)]

    thresh_to_prec = {}
    thresh_to_recall = {}
    thresh_to_fmeasure = {}
    thresh_to_granularity = {}
    thresh_to_overall = {}

    # doc_to_thresh_to_result[i] = {thresh -> (prec, recall)}
    doc_to_thresh_to_result = []
    
    count = 0
    valid_reduced_docs = []

    for i, d in enumerate(reduced_docs):
        doc_to_thresh_to_result.append({})
        count += 1
        
        print "On document", d, ". The", count, "th document."

        feature_vecs = d.get_feature_vectors(features, session, cheating=cheating, cheating_min_len=cheating_min_len)
        # skip if there are no feature_vectors
        if cheating and len(feature_vecs) < 7: # 7, because that's what Benno did
            continue
        valid_reduced_docs.append(d)

        if feature_vector_weights:
            weighted_vecs = []
            for vec in feature_vecs:
                cur_weight_vec = []
                for i, weight in enumerate(feature_vector_weights, 0):
                    cur_weight_vec.append(vec[i] * weight)
                weighted_vecs.append(cur_weight_vec)
            feature_vecs = weighted_vecs

        # Cluster to get plag probs
        plag_likelihoods = cluster(cluster_type, k, feature_vecs, **clusterargs)
        for thresh in thresholds:
            prec, rec, fmeasure, granularity, overall = _one_doc_all_measures(d, plag_likelihoods, thresh)

            # If measure wasn't well defined, None is returned. NOTE (nj) sneaky bug:
            # if we use the construct 
            # if prec:
            #   <add it to the dict>
            # never add prec or recall when they're 0. 
            # Thus we explicitly check for None-ness
            if prec is not None:
                thresh_to_prec.setdefault(thresh, []).append(prec)
            if rec is not None:
                thresh_to_recall.setdefault(thresh, []).append(rec)
            if fmeasure is not None:
                thresh_to_fmeasure.setdefault(thresh, []).append(fmeasure)
            if granularity is not None:
                thresh_to_granularity.setdefault(thresh, []).append(granularity)
            if overall is not None:
                thresh_to_overall.setdefault(thresh, []).append(overall)

            doc_to_thresh_to_result[i][thresh] = (prec, rec, fmeasure, granularity, overall)


    # For a given threshold, how many documents had valid precisions?
    print 'Valid precision:', sorted([(th, len(l)) for th, l in thresh_to_prec.iteritems()])
    # For a given threshold, how many documents had valid recall?
    print 'Valid recall (this number should not change):', sorted([(th, len(l)) for th, l in thresh_to_recall.iteritems()])

    thresh_prec_avgs = {t : sum(l) / len(l) for t, l in thresh_to_prec.iteritems()}
    thresh_recall_avgs = {t : sum(l) / len(l) for t, l in thresh_to_recall.iteritems()}
    thresh_fmeasure_avgs = {t : sum(l) / len(l) for t, l in thresh_to_fmeasure.iteritems()}
    thresh_granularity_avgs = {t : sum(l) / len(l) for t, l in thresh_to_granularity.iteritems()}
    thresh_overall_avgs = {t : sum(l) / len(l) for t, l in thresh_to_overall.iteritems()}

    if DEBUG:
        for thresh in sorted(thresh_prec_avgs.keys()):
            print thresh
            print 'Prec:', thresh_prec_avgs[thresh]
            print 'Recall:', thresh_recall_avgs[thresh]
            print 'F-Measure:', thresh_fmeasure_avgs[thresh]
            print 'Granularity:', thresh_granularity_avgs[thresh]
            print 'Overall:', thresh_overall_avgs[thresh]
            print '-'*40

    return thresh_prec_avgs, thresh_recall_avgs, thresh_fmeasure_avgs, thresh_granularity_avgs, thresh_overall_avgs

def _one_doc_all_measures(doc, plag_likelihoods, prob_thresh, cheating=False, cheating_min_len=5000, **metadata):
    '''
    Returns the precision and recall for a given ReducedDoc <doc> using <plag_likelihoods> and 
    <prob_thresh> as a cutoff for whether or not a given section is called plagiarism. 
    '''
    spans = doc.get_spans(cheating, cheating_min_len)
    assert len(spans) == len(plag_likelihoods)
    actual_plag_spans = doc.get_plag_spans()

    # Keep the spans above <prob_thresh>
    detected_spans = [spans[i] for i in xrange(len(spans)) if plag_likelihoods[i] > prob_thresh]

    if DEBUG:
        print 'Thresh: %f. Detected: %i. Actual: %i' % (prob_thresh, len(detected_spans), len(actual_plag_spans))
        
    prec, recall, fmeasure, granularity, overall = get_all_measures(actual_plag_spans, detected_spans)

    return prec, recall, fmeasure, granularity, overall

def get_all_measures(plag_spans, detected_spans):
    '''
    Returns all measures:
    prec, recall, fmeasure, granularity, overall
    '''
    prec, recall = _benno_precision_and_recall(plag_spans, detected_spans)
    if prec is None or recall is None:
        fmeasure = None
        granularity = None
        overall = None
    else:
        fmeasure = _fmeasure(prec, recall)
        granularity = _benno_granularity(plag_spans, detected_spans)
        overall = _benno_overall(fmeasure, granularity)

    return prec, recall, fmeasure, granularity, overall

def _benno_precision_and_recall(plag_spans, detected_spans):
    '''
    Paper referred to is "Overview of the 1st International Competition on Plagiarism Detection"
    <plag_spans> (set S in paper) is a list of spans like (start_char, end_char) of plag. spans
    <detected_spans> (set R in paper) is a list of spans like (start_char, end_char) that we defined as plag.

    Edge cases: if there are no plagiarized spans, there is no notion of recall. Returns None.
    If we detect nothing, then there is no notion of precision. Returns None.
    '''
    util = BaseUtility()

    if len(plag_spans) == 0:
        recall = None
    else:
        recall_sum = 0.0

        # recall defined over all plag spans
        for pspan in plag_spans:
            pspan_len = float(pspan[1] - pspan[0])

            for dspan in detected_spans:
                temp_recall = util.overlap(pspan, dspan) / pspan_len
                recall_sum += temp_recall

        recall = recall_sum / len(plag_spans)

    if len(detected_spans) == 0:
        prec = None
    else:
        prec_sum = 0.0

        for dspan in detected_spans:
            dspan_len = float(dspan[1] - dspan[0])

            for pspan in plag_spans:
                temp_prec = util.overlap(dspan, pspan) / dspan_len
                prec_sum += temp_prec

        prec = prec_sum / len(detected_spans)

    return prec, recall

def _fmeasure(prec, recall):
    '''
    Returns harmonic mean (F-measure) of a given precision and recall
    '''
    if prec is None or recall is None:
        fmeasure = None
    elif (prec + recall) == 0.0:
        fmeasure = 0.0
    else:
        fmeasure = (2 * prec * recall) / (prec + recall)

    return fmeasure

def _benno_granularity(plag_spans, detected_spans):
    '''
    Granularity is defined in the paper -- essentially trying to measure
    how fine-grained a given detected_span is. 
    '''
    if len(detected_spans) == 0:
        return 1.0

    util = BaseUtility()
    # The S_R defined in the paper: set of plag_spans that overlap 
    # some detected span
    detected_overlaps = []

    # The C_s defined in the paper: set of detected_spans that overlap 
    # plag_span s
    # actual_overlaps[plag_span] = [list of detected_spans that overlap plag_span]
    actual_overlaps = {}
    
    for pspan in plag_spans:
        for dspan in detected_spans:
            if util.overlap(pspan, dspan) > 0:
                detected_overlaps.append(pspan)
                actual_overlaps.setdefault(tuple(pspan), []).append(dspan)

    gran_sum = 0.0
    for d_overlap in detected_overlaps:
        gran_sum += len(actual_overlaps[tuple(d_overlap)])

    if len(detected_overlaps) == 0:
        gran = 1.0 
    else:
        gran = gran_sum / len(detected_overlaps)

    return gran

def _benno_overall(fmeasure, gran):
    '''
    Returns overall measure defined in Stein's paper
    '''
    return fmeasure / math.log(1 + gran, 2)

def _deprecated_benno_precision_and_recall(plag_spans, detected_spans):
    '''
    NOTE (nj) this is the way the competition specified precision and recall, but doesn't
    seem to make a ton of sense: when choosing a threshold, it's in our best interest to
    call everything non-plagiarized and get prec and recall values of 1.0 for all the non-plagiarized
    documents. We could create a corpus of docs containing plag., but that also doesn't seem to
    be in the spirit of detection in general.
    
    Paper referred to is "Overview of the 1st International Competition on Plagiarism Detection"
    <plag_spans> (set S in paper) is a list of spans like (start_char, end_char) of plag. spans
    <detected_spans> (set R in paper) is a list of spans like (start_char, end_char) that we defined as plag.
    '''
    util = BaseUtility()

    # Edge cases -- defined according to performance_measures script provided online
    # http://www.uni-weimar.de/medien/webis/research/events/pan-09/pan09-code/pan09-plagiarism-detection-performance-measures.py
    if len(plag_spans) == 0 and len(detected_spans) == 0:
        prec = 1.0
        recall = 1.0
    elif len(plag_spans) == 0 or len(detected_spans) == 0:
        prec = 0.0
        recall = 0.0
    else:
        recall_sum = 0.0

        # recall defined over all plag spans
        for pspan in plag_spans:
            pspan_len = float(pspan[1] - pspan[0])

            for dspan in detected_spans:
                temp_recall = util.overlap(pspan, dspan) / pspan_len
                recall_sum += temp_recall

        recall = recall_sum / len(plag_spans)

   
        prec_sum = 0.0
        for dspan in detected_spans:
            dspan_len = float(dspan[1] - dspan[0])

            for pspan in plag_spans:
                temp_prec = util.overlap(dspan, pspan) / dspan_len
                prec_sum += temp_prec

        prec = prec_sum / len(detected_spans)

    return prec, recall


def visualize_overlaps(plag_spans, detected_spans, **metadata):
    plag_y = .51
    detected_y = .49

    for pspan_start, pspan_end in plag_spans:
        width = pspan_end - pspan_start
        plt.barh(plag_y, width, height=.01, align='center', left=pspan_start, color='blue')

    for dspan_start, dspan_end in detected_spans:
        width = dspan_end - dspan_start
        plt.barh(detected_y, width, height=.01, align='center', left=dspan_start, color='red')
    plt.yticks([.4, .6], ['plag', 'detected'])
    if 'doc_name' in metadata:
        plt.title(metadata['doc_name'])
    if 'thresh' in metadata:
        plt.set_xlabel(metadata['thresh'])

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.set_yticks((0, 1))
    # ax.set_yticklabels(('Plag Spans', 'Detected Spans'))
    # ax.set_ybound((-.2, 1.2))
    # ax.set_xlabel('Span Index')

    # for pspan_start, pspan_end in plag_spans:
    #     ax.hlines(plag_y, pspan_start, pspan_end, colors ='blue', linewidths = 4)

    # for dspan_start, dspan_end in detected_spans:
    #     ax.hlines(detected_y, dspan_start, dspan_end, colors ='red', linewidths = 4)


    path = os.path.join(os.path.dirname(__file__), "../figures/overlap_viz/"+str(time.time())+".pdf")
    plt.savefig(path)

def _test():
    plag_spans = [
        [10, 21],
        [32, 40],
        [51, 57]
    ]

    detected_spans = [
        [22, 34],
        [45, 55]
    ]
    expected_recall = ((2.0 / 8) + (4.0 / 6)) / 3.0
    expected_prec = ((2.0 / 12) + 4.0 / 10) / 2.0
    expected_fmeasure = _fmeasure(expected_prec, expected_recall)
    expected_gran = 1

    prec, recall = _benno_precision_and_recall(plag_spans, detected_spans)
    fmeasure = _fmeasure(prec, recall)
    gran = _benno_granularity(plag_spans, detected_spans)
    overall = _benno_overall(fmeasure, gran)

    print 'Prec: expected %f, got %f' % (expected_prec, prec)
    print 'Recall: expected %f, got %f' % (expected_recall, recall)
    print 'Fmeasure: expected %f, got %f' % (expected_fmeasure, fmeasure)
    print 'Granularity: exepected %f, got %f' % (expected_gran, gran)
    print 'Overall %f' % overall
    visualize_overlaps(plag_spans, detected_spans)

def _return_all_plag_test():
    plag_spans = [
        [10, 21],
        [32, 40],
        [51, 57]
    ]

    detected_spans = [
        [0, 65]
    ]

    prec, recall, fmeasure, granularity, overall = get_all_measures(plag_spans, detected_spans)
    print 'Prec:', prec
    print 'Recall:', recall
    print 'F-Measure:', fmeasure
    print 'Granularity:', granularity
    print 'Overall:', overall


if __name__ == '__main__':
    #_return_all_plag_test()
    _test()
