from plagcomps.shared.util import BaseUtility, IntrinsicUtility
from plagcomps.intrinsic.cluster import cluster


def prec_recall_evaluate(reduced_docs, session, features, cluster_type, k, atom_type, corpus='intrinsic', feature_vector_weights=None, 
            metadata={}, cheating=False, cheating_min_len=5000, **clusterargs):
    '''
    features is a list of strings where each string is the name of a StylometricFeatureEvaluator method.
    cluster_type is "kmeans", "hmm", or "agglom".
    k is an integer.
    docs should be a list of full path strings.
    '''

    # From a few runs, looks like prec. is almost always bad (and doesn't improve
    # much with greater thresholds). So we might as well use lower thresholds
    # to bump up our recall (without hurting prec. very much)
    thresholds = [.001, .005, .01, .05, .1, .15, .2, .25, .35, .6, .8, .99]
    thresh_to_prec = {}
    thresh_to_recall = {}

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
            prec, rec = _one_doc_precision_and_recall(d, plag_likelihoods, thresh)
            thresh_to_prec.setdefault(thresh, []).append(prec)
            thresh_to_recall.setdefault(thresh, []).append(rec)
            doc_to_thresh_to_result[i][thresh] = (prec, rec)

    thresh_prec_avgs = {t : sum(l) / len(l) for t, l in thresh_to_prec.iteritems()}
    thresh_recall_avgs = {t : sum(l) / len(l) for t, l in thresh_to_recall.iteritems()}

    return doc_to_thresh_to_result, thresh_prec_avgs, thresh_recall_avgs

def _one_doc_precision_and_recall(doc, plag_likelihoods, prob_thresh, cheating=False, cheating_min_len=5000, **metadata):
    '''
    Returns the precision and recall for a given ReducedDoc <doc> using <plag_likelihoods> and 
    <prob_thresh> as a cutoff for whether or not a given section is called plagiarism. 
    '''
    spans = doc.get_spans(cheating, cheating_min_len)
    assert len(spans) == len(plag_likelihoods)
    actual_plag_spans = doc.get_plag_spans()

    # Keep the spans above <prob_thresh>
    detected_spans = [spans[i] for i in xrange(len(spans)) if plag_likelihoods[i] > prob_thresh]
    # if prob_thresh > .98:
    #     print detected_spans
    #     print plag_likelihoods
    #     print '-'*40
    prec, recall = _benno_precision_and_recall(actual_plag_spans, detected_spans)

    return prec, recall

def _benno_precision_and_recall(plag_spans, detected_spans):
    '''
    Paper referred to is "Overview of the 1st International Competition on Plagiarism Detection"
    <plag_spans> (set S in paper) is a list of spans like (start_char, end_char) of plag. spans
    <detected_spans> (set R in paper) is a list of spans like (start_char, end_char) that we defined as plag.
    '''
    util = BaseUtility()
    recall_sum = 0.0

    if len(plag_spans) == 0:
        # No plagiarism and we detected none -- recall of 1.0
        if len(detected_spans) == 0:
            recall = 1.0
        # No plagiarism, but we detected some -- recall of 0.0
        else:
            recall = 0.0
    else:
        # recall defined over all plag spans
        for pspan in plag_spans:
            pspan_len = float(pspan[1] - pspan[0])

            for dspan in detected_spans:
                temp_recall = util.overlap(pspan, dspan) / pspan_len
                recall_sum += temp_recall

        recall = recall_sum / len(plag_spans)

    if len(detected_spans) == 0:
        # Detected no plag., and there wasn't any. precision is 1.0
        if len(plag_spans) == 0:
            prec = 1.0
        # Detected no plag., but there was some! precision is 0
        else:
            prec = 0.0
    else:
        prec_sum = 0.0
        for dspan in detected_spans:
            dspan_len = float(dspan[1] - dspan[0])

            for pspan in plag_spans:
                temp_prec = util.overlap(dspan, pspan) / dspan_len
                prec_sum += temp_prec

        prec = prec_sum / len(detected_spans)

    return prec, recall

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

    prec, recall = _benno_precision_and_recall(plag_spans, detected_spans)
    print 'Prec: expected %f, got %f' % (expected_prec, prec)
    print 'Recall: expected %f, got %f' % (expected_recall, recall)


if __name__ == '__main__':
    _test()