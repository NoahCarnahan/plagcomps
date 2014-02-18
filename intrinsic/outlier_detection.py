from numpy import mean, std, sqrt, log, exp
from scipy.stats import norm, shapiro, anderson
import math


# (nj) TODO: Come up with different (better?) ways of
# (1) Combining "probabilities" for a single passage. Right now we just mulitply 
# the probabilities we get from each of the features. Done in:
# _combine_feature_probs(prob_vector)
# (2) Building a "confidence" score based on the relative "probability" of 
# a passage being plagiarized vs. not-plagiarized. Done in:
# _get_confidence(plag_prob, non_plag_prob) 
# (3) Scaling the "confidence" scores across an entire document (set of 
# stylometric features). Done in:
# _scale_confidences(confs) 

MIN_PROB = 10**(-30)
# The confidence we use when we are unable to calculate anything about the
# likelihood of plag
IMPURITY_ASSUMPTION = .2


def density_based(stylo_vectors, center_at_mean=True, num_to_ignore=1, impurity=.2, feature_confidence_weights=None):
    '''
    Implements the algorithm described in Stein, Lipka, Prettenhofer's
    "Intrinsic Plagiarism Analysis", Section 2.4 "Outlier Detection"

    Estimates the distribution of each feature's non-plag. portion as a normal distribution
    simply using the MLE (i.e. a normal distribution with sample mean
    and sample std as parameters). Note that the min and max of the observed
    features are removed before computing sample mean/std (and this could perhaps
    be extended to remove the <k> largest/smallest observations before computing
    sample mean/std)

    if <center_at_mean>, then normal dist is centered at feature's mean
    if not, then normal dist at feature's median

    When calculating the mean, the minimum and maximum <num_to_ignore> elements
    are ignored. 

    NOTE <impurity> argument is ignored right now.
    '''
    transpose = _rotate_vectors(stylo_vectors)
    confidences = []

    means, stds, mins, medians, maxs = [], [], [], [], []
    normality_pvals = []

    for row in transpose:
        cur_mean, cur_std, cur_min, cur_median, cur_max = _get_distribution_features(row, num_to_ignore)
        means.append(cur_mean)
        stds.append(cur_std)
        mins.append(cur_min)
        medians.append(cur_median)
        maxs.append(cur_max)

        # Print some data about the normality of our features
        norm_p = _test_normality(row)
        normality_pvals.append(norm_p)

    #print 'Normality pvals have min, max, mean:', min(normality_pvals), max(normality_pvals), mean(normality_pvals)
    
    for i in xrange(len(stylo_vectors)):
        vec = stylo_vectors[i]
        # For current <vec>,
        # featurewise_plag_prob[i] == prob. that feature <i> was plagiarized in <vec>
        # NOTE that these are taken from PDFs, so they don't actually correspond
        # to real probabilities 
        featurewise_nonplag_prob = []
        featurewise_plag_prob = []
        featurewise_confs = []

        for feat_num in xrange(len(vec)):
            # TODO plag_prob is just constant -- precompute this
            cur_val = vec[feat_num]
            cur_center = means[feat_num] if center_at_mean else medians[feat_num]
            cur_std = stds[feat_num]
            cur_min, cur_max = mins[feat_num], maxs[feat_num]

            
            # A std of 0 => the feature is constant, and therefore won't 
            # help us distinguish anything!
            if cur_std != 0.0 and not _in_uncertainty_interval(cur_val, cur_center, cur_std):

                cur_norm_prob = _get_norm_prob(cur_val, cur_center, cur_std)
                if math.isnan(cur_norm_prob) or cur_norm_prob == 0.0:
                    cur_norm_prob = MIN_PROB
                    #print 'Norm prob was nan or 0: %f. Using MIN_PROB' % cur_norm_prob

                cur_unif_prob = _get_unif_prob(cur_val, cur_min, cur_max)
                if math.isnan(cur_unif_prob) or cur_unif_prob == 0.0:
                    cur_unif_prob = MIN_PROB
                    print 'Unif prob was nan or 0: %f. Using MIN_PROB' % cur_unif_prob


                featurewise_confs.append(_get_confidence(cur_unif_prob, cur_norm_prob))
            # TODO what happens if all points are in uncertainty interval??
        confidences.append(_combine_feature_probs(featurewise_confs))
            
    # No more scaling -- we're already returning probs between 0 and 1, so should be all good
    # scaled = _scale_confidences(confidences)
    # return scaled

    return confidences

def _combine_feature_probs(prob_vector):
    '''
    Returns the Naive Bayes version of combining probabilites: just multiply them.
    Note that we take the sum of the log of each probability and exponentiate
    in hopes of avoiding underflow when multiplying many small numbers

    If the probability vector is empty, then all features were in the uncertainty
    interval. Return the IMPURITY_ASSUMPTION (unless something else logical comes
    about)
    '''
    if len(prob_vector) == 0:
        return IMPURITY_ASSUMPTION
    else:
        return exp(sum(log(prob_vector)))

def _get_confidence(plag_prob, non_plag_prob):
    '''
    Returns some notion of confidence:
    If we think there's plag., return the Naive Bayes estimated prob of plag
    If not, return the negative of the Naive Bayes estimate prob of NOT plag

    Other options:
    if plag_prob > non_plag_prob:
        return plag_prob
    else:
        return -non_plag_prob
    
    The above is an old notion of confidence, which eventually gets
    scaled. It worked well at the start, but also doesn't make a ton
    of sense...

    Note that these values are scaled later on to be between 0 and 1
    '''
    return plag_prob / (plag_prob + non_plag_prob) 

def _scale_confidences(confs):
    '''
    NOT BEING USED ANYMORE. The way we combine combine plag/non-plag probs 
    should already give confs between 0 and 1.

    Scales all "confidences" to (0, 1) interval simply by dividing by 
    the maximum "confidence"

    If <confs> is constant (i.e. contains all the same values), return
    a vector of all IMPURITY_ASSUMPTIONs. 
    '''
    # offset will be either 0 or some negative number, in which case
    # we subtract the negative offset (i.e. add)
    offset = min(min(confs), 0.0)
    max_conf_with_offset = max(confs) - offset

    if max_conf_with_offset == 0.0:
        return [IMPURITY_ASSUMPTION for x in confs]
    else:
        return [(x - offset) / max_conf_with_offset for x in confs]

def _get_distribution_features(row, extremes_to_ignore):
    '''
    <row> corresponds to all the observed values of feature

    Per Section 2.4 in Stein, Lipka, Prettenhofer's "Intrinsic Plagiarism Analysis",
    removes min and max when calculating parameters of Gaussian distribution
    '''
    sorted_row = sorted(row)

    min_val = sorted_row[0]
    max_val = sorted_row[-1]
    median_val = sorted_row[len(sorted_row) / 2]

    row_without_extremes = sorted_row[extremes_to_ignore : -extremes_to_ignore]
    mean_of_row = mean(row_without_extremes)
    std_of_row = std(row_without_extremes)

    return mean_of_row, std_of_row, min_val, median_val, max_val


def _in_uncertainty_interval(x, center, sd_dist):
    '''
    Returns whether <x> is between 1 and 2 standard deviations away from
    <center> in either direction. Such points are dubbed "uncertain"
    in Stein et. al.'s paper
    '''
    right_interval = (center + sd_dist, center + 2 * sd_dist)
    left_interval = (center - 2 * sd_dist, center - sd_dist)

    if right_interval[0] <= x <= right_interval[1] or \
        left_interval[0] <= x <= left_interval[1]:
        return True

    return False


def _get_norm_prob(x, loc, scale):
    '''
    Returns normal PDF evaluated at <x> for a normal dist centered at <loc> 
    with SD == <scale>
    '''
    return norm.pdf(x, loc, scale)

def _get_unif_prob(v, min_val, max_val):
    '''
    Returns uniform PDF (probability density function)
    '''
    diff = max_val - min_val
    if diff == 0:
        return 1.0
    else:
        return 1.0 / (max_val - min_val)

def _rotate_vectors(mat):
    '''
    This is more verbose than it needs to be, but also perhaps 
    more readable. Transposes <mat> and returns it
    '''
    rotated = []

    for col_num in range(len(mat[0])):
        # Append one column at a time
        rotated.append([mat[row_num][col_num] for row_num in range(len(mat))])

    return rotated

def _test_normality(data, threshold=.05):
    '''
    '''
    if len(data) < 4:
        return 0
    else:
        test_stat, pval = shapiro(data)

        return pval


def _test():
    '''
    '''
    dim1_mean = 0
    dim1_std = 3

    dim2_mean = 20
    dim2_std = 3
    # 20 normal RVs with mean=0, std=
    dim1 = list(norm.rvs(dim1_mean, dim1_std, size=20))
    # Add a couple of obvious outliers
    dim1.append(-10)
    dim1.append(10)

    dim2 = list(norm.rvs(dim2_mean, dim2_std, size=20))
    dim2.append(10)
    dim2.append(30)

    data = zip(dim1, dim2)

    confs = density_based(data)

    print 'Dim1 params:', dim1_mean, dim1_std
    print 'Dim2 params:', dim2_mean, dim2_std
    for d, conf in zip(data, confs):
        print d, conf

if __name__ == '__main__':
    _test()
