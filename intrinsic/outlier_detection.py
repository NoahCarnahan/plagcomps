from numpy import mean, std
from scipy.stats import norm

def density_based(stylo_vectors, impurity=.2):
    transpose = _rotate_vectors(stylo_vectors)

    # Could speed this up with just one (two?) passes over the data
    # means[i] is the mean of the i_th feature
    means = [mean(row) for row in transpose]
    stds = [std(row) for row in transpose]

    mins = [min(row) for row in transpose]
    maxs = [max(row) for row in transpose]

    for i in xrange(len(stylo_vectors)):
        vec = stylo_vectors[i]
        plag_prob = impurity
        non_plag_prob = (1 - impurity)

        for feat_num in xrange(len(vec)):
            plag_prob *= _get_norm_prob(vec[feat_num], means[feat_num], stds[feat_num])
            non_plag_prob *= _get_unif_prob(vec[feat_num], mins[feat_num], maxs[feat_num])

        if plag_prob > non_plag_prob:
            print 'Plag!'
        else:
            print 'Think it\'s all good...'
        print plag_prob, non_plag_prob
        print '-'*40





    
def _get_norm_prob(x, loc, scale):
    return norm.pdf(x, loc, scale)

def _get_unif_prob(v, min_val, max_val):
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

