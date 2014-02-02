import sys
import math
from random import random, uniform

from ..shared.util import IntrinsicUtility
from ..shared.util import BaseUtility
from ..dbconstants import username
from ..dbconstants import password
from ..dbconstants import dbname
from plagcomps.intrinsic.cluster import cluster

sys.path.append("../../PyGene/")
from pygene.prog import ProgOrganism
from pygene.population import Population

import sqlalchemy
from sqlalchemy.orm import sessionmaker

"""
Demo of genetic programming

This gp setup seeks to breed an organism which
implements func x^2 + y

Takes an average of about 40 generations
to breed a matching program
"""

features = ["num_chars", "avg(num_chars)"]
session = Session()

# a tiny batch of functions
def add(x,y):
    #print "add: x=%s y=%s" % (repr(x), repr(y))
    try:
        return x+y
    except:
        #raise
        return x

def sub(x,y):
    #print "sub: x=%s y=%s" % (repr(x), repr(y))
    try:
        return x-y
    except:
        #raise
        return x

def mul(x,y):
    #print "mul: x=%s y=%s" % (repr(x), repr(y))
    try:
        return x*y
    except:
        #raise
        return x

def div(x,y):
    #print "div: x=%s y=%s" % (repr(x), repr(y))
    try:
        return x / y
    except:
        #raise
        return x

def sqrt(x):
    #print "sqrt: x=%s" % repr(x)
    try:
        return math.sqrt(x)
    except:
        #raise
        return x

def pow(x,y):
    #print "pow: x=%s y=%s" % (repr(x), repr(y))
    try:
        return x ** y
    except:
        #raise
        return x

def log(x):
    #print "log: x=%s" % repr(x)
    try:
        return math.log(float(x))
    except:
        #raise
        return x

def sin(x):
    #print "sin: x=%s" % repr(x)
    try:
        return math.sin(float(x))
    except:
        #raise
        return x

def cos(x):
    #print "cos: x=%s" % repr(x)
    try:
        return math.cos(float(x))
    except:
        #raise
        return x

def tan(x):
    #print "tan: x=%s" % repr(x)
    try:
        return math.tan(float(x))
    except:
        #raise
        return x

# define the class comprising the program organism
class FeatureGeneticProgram(ProgOrganism):
    """
    """
    funcs = {
        '+': add,
#        '-':sub,
        '*': mul,
#        '/':div,
#        '**': pow,
#        'sqrt': sqrt,
#        'log' : log,
#        'sin' : sin,
#        'cos' : cos,
#        'tan' : tan,
        }
    vars = features
    consts = [0.0, 2.0, 18.0, 13.0]

    num_training = 10
    num_testing = 25
    training_files = IntrinsicUtility().get_n_training_files(n=num_training)
    test_files = IntrinsicUtility().get_n_training_files(n=num_testing, first_doc_num=num_training)

    testVals = [{'x':uniform(-10.0, 10.0),
                 'y':uniform(-10.0, 10.0),
                 } for i in xrange(20)
                ]

    mutProb = 0.4

    def fitness(self):
        # choose 10 random values
        badness = 0.0
        actuals = []
        try:
            for doc in training_files:
                for span in doc.get_spans():
                    actuals.append(1 if doc.span_is_plagiarized(span) else 0)

                computed_feature_vector = []
                feature_vectors = doc.get_feature_vectors(features, session)
                for i in range(len(feature_vectors[0])):
                    ith_feature_slice = [f[i] for f in feature_vectors]
                    computed_feature = self.calc(**ith_feature_slice)
                    computed_feature_vector.append(computed_feature)

                confidences = cluster("outlier", 2, [computed_feature_vector])

                # For each document, add (1 - AUC) for our ROC calculation
                badness += (1 - BaseUtility.draw_roc(actuals, confidences, save_figure=False)[1])

            return badness
        except OverflowError:
            return 1.0e+255 # infinitely bad

    # maximum tree depth when generating randomly
    initDepth = 6


class ProgPop(Population):
    u"Population class for the experiment"
    species = FeatureGeneticProgram
    initPopulation = 10

    # cull to this many children after each generation
    childCull = 20

    # number of children to create after each generation
    childCount = 20

    mutants = 0.3

def graph(orig, best):
    print "ORIG                                  BEST:"
    for y in range(10, -11, -2):
        for x in range(-10, 11, 3):
            z = orig(x=float(x), y=float(y))
            print "%03.0f " % z,

        print "  ",
        for x in range(-10, 11, 3):
            z = best(x=float(x), y=float(y))
            print "%03.0f " % z,
        print


def main(nfittest=10, nkids=100):

    extractor = FeatureExtractor()

    pop = ProgPop()

    ngens = 0
    i = 0
    while ngens < 100:
        b = pop.best()
        print "Generation %s: %s best=%s average=%s)" % (
            i, str(b), b.fitness(), pop.fitness())
        b.dump()

        #graph(b.testFunc, b.calc)

        ngens += 1

    print "Generation %s: %s best=%s average=%s)" % (
        i, str(b), b.fitness(), pop.fitness())

if __name__ == '__main__':
    main()
    pass


