import sys, math
from random import random, uniform

from ..shared.util import IntrinsicUtility
from ..shared.util import BaseUtility
from ..dbconstants import username
from ..dbconstants import password
from ..dbconstants import dbname

from plagcomps.intrinsic.cluster import cluster
from plagcomps.evaluation.intrinsic import _get_reduced_docs
from plagcomps.intrinsic.featureextraction import FeatureExtractor

sys.path.append("../PyGene/")
from pygene.prog import ProgOrganism
from pygene.population import Population

import sqlalchemy
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

url = "postgresql://%s:%s@%s" % (username, password, dbname)
engine = sqlalchemy.create_engine(url)
Base = declarative_base()
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

features = FeatureExtractor.get_all_feature_function_names()
num_training = 50 
num_testing = 500
starting_doc = 0
training_files = IntrinsicUtility().get_n_training_files(n=num_training, first_doc_num=starting_doc)
test_files = IntrinsicUtility().get_n_training_files(n=num_testing, first_doc_num=starting_doc + num_training)
cached_reduced_docs = {}
cached_confidences = {}

# set base values for globals
atom_type, cluster_type = "paragraph", "kmeans"

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
        '-':sub,
        '*': mul,
        #'/':div,
        '**': pow,
        #'sqrt': sqrt,
#        'log' : log,
#        'sin' : sin,
#        'cos' : cos,
#        'tan' : tan,
        }
    vars = [chr(ord("A") + x) for x in range(len(features))]
    consts = [-1.0] + [x/2.0 for x in range(1, 11)]

    mutProb = 0.4

    def fitness(self, training=True):
        # choose 10 random values
        badness = 0.0
        actuals = []
        confidences = []

        files = training_files if training else test_files


        try:
            for doc in get_cached_reduced_docs(atom_type, files):
                for span in doc.get_spans():
                    actuals.append(1 if doc.span_is_plagiarized(span) else 0)

                computed_feature_vector = []
                feature_vectors = doc.get_feature_vectors(features, session)
                for feature_tuple in feature_vectors:
                    ith_feature_slice = {chr(ord("A") + j):value for j,value in enumerate(feature_tuple)}
                    #print ith_feature_slice
                    computed_feature = self.calc(**ith_feature_slice)
                    computed_feature_vector.append([computed_feature])

                #print "clustering with", computed_feature_vector
                confidences += cluster(cluster_type, 2, computed_feature_vector)

            #print "Conf, Actual", confidences, actuals
            # For each document, add (1 - AUC) for our ROC calculation
            badness += (1 - BaseUtility.draw_roc(actuals, confidences, save_figure=False)[1])

            return badness
        except OverflowError:
            return 1.0e+255 # infinitely bad

        #output = ";".join([",".join([chr(ord("A") + j),value]) for j,value in enumerate(features)])

    def calc_dump(self, node):
        if hasattr(node, 'children'):
            unformatted = "(%s" + "%s%s" * (len(node.children) - 1) + ")" 
            args = [self.calc_dump(node.children[0])] + (",".join([",".join([node.name, self.calc_dump(k)]) for k in node.children[1:]])).split(",")
            if '' in args:
                args.remove('')
            return unformatted % tuple(args)
        else:
            if hasattr(node, 'value'):
                return str(node.value)
            else:
                return str(node.name)
        
    def dump(self):
        print self.calc_dump(self.tree)
        print self.tree.dump()
        return self.calc_dump(self.tree)

    # maximum tree depth when generating randomly
    initDepth = 6


# define the class comprising the program organism
class ConfidenceGeneticProgram(ProgOrganism):
    """
    """
    funcs = {
        '+': add,
        '-':sub,
        '*': mul,
        #'/':div,
        '**': pow,
        #'sqrt': sqrt,
#        'log' : log,
#        'sin' : sin,
#        'cos' : cos,
#        'tan' : tan,
        }
    vars = [chr(ord("A") + x) for x in range(len(features))]
    consts = [-1.0] + [x/2.0 for x in range(1, 11)]

    mutProb = 0.4

    def fitness(self, training=True):
        actuals = []
        confidences = []

        files = training_files if training else test_files

        try:
            for doc in get_cached_reduced_docs(atom_type, files):
                for span in doc.get_spans():
                    actuals.append(1 if doc.span_is_plagiarized(span) else 0)

                confidence_vectors = get_cached_confidences(doc)
                #print "for doc", doc, "we have", len(doc.get_spans()), "spans and", len(confidence_vectors), "confidence"

                for confidence_tuple in confidence_vectors:
                    ith_confidence_slice = {chr(ord("A") + j):value for j,value in enumerate(confidence_tuple)}
                    #print ith_confidence_slice
                    computed_confidence = self.calc(**ith_confidence_slice)
                    confidences.append(computed_confidence)

            print self.calc_dump(self.tree)
            #print "Conf, Actual", confidences, actuals
            return (1 - BaseUtility.draw_roc(actuals, confidences, save_figure=False)[1])

        except OverflowError:
            return 1.0e+255 # infinitely bad

        #output = ";".join([",".join([chr(ord("A") + j),value]) for j,value in enumerate(features)])

    def calc_dump(self, node):
        if hasattr(node, 'children'):
            unformatted = "(%s" + "%s%s" * (len(node.children) - 1) + ")" 
            args = [self.calc_dump(node.children[0])] + (",".join([",".join([node.name, self.calc_dump(k)]) for k in node.children[1:]])).split(",")
            if '' in args:
                args.remove('')
            return unformatted % tuple(args)
        else:
            if hasattr(node, 'value'):
                return str(node.value)
            else:
                return str(node.name)
        
    def dump(self):
        print self.calc_dump(self.tree)
        print self.tree.dump()
        return self.calc_dump(self.tree)

    # maximum tree depth when generating randomly
    initDepth = 6


class FeaturePop(Population):
    u"Population class for the experiment"
    species = FeatureGeneticProgram
    initPopulation = 10

    # cull to this many children after each generation
    childCull = 20

    # number of children to create after each generation
    childCount = 20

    mutants = 0.3

class ConfidencePop(Population):
    u"Population class for the experiment"
    species = ConfidenceGeneticProgram
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

def get_cached_reduced_docs(atom_type, files):
    cached_docs = cached_reduced_docs.get(atom_type, {})
    return_docs = []
    need_to_query = []
    for f in files:
        if f in cached_docs:
            return_docs.append(cached_docs[f])
        else:
            need_to_query.append(f)
    
    queried = _get_reduced_docs(atom_type, need_to_query, session)  
    for q in queried:
       return_docs.append(q)
       cached_docs[q.full_path] = q
    
    cached_reduced_docs[atom_type] = cached_docs
    return return_docs

def get_cached_confidences(doc):
    if doc in cached_confidences:
        return cached_confidences[doc]

    confidence_vectors = [ [] for x in range(len(doc.get_spans())) ] 
    feature_vectors = doc.get_feature_vectors(features, session)
    num_passages = len(feature_vectors)
    num_features = len(features)
    for feature_index in range(num_features):
        one_feature_vector = [passage_features[feature_index] for passage_features in feature_vectors]
        one_feature_confidences = cluster(cluster_type, 2, [[feature_value] for feature_value in one_feature_vector])
        for passage_index in range(num_passages):
            confidence_vectors[passage_index].append(one_feature_confidences[passage_index])

    cached_confidences[doc] = confidence_vectors
    return confidence_vectors

def test_evaluate(formula, feature_slice, feature_mapping):
    #print "feature_slice", feature_slice
    #print "feature_mapping", feature_mapping
    
    new_formula = formula
    for key in feature_mapping.keys():
        if "<" + key + ">" in new_formula:
            new_formula = new_formula.replace("<" + key + ">", "feature_slice['" + feature_mapping[key] + "']")
   
    #print "formula", new_formula
    try:
        comp = eval(new_formula)
    except OverflowError:
        comp = 1.0e+255

    #print "computed:", comp
    return comp

def feature_test(formula, feature_mapping):

    actuals = []
    confidences = []

    files = training_files + test_files

    #try:
    for doc_num,doc in enumerate(get_cached_reduced_docs("paragraph", files)):
        print str(doc_num)
        for span in doc.get_spans():
            actuals.append(1 if doc.span_is_plagiarized(span) else 0)

        computed_feature_vector = []
        feature_vectors = doc.get_feature_vectors(features, session)
        for feature_tuple in feature_vectors:
            ith_feature_slice = {features[j]:value for j,value in enumerate(feature_tuple)}
            #print ith_feature_slice
            computed_feature = test_evaluate(formula, ith_feature_slice, feature_mapping)
            computed_feature_vector.append([computed_feature])

        #print "clustering with", computed_feature_vector
        confidences += cluster("kmeans", 2, computed_feature_vector)

    #print "Conf, Actual", confidences, actuals
    # For each document, add (1 - AUC) for our ROC calculation
    return (1 - BaseUtility.draw_roc(actuals, confidences, save_figure=False)[1])
   
    #except OverflowError:
     #   return 1.0e+255 # infinitely bad

def feature_main(nfittest=10, nkids=100):

    pop = FeaturePop()

    ngens = 0
    i = 0
    while ngens < 10:
        b = pop.best()
        print "Generation %s: %s best=%s average=%s" % (
            i, repr(b), b.fitness(), pop.fitness())
        b.dump()

        i += 1
        ngens += 1
        pop.gen()

    b = pop.best()
    print "Generation %s: %s" % (
        i, repr(b))
    print "Training: best=%s average=%s" % (b.fitness(), pop.fitness())
    print "Testing: best=%s" % (b.fitness(training=False))
    b.dump()

    with open("genetic_program_outputs.txt", "a") as outfile:
        outfile.write("\n\n----\n")
        outfile.write("using " + atom_type + ", " + cluster_type + "\n")
        outfile.write("training on " + str(starting_doc) + ":" + str(starting_doc+num_training) + "\t")
        outfile.write("testing on " + str(starting_doc+num_training) + ":" + str(starting_doc+num_training + num_testing) + "\n")
        outfile.write(";".join([",".join([chr(ord("A") + j),value]) for j,value in enumerate(features)]) +"\n")
        outfile.write(b.dump())
        outfile.write("\n")
        outfile.write("Training: best=" + str(b.fitness()) + "; avg=" + str(pop.fitness())+ "\n")
        outfile.write("Testing: best=" + str(b.fitness(training=False)) + "\n")

def confidence_main(nfittest=10, nkids=100):
    pop = ConfidencePop()

    ngens = 0
    i = 0
    while ngens < 1:
        b = pop.best()
        print "Generation %s: %s best=%s average=%s" % (
            i, repr(b), b.fitness(), pop.fitness())
        b.dump()

        i += 1
        ngens += 1
        pop.gen()

    b = pop.best()
    print "Generation %s: %s" % (
        i, repr(b))
    print "Training: best=%s average=%s" % (b.fitness(), pop.fitness())
    print "Testing: best=%s" % (b.fitness(training=False))
    b.dump()

    with open("genetic_program_confidences.txt", "a") as outfile:
        outfile.write("\n\n----\n")
        outfile.write("using " + atom_type + ", " + cluster_type + "\n")
        outfile.write("training on " + str(starting_doc) + ":" + str(starting_doc+num_training) + "\t")
        outfile.write("testing on " + str(starting_doc+num_training) + ":" + str(starting_doc+num_training + num_testing) + "\n")
        outfile.write(";".join([",".join([chr(ord("A") + j),value]) for j,value in enumerate(features)]) +"\n")
        outfile.write(b.dump())
        outfile.write("\n")
        outfile.write("Training: best=" + str(b.fitness()) + "; avg=" + str(pop.fitness())+ "\n")
        outfile.write("Testing: best=" + str(b.fitness(training=False)) + "\n")

if __name__ == '__main__':


    #print feature_test("(((((1.5*<D>)-<Y>)-(-12.0**10.0))+(((<L>+10.0)+10.0)+10.0))*(10.0*(((1.5+<P>)**(-12.0-<H>))-((<X>*-12.0)+-1.0))))", {"D":"avg_internal_word_freq_class", "H" : "honore_r_measure", "L":"syntactic_complexity", "P":"pos_trigram,NN,NN,VB", "X":"pos_trigram,NN,NN,NN", "Y":"pos_trigram,NN,IN,DT"})


    for i in range(100):
        for at in ["paragraph", "nchars"]:
            for ct in ["kmeans", "outlier"]:
                atom_type, cluster_type = at, ct
                feature_main()
