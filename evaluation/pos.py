import nltk
from nltk.corpus import brown

trigrams = {}
simple_trigrams = {}
simple_tags = set()

def catalog(sent, d):
    for (w1, t1), (w2, t2), (w3, t3) in nltk.trigrams(sent):
        d[(t1, t2, t3)] = d.get((t1, t2, t3), 0) + 1

for tagged_sent in brown.tagged_sents():
    catalog(tagged_sent, trigrams)
    
for tagged_sent in brown.tagged_sents(simplify_tags=True):
    catalog(tagged_sent, simple_trigrams)
    for word, tag in tagged_sent:
        simple_tags.add(tag)

def print_simple_tags():
    print simple_tags
    
def print_trigrams():
    pairs = zip(trigrams.values(), trigrams.keys())
    pairs.sort()
    for p in pairs:
        print p

def print_simple_trigrams():
    pairs = zip(simple_trigrams.values(), simple_trigrams.keys())
    pairs.sort()
    for p in pairs:
        print p

def unique_trigrams(d, min=24):
    print len([1 for tri, count in d.items() if count > min])

def instances_of_ith_gram(i):
    '''
    Prints the number of instance of the ith most common trigram.
    '''
    pairs = zip(trigrams.values(), trigrams.keys())
    pairs.sort(reverse=True)
    
    print pairs[i-1][0]
    

if __name__ == "__main__":
    #print_trigrams()
    #print_simple_tags()
    unique_trigrams(trigrams)
    instances_of_ith_gram(25)