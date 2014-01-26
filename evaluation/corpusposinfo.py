from ..shared.util import IntrinsicUtility
import nltk
import cPickle
from os import path as ospath

tris = {"current_doc":None}
path = ospath.join(ospath.dirname(__file__), "trigrams.pkl")

training_files = IntrinsicUtility().get_n_training_files()
for i in range(len(training_files)):
    print "On the", i+1, "th document of", len(training_files)
    tris["current_doc"] = i
    f = open(training_files[i])
    t = f.read()
    f.close()
    
    text = nltk.word_tokenize(t)
    tags = nltk.pos_tag(text)
    
    for (w1,t1), (w2,t2), (w3,t3) in nltk.trigrams(tags):
        if (t1, t2, t3) in tris:
            tris[(t1, t2, t3)].add(training_files[i])
        else:
            tris[(t1, t2, t3)] = set([training_files[i]])
    
    print "unsafe to quit!"
    f = open(path, "w")
    cPickle.dump(tris, f)
    f.close()
    print "safe to quit!"