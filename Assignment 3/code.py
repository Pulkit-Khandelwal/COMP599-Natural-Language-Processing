# -*- coding: utf-8 -*
"""
Created on Tue Nov 11 21:41:14 2016

@author: pulkit
"""

#import libraries
import loader
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import sys
loader.init();

stoplist = set(stopwords.words('english'))

#baseline key
def baselinekey(lemma,context):
    return wn.synsets(lemma)[0].lemmas()[0].key()

#lesk key
def leskkey(lemma,context):
    return lesk(context,lemma, 'n').lemmas()[0].key()
    
    
# system arguments to run the code

if "-baseline" in sys.argv:
    instances = loader.dev_instances
    key = loader.dev_key
    predict_key = {wsd_id:baselinekey(wi.lemma,wi.context) for wsd_id, wi in instances.iteritems()}
        
if "-lesk" in sys.argv:
    instances = loader.dev_instances
    key = loader.dev_key
    predict_key = {wsd_id:leskkey(wi.lemma,wi.context) for wsd_id, wi in instances.iteritems()}
    
if "-proposed" in sys.argv:
    syns = wn.synsets(loader.dev_instances.lemma())[0]
    for syn in syns:
			signature = transform_def(syn.definition())
    intersection = (set(context) & set(signature))
    print intersection

        
#perform lemmatization and remove words from stop list

def lemmatization(tokens):
    lm=WordNetLemmatizer();
    for i in range(0,len(tokens)):
        tokens[i]=lm.lemmatize(tokens[i])
    return tokens


def stopwords(context):
    var=[]
    for x in context:
        if x not in stoplist:
            var.append(x)
    return var
    
#metrics

preds=len([wsd_id for wsd_id, sense in predict_key.iteritems() if wsd_id in key and sense in key[wsd_id]])
precision=float(preds)/len([sense for _, sense in predict_key.iteritems() if sense is not None])
recall=float(preds)/len(predict_key)
f1=(float(2)*precision*recall)/(precision+recall)

print "Calculations"
print "Precision",precision
print "Recall",recall
print "F1 Score",f1
