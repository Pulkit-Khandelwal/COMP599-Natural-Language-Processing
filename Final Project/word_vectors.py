from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import pandas
from scipy.stats import mode
import codecs
import sys
import re
import nltk
from nltk import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
import glob
import csv
import itertools
import gensim

from sklearn.lda import LDA
from sklearn.qda import QDA
import numpy as np
import matplotlib.pyplot as plt
import pandas
from scipy.stats import mode
import codecs
import sys
import re
import nltk
from nltk import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
import glob
import csv
import itertools
import gensim
import codecs
import sys
import re
import nltk
from nltk import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
import glob
import sys, os, codecs
import sklearn
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn import svm, linear_model, naive_bayes
from sklearn.feature_extraction.text import CountVectorizer
import math
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

dataset_text = pandas.read_csv('data.csv')

def preprocess(dataset_text):
    dataset_text_yo = dataset_text['TEXT'].tolist()
    
    list_of_all_notes = []
    for i in range(len(dataset_text_yo)):
        sentences = sent_tokenize(dataset_text_yo[i])
        
        list_of_all_notes.append(sentences)
   
    def get_wordnet_pos(treebank_tag):
    
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None
        
    def ispunct(some_string):
        return not any(char.isalnum() for char in some_string)
    
    def init_list_of_objects(size):
        
        list_of_objects = list()
        for i in range(0,size):
             list_of_objects.append( list() ) #different object reference each time
        return list_of_objects
        
    list_of_all_preprocessed_notes_in_dataset = init_list_of_objects(len(dataset_text_yo))
    words_in_each_document = init_list_of_objects(len(dataset_text_yo))
    for lt in range(len(list_of_all_notes)):
        note = list_of_all_notes[lt]
        words_in_sentences = [w.lower() for w in note]
        words_in_sentences = [word_tokenize(t) for t in note]
        
        for i, w in enumerate(words_in_sentences):
            pos = pos_tag(w)
        
            pos_tags_only = [s[1] for s in pos]
        
        
            all_words =[]
            count = 0
            for word in w:
            
                pos_tree = get_wordnet_pos(pos_tags_only[count])
                if pos_tree:
                    lemma = nltk.stem.WordNetLemmatizer().lemmatize(word,pos_tree)
                else:  
                    lemma = nltk.stem.WordNetLemmatizer().lemmatize(word)
           
            
                all_words.append(lemma)
                count = count + 1
                    
       
            words_in_sentences[i] = [k for k in all_words if k not in stopwords.words('english') and not ispunct(k)]
        
            words_in_each_document[lt].append(words_in_sentences[i])
            preprocessed_sentence = ' '.join(words_in_sentences[i])
            list_of_all_preprocessed_notes_in_dataset[lt].append(preprocessed_sentence)
        
        list_of_all_preprocessed_notes_in_dataset[lt] = ' '.join(list_of_all_preprocessed_notes_in_dataset[lt])
        
        final_list_words = []
        for k in range(len(words_in_each_document)):
            final_list_words.append(list(itertools.chain.from_iterable(words_in_each_document[k])))
        
                                
        
    
    
    
    return list_of_all_preprocessed_notes_in_dataset,words_in_each_document,final_list_words
            

    
data = preprocess(dataset_text)[0]
data = pandas.read_csv('data_11000.csv')
data = dataset_text[:2000]
data = (data['TEXT']).tolist()
text_in_one_chunk = list(itertools.chain.from_iterable(data))
thefile = open('text_chunk.txt', 'w')

for item in data:
  thefile.write("%s\n" % item)


with open('text_chunk.txt', 'r') as myfile:
    data=myfile.read().replace('\n', '')
    
sentences = sent_tokenize(data)
words_in_sentences = [word_tokenize(t) for t in sentences]
print words_in_sentences

model = gensim.models.Word2Vec(words_in_sentences, size=20, window=5)
#, min_count=5, workers=4
print model.similarity('chest', 'scan')
print model['chest']
print model['scan']



