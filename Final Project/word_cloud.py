
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
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
import numpy as np
from PIL import Image
from os import path
import matplotlib.pyplot as plt
import random

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
        
        
            preprocessed_sentence = ' '.join(words_in_sentences[i])
            list_of_all_preprocessed_notes_in_dataset[lt].append(preprocessed_sentence)
        
        list_of_all_preprocessed_notes_in_dataset[lt] = ' '.join(list_of_all_preprocessed_notes_in_dataset[lt])
                              
    
    
    
    
    return list_of_all_preprocessed_notes_in_dataset
            

    
data = preprocess(dataset_text)
    

Y = dataset_text[['FINAL CODES']]
Y = Y.as_matrix()

train_size = int(len(data) * 0.67)
test_size = len(data) - train_size
train_data, test_data = data[0:train_size], data[train_size:len(data)]
Y_train,Y_test = Y[0:train_size], Y[train_size:len(data)]

print(len(train_data), len(test_data))
print(len(Y_train), len(Y_test))

vectorizer = CountVectorizer(min_df=1,ngram_range=(1, 1),max_features=40)
#,max_features=10
print vectorizer

X = vectorizer.fit_transform(train_data).toarray()
print vectorizer.vocabulary_

X = np.array(X)

test_data = vectorizer.transform(test_data).toarray()
mask = np.array(Image.open("human-icon-png-1887.png"))
words = np.array(vectorizer.get_feature_names())
print words
no_urls_no_tags = " ".join([word for word in words])
wordcloud = WordCloud(mask =mask,
                      
                      stopwords=STOPWORDS,
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(no_urls_no_tags)

plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('./wordcloud.png', dpi=2000)
plt.show()