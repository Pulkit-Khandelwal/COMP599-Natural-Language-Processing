import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
import pandas
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE

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

sm = SMOTE(kind='regular')



dataset_text = pandas.read_csv('data.csv')


Y = dataset_text[['FINAL CODES']]

Y = np.ravel(Y)
print Y.shape

from scipy.stats import itemfreq

print itemfreq(Y)


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


vectorizer = CountVectorizer(min_df=1,ngram_range=(1, 1))
X = vectorizer.fit_transform(data).toarray()
X = np.array(X)

#SMOTE ALGORITHM

num_classes = 18
for i in range(num_classes-1):
    X_resampled, y_resampled = sm.fit_sample(X, Y)
    X, Y = X_resampled, y_resampled
    
print X.shape
print Y.shape
    
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
train_data, test_data = X[0:train_size], X[train_size:len(X)]
Y_train,Y_test = Y[0:train_size], Y[train_size:len(Y)]

print(len(train_data), len(test_data))
print(len(Y_train), len(Y_test))



expected = Y_test

#Naive Bayes
gnb = GaussianNB()
gnb.fit(train_data,Y_train)
predicted_naive = gnb.predict(test_data)

print "Naive Bayes Classification Report",(metrics.classification_report(expected, predicted_naive))
print "Naive Bayes Confusion Matrix",(metrics.confusion_matrix(expected, predicted_naive))
print "Naive Bayes Accuracy",(metrics.accuracy_score(expected, predicted_naive))



#Logistic Regression
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(train_data,Y_train)
predicted_logreg = logreg.predict(test_data)

print "Logistic Regression Classification Report",(metrics.classification_report(expected, predicted_logreg))
print "Logistic Regression Confusion Matrix",(metrics.confusion_matrix(expected, predicted_logreg))
print "Logistic Regression Accuracy",(metrics.accuracy_score(expected, predicted_logreg))


#SVM
sv = svm.SVC()
sv.fit(X,Y_train)
predicted_svm = sv.predict(test_data)

print "SVM Classification Report",(metrics.classification_report(expected, predicted_svm))
print "SVM Confusion Matrix",(metrics.confusion_matrix(expected, predicted_svm))
print "SVM Accuracy",(metrics.accuracy_score(expected, predicted_svm))

#QDA

qd = QDA()
qd.fit(X,Y_train)
predicted_qd = qd.predict(test_data)
print "QDA Classification Report",(metrics.classification_report(expected, predicted_qd))
print "QDA Confusion Matrix",(metrics.confusion_matrix(expected, predicted_qd))
print "QDA Accuracy",(metrics.accuracy_score(expected, predicted_qd))

#LDA

ld = LDA()
ld.fit(X,Y_train)
predicted_ld = ld.predict(test_data)
print "LDA Classification Report",(metrics.classification_report(expected, predicted_ld))
print "LDA Confusion Matrix",(metrics.confusion_matrix(expected, predicted_ld))
print "LDA Accuracy",(metrics.accuracy_score(expected, predicted_ld))


#Neural Network

mlp = MLPClassifier(solver='adam', alpha=1e-5,
                    hidden_layer_sizes=(10,20), random_state=1, tol=0.0000000001, max_iter=10000)
mlp.fit(X,Y_train)
predicted_nn = mlp.predict(test_data)
print "NN Classification Report",(metrics.classification_report(expected, predicted_nn))
print "NN Confusion Matrix",(metrics.confusion_matrix(expected, predicted_nn))
print "NN Accuracy",(metrics.accuracy_score(expected, predicted_nn))

#Ensemble Methods

from sklearn.ensemble import VotingClassifier
eclf = VotingClassifier(estimators=[
        ('svm', sv), ('gnb', gnb), ('logreg',logreg), ('ld',ld) ,('mlp',mlp)],
        voting='hard')

eclf.fit(X,Y_train)
predicted_eclf = eclf.predict(test_data)

print "VotingClassifier Classification Report",(metrics.classification_report(expected, predicted_eclf))
print "VotingClassifier Confusion Matrix",(metrics.confusion_matrix(expected, predicted_eclf))
print "VotingClassifier Accuracy",(metrics.accuracy_score(expected, predicted_eclf))

