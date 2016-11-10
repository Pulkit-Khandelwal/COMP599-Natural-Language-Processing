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

#all libraries and modules used have been called above

#training data: 2010 text corpus

#store the text data as a list
list_of_files = []

year = 2010
#change the path accordingly
sub_folder = '../data/tac%s' % year
template = 'tac10-%04d.txt'

for i in range(1,921):
    fname = os.path.join(sub_folder, template % i)
    list_of_files.append(fname)

#the training data from al the files is stored as a list
array_files = []
for ele in list_of_files:
    a = codecs.open(ele, 'r', encoding = 'utf-8').read()
    a = codecs.encode(a, 'ascii', 'ignore')
    array_files.append(a)

# CountVectorizer is used to extract features
vectorizer = CountVectorizer(min_df=1,ngram_range=(1, 1),max_features=200)
print vectorizer

X = vectorizer.fit_transform(array_files).toarray()

Y = []

#extract the labels and store it as a list
labels_f = 'tac%s.labels' % year
fh = open(os.path.join(sub_folder, labels_f))
for line in fh:
	docid, label = line.split()
	Y.append(int(label))

X = np.array(X)
Y = np.array(Y)

# shuffle the data
k = np.zeros((920,201))
k = np.insert(X, 200, Y, axis=1)
np.random.shuffle(k)

# validation set: 20 percent of the training data
k_val = k[:184,:]
# use k_val for validating after training

#train the models on the training dataset and check the evaluation metrics on
#both the training and validation set

#naive bayes

gnb = GaussianNB()
gnb.fit(k[:,0:-1],k[:,-1])
expected = k[:,-1]
predicted_naive = gnb.predict(k[:,0:-1])

print "Naive Bayes Classification Report",(metrics.classification_report(expected, predicted_naive))
print "Naive Bayes Confusion Matrix",(metrics.confusion_matrix(expected, predicted_naive))

#logistic regression


logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(k[:,0:-1],k[:,-1])
expected= k[:,-1]
predicted_logreg = logreg.predict(k[:,0:-1])

print "Logistic Regression Classification Report",(metrics.classification_report(expected, predicted_logreg))
print "Logistic Regression Confusion Matrix",(metrics.confusion_matrix(expected, predicted_logreg))

#SVM


sv = svm.SVC()
sv.fit(k[:,0:-1],k[:,-1])
expected= k[:,-1]
predicted_svm = sv.predict(k[:,0:-1])

print "SVM Classification Report",(metrics.classification_report(expected, predicted_svm))
print "SVM Confusion Matrix",(metrics.confusion_matrix(expected, predicted_svm))

#testing data
#create testing data in a similar way tp the training data

list_test = []

year_test = 2011
sub_folder_test = '../data/tac%s' % year_test
template_test = 'tac11-%04d.txt'

for i in range(921,1801):
    fname_test = os.path.join(sub_folder_test, template_test % i)
    list_test.append(fname_test)

array_files_test = []
for ele in list_test:
    a_test = codecs.open(ele, 'r', encoding = 'utf-8').read()
    a_test = codecs.encode(a_test, 'ascii', 'ignore')
    array_files_test.append(a_test)

array_files_test = np.array(array_files_test)

Y_test = []
labels_f_test = 'tac%s.labels' % year_test
fh_test = open(os.path.join(sub_folder_test, labels_f_test))
for line_test in fh_test:
	docid_test, label_test = line_test.split()
	Y_test.append(int(label_test))

#use vectorizer.transform to use the features (vocab list) obtained from the training data
#and then fit the model

testdata = vectorizer.transform(array_files_test).toarray()


#calculate the evaluation metrics on the test set


#naive bayes

gnb = GaussianNB()
gnb.fit(k[:,0:-1],k[:,-1])
expected = Y_test
predicted_naive = gnb.predict(testdata)

print "Naive Bayes Classification Report",(metrics.classification_report(expected, predicted_naive))
print "Naive Bayes Confusion Matrix",(metrics.confusion_matrix(expected, predicted_naive))
print "Naive Bayes Accuracy",(metrics.accuracy_score(expected, predicted_naive))

#logistic regression

logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(k[:,0:-1],k[:,-1])
expected= Y_test
predicted_logreg = logreg.predict(testdata)

print "Logistic Regression Classification Report",(metrics.classification_report(expected, predicted_logreg))
print "Logistic Regression Confusion Matrix",(metrics.confusion_matrix(expected, predicted_logreg))
print "Naive Logistic Regression Accuracy",(metrics.accuracy_score(expected, predicted_logreg))

#SVM

sv = svm.SVC()
sv.fit(k[:,0:-1],k[:,-1])
expected= Y_test
predicted_svm = sv.predict(testdata)

print "SVM Classification Report",(metrics.classification_report(expected, predicted_svm))
print "SVM Confusion Matrix",(metrics.confusion_matrix(expected, predicted_svm))
print "Naive SVM Accuracy",(metrics.accuracy_score(expected, predicted_svm))
