from __future__ import print_function
import os
import numpy as np
np.random.seed(7)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM
from keras.layers import Dropout
from keras.models import Model
import sys

from keras.layers.recurrent import GRU, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import pydot
import matplotlib.image as mpimg
from keras.datasets import mnist
from keras import callbacks
from keras.utils import np_utils
from keras import backend as K
import tensorflow as tf
tf.python.control_flow_ops = tf
from keras.utils.visualize_util import plot
from IPython.display import Image


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

BASE_DIR = '/Users/pulkit/Desktop/cnn'
GLOVE_DIR = BASE_DIR + '/glove.6B/'
MAX_SEQUENCE_LENGTH = 100
MAX_NB_WORDS = 150
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2


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
            
# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'vectors.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# second, prepare text samples and their labels
print('Processing text dataset')
    
texts = preprocess(dataset_text)[0]


Y = dataset_text['FINAL CODES']
labels = Y.tolist()
labels  = [int(x) for x in labels]
print(labels)
print(type(labels))

unique_labels = set(labels)           
unique_labels_count = len(unique_labels) 
print(unique_labels_count)
print('Found %s texts.' % len(texts))

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print(labels[0])
print(len(labels))
print(labels.shape)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

print('Preparing embedding matrix.')

# prepare embedding matrix
nb_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(nb_words + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

print('Training model.')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

'''
x= SimpleRNN(500,return_sequences=True)(embedded_sequences)
x= SimpleRNN(250,input_dim=500,return_sequences=True)(x)
x= SimpleRNN(250,input_dim=250,return_sequences=True)(x)
x= SimpleRNN(250,input_dim=250,return_sequences=True)(x)
x= SimpleRNN(100,input_dim=250,return_sequences=False)(x)
x = Dropout(0.2)(x)
'''

x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 2, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 1, activation='relu')(x)
x = MaxPooling1D(2)(x)
x = Flatten()(x)

x = Dense(128, activation='relu')(x)
preds = Dense(19, activation='softmax')(x)

#unique_labels_count
model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])


history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
          nb_epoch=10, batch_size=500)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Performance of the model')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper right')
plt.show()

plot(model, to_file='./model.png',show_shapes=True)


