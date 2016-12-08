from __future__ import division

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 01:51:22 2016

@author: pulkit
Implementation of the sumbasic algorithm and its variants for Automatic Summarization of Text.

"""

# import the required libraries
import codecs
import sys
import re
import nltk
from nltk import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
import glob

# arguments for command line input

folder = sys.argv[2]
use_this_1 = sys.argv[1]
use_this_2 = int(re.search(r'\d+', folder).group())
joined_string = use_this_1 + '-' + str(use_this_2)


# function which does pre-processing and calculate unigram probabilites
def pulks(sentences):
    
    #get the POS tags for lemmatization
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
            
    d = {}
    preprocessed_text = []

    #convert to lower case
    words_in_sentences = [w.lower() for w in sentences] 
    #tokenize the sentence to get words
    words_in_sentences = [word_tokenize(t) for t in sentences] 
    
                          
    #remove punctuations
    #lemmatize each word
    
    def ispunct(some_string):
        return not any(char.isalnum() for char in some_string)
        
    for i, w in enumerate(words_in_sentences):
        pos = pos_tag(w)
        
        pos_tags_only = [s[1] for s in pos]
        
        
        yo =[]
        count = 0
        for word in w:
            
            pos_tree = get_wordnet_pos(pos_tags_only[count])
            if pos_tree:
                man = nltk.stem.WordNetLemmatizer().lemmatize(word,pos_tree)
            else:  
                man = nltk.stem.WordNetLemmatizer().lemmatize(word)
           
            
            yo.append(man)
            count = count + 1
            
        #remove stopwords
        words_in_sentences[i] = [k for k in yo if k not in stopwords.words('english') and not ispunct(k)]
        
        
        preprocessed_sentence = ' '.join(words_in_sentences[i])
        preprocessed_text.append(preprocessed_sentence)
        d[i] = [sentences[i], preprocessed_sentence]
    
    preprocessed_text = ' '.join(preprocessed_text)
    preprocessed_text = nltk.word_tokenize(preprocessed_text)
    
    
    my_unigrams = nltk.FreqDist(preprocessed_text)
    unigrams_count_word = my_unigrams.items()
    lexicon_size = len(preprocessed_text)
    
    #probabilites of each word- unigrams calculated
    
    counts = [s[1] for s in unigrams_count_word]
    wordz = [s[0] for s in unigrams_count_word]
    for i in range(len(counts)):
        counts[i]  = float(counts[i] )/ float(lexicon_size)
    
    
    lolo =[]
    for i in range(len(d)):
        k = d[i]
        
        lolo.append(k[1])
   
    return lolo,sentences,counts,wordz

#######

# Compute three different Automatic Summarization Algorithms

#######

#simplified verison of sumbasic algorithm without non-redundancy update
if "simplified" in sys.argv[1]:
    
    documents = glob.glob(sys.argv[2])
    
 
    print "Summarizing all documents of the given cluster"
    cluster_summary = []
    for doc in range(len(documents)):
        
        sentences = codecs.open(documents[doc] , 'r', encoding='utf-8').read()
        sentences = codecs.encode(sentences, 'ascii', 'ignore')
        sentences = sent_tokenize(sentences)
        
        
        lolo,sentences,counts,wordz = pulks(sentences)
        #calculate average word probabilities of each sentence
        avg_word_probab_sentence = []
        for i in range(len(lolo)):
            words = lolo[i].split()
            sum_probab_sentence = 0
            for w in words:
                k = wordz.index(w)
                
                probab_word = counts[k]
                
                sum_probab_sentence += probab_word
            avg_word_probab_sentence.append(float(sum_probab_sentence)/ float(len(words)))
                
        #rank sentences by their avergae word probabilities
        
        probab_original = zip(avg_word_probab_sentence,sentences,lolo)
        probab_original_sorted = sorted(probab_original, key=lambda tup: tup[0],reverse=True)
        original_sentences_only_sorted = [s[1] for s in probab_original_sorted]
        preprocess_sentences_only_sorted = [s[2] for s in probab_original_sorted]
        
        summary = []
        total_words = 0
        summary_length = 100
        
        for i in range(len(original_sentences_only_sorted)):
            k = original_sentences_only_sorted[i].split()
            l = preprocess_sentences_only_sorted[i].split()
    
            
            total_words = total_words + len(k)
            if total_words < summary_length:
                summary.append(original_sentences_only_sorted[i])
        summary_text = ' '.join(summary)
        word_count_summary = summary_text.split()
        
        
        #save the entire cluster summary
        
        cluster_summary.append(summary_text)
    
    sum_up = ' '.join(cluster_summary)
    
    print "Placing all the summaries of the given cluster in one file"
    text_file = open("%s.txt" % joined_string, "w")
    text_file.write(sum_up)
    text_file.close()
  

# Original SumBasic algorithm

if "orig" in sys.argv[1]:
    documents = glob.glob(sys.argv[2])
    
    print "Summarizing all documents of the given cluster"
    cluster_summary = []
    for doc in range(len(documents)):
        
        sentences = codecs.open(documents[doc] , 'r', encoding='utf-8').read()
        sentences = codecs.encode(sentences, 'ascii', 'ignore')
        sentences = sent_tokenize(sentences)
        
        #calculate average word probabilities of each sentence and rank the sentences
        lolo,sentences,counts,wordz = pulks(sentences)
        avg_word_probab_sentence = []
        for i in range(len(lolo)):
            words = lolo[i].split()
            sum_probab_sentence = 0
            for w in words:
                k = wordz.index(w)
                
                probab_word = counts[k]
                
                sum_probab_sentence += probab_word
            avg_word_probab_sentence.append(float(sum_probab_sentence)/ float(len(words)))
                
        
        probab_original = zip(avg_word_probab_sentence,sentences,lolo)
        probab_original_sorted = sorted(probab_original, key=lambda tup: tup[0],reverse=True)
        original_sentences_only_sorted = [s[1] for s in probab_original_sorted]
        preprocess_sentences_only_sorted = [s[2] for s in probab_original_sorted]
        
        summary = []
        total_words = 0
        summary_length = 100
        
        for i in range(len(original_sentences_only_sorted)):
            k = original_sentences_only_sorted[i].split()
            l = preprocess_sentences_only_sorted[i].split()
        
                
            total_words = total_words + len(k)
            if total_words < summary_length:
                summary.append(original_sentences_only_sorted[i])
          
            for s in l:
                m = wordz.index(s)
                counts[m] = counts[m] * counts[m]
        
        
            avg_word_probab_sentence = []
            for i in range(len(preprocess_sentences_only_sorted)):
                
                words = preprocess_sentences_only_sorted[i].split()
                sum_probab_sentence = 0
                for w in words:
                    k = wordz.index(w)
                
                    probab_word = counts[k]
                
                    sum_probab_sentence += probab_word
                    avg_word_probab_sentence.append(float(sum_probab_sentence)/ float(len(words)))
            probab_original = zip(avg_word_probab_sentence,sentences,preprocess_sentences_only_sorted)
            probab_original_sorted = sorted(probab_original, key=lambda tup: tup[0],reverse=True)
            original_sentences_only_sorted = [s[1] for s in probab_original_sorted]
            preprocess_sentences_only_sorted = [s[2] for s in probab_original_sorted]
        
        
                
        #save the summaries of the entire cluster
        summary_text = ' '.join(summary)
        word_count_summary = summary_text.split()
        
        cluster_summary.append(summary_text)
   
    sum_up = ' '.join(cluster_summary)
    
    print "Placing all the summaries of the given cluster in one file"
    text_file = open("%s.txt" % joined_string, "w")
    text_file.write(sum_up)
    text_file.close()
    
    
if "leading" in sys.argv[1]:
    def pulkit(sentences):
        
        
        sentences = sent_tokenize(sentences)
    
    
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
            
        d = {}
        preprocessed_text = []
        
        words_in_sentences = [w.lower() for w in sentences] #lower case
        words_in_sentences = [word_tokenize(t) for t in sentences] #word tokenize
        
                              
        #remove punctuation
        #lemmatize
        def ispunct(some_string):
            return not any(char.isalnum() for char in some_string)
        
        for i, w in enumerate(words_in_sentences):
            pos = pos_tag(w)
            
            pos_tags_only = [s[1] for s in pos]
            
            
            yo =[]
            count = 0
            for word in w:
                
                pos_tree = get_wordnet_pos(pos_tags_only[count])
                if pos_tree:
                    man = nltk.stem.WordNetLemmatizer().lemmatize(word,pos_tree)
                else:  
                    man = nltk.stem.WordNetLemmatizer().lemmatize(word)
               
                
                yo.append(man)
                count = count + 1
        
            words_in_sentences[i] = [k for k in yo if k not in stopwords.words('english') and not ispunct(k)]
            
            
            preprocessed_sentence = ' '.join(words_in_sentences[i])
            preprocessed_text.append(preprocessed_sentence)
            d[i] = [sentences[i], preprocessed_sentence]
        
        preprocessed_text = ' '.join(preprocessed_text)
        preprocessed_text = nltk.word_tokenize(preprocessed_text)
    
    
        my_unigrams = nltk.FreqDist(preprocessed_text)
        unigrams_count_word = my_unigrams.items()
        lexicon_size = len(preprocessed_text)
        
        #probabilites of each word
        
        counts = [s[1] for s in unigrams_count_word]
        wordz = [s[0] for s in unigrams_count_word]
        for i in range(len(counts)):
            counts[i]  = float(counts[i] )/ float(lexicon_size)
        
       
        #rank sentences by their avergae word probabilities
        
        lolo =[]
        for i in range(len(d)):
            k = d[i]
            
            lolo.append(k[1])
        #print lolo
        #calculate average word probabilities of each sentence and rank the sentences
    
    
        avg_word_probab_sentence = []
        for i in range(len(lolo)):
            words = lolo[i].split()
            sum_probab_sentence = 0
            for w in words:
                k = wordz.index(w)
                
                probab_word = counts[k]
                
                sum_probab_sentence += probab_word
            avg_word_probab_sentence.append(float(sum_probab_sentence)/ float(len(words)))
            
    
        probab_original = zip(avg_word_probab_sentence,sentences,lolo)
        probab_original_sorted = sorted(probab_original, key=lambda tup: tup[0],reverse=True)
        original_sentences_only_sorted = [s[1] for s in probab_original_sorted]
        preprocess_sentences_only_sorted = [s[2] for s in probab_original_sorted]
        
        summary = []
        total_words = 0
        summary_length = 100
    
        for i in range(len(original_sentences_only_sorted)):
            k = original_sentences_only_sorted[i].split()
            l = preprocess_sentences_only_sorted[i].split()
        
                
            total_words = total_words + len(k)
            if total_words < summary_length:
                summary.append(original_sentences_only_sorted[i])
          
            for s in l:
                m = wordz.index(s)
                counts[m] = counts[m] * counts[m]
        
        
            avg_word_probab_sentence = []
            for i in range(len(preprocess_sentences_only_sorted)):
                
                words = preprocess_sentences_only_sorted[i].split()
                sum_probab_sentence = 0
                for w in words:
                    k = wordz.index(w)
                
                    probab_word = counts[k]
                
                    sum_probab_sentence += probab_word
                    avg_word_probab_sentence.append(float(sum_probab_sentence)/ float(len(words)))
            probab_original = zip(avg_word_probab_sentence,sentences,preprocess_sentences_only_sorted)
            probab_original_sorted = sorted(probab_original, key=lambda tup: tup[0],reverse=True)
            original_sentences_only_sorted = [s[1] for s in probab_original_sorted]
            preprocess_sentences_only_sorted = [s[2] for s in probab_original_sorted]
    
    
            
        
        summary_text = ' '.join(summary)
        word_count_summary = summary_text.split()
       
        return summary[0]
    print "Summarizing all documents of the given cluster"
    leading_summary = []
    documents = glob.glob(sys.argv[2])
    for i in range(len(documents)):
        sentences = codecs.open(documents[i], 'r', encoding='utf-8').read()
        sentences = codecs.encode(sentences, 'ascii', 'ignore')
        leading_summary.append(pulkit(sentences))
    
    leading_summary = ' '.join(leading_summary)
    #save the summaries of the cluster
    print "Placing all the summaries of the given cluster in one file"
    text_file = open("%s.txt" % joined_string, "w")
    text_file.write(leading_summary)
    text_file.close()

