# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 00:13:35 2016

@author: pulkit
"""

#import all the required libraries and modules
import numpy as np
import os
import sys
import nltk
import string
import re
from nltk.probability import *

#declare variables which are used for command line sys arguments
laplace = 0
lm = 0

#check the command line for arguments: laplace, lm and the folder
if "-laplace" in sys.argv:
    laplace = 1
if "-lm" in sys.argv:
    lm = 1
    
folder = sys.argv[len(sys.argv)-1]

#29 symbols and states for the corpus
symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 
'i', 'j', 'k', 'l', 'm', 'n', 
'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 
'w', 'x', 'y', 'z','.',',','']

states = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 
'i', 'j', 'k', 'l', 'm', 'n', 
'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 
'w', 'x', 'y', 'z','.',',','']

#HMM trainer initiliazed

trainer = nltk.tag.hmm.HiddenMarkovModelTrainer(symbols, states)

#read all the traing and testing files from all the folders

train_cipher = open("a2data/"+folder+"/" + "/train_cipher.txt").read()
train_cipher = train_cipher.strip()
train_cipher = train_cipher.split('\r\n')
train_plain = open("a2data/"+folder+"/" + "/train_plain.txt").read()
train_plain = train_plain.strip()
train_plain = train_plain.split('\r\n')

test_cipher = open("a2data/"+folder+"/" + "/test_cipher.txt").read()
test_cipher = test_cipher.strip()
test_cipher = test_cipher.split('\r\n')
test_plain = open("a2data/"+folder+"/" + "/test_plain.txt").read()
test_plain = test_plain.strip()
test_plain = test_plain.split('\r\n')

#raed a1 data from the given text file "full_file.txt"
text_full = open('full_file.txt').read()
text_full = text_full.lower()
text_full = re.sub(r'[-+=:;/{\|}!?["^@]#$%^&*()]', '', text_full)
text_full = text_full.strip()
text_full = text_full.split('\r\n')

#separate the data into individual characters and then store as a list of tuples by using zip function

train = []
test = []


for i in range(len(train_cipher)):
    if i+1 < len(train_cipher):
        train.append(zip(train_cipher[i],train_plain[i]))
       
for i in range(len(test_cipher)):
    test.append(zip(test_cipher[i],test_plain[i]))

#If lm is used then also use the bigram transitional probabibilites from a1 data
if lm == 1:
    for i in range(len(text_full)):
        if i+1 < len(text_full):
            train.append(zip(text_full[i],text_full[i-1]))

#check what command line sys argument has been passed and proceeed acccordingly
if laplace == 0 and lm == 0:
    tagged = trainer.train_supervised(train)
elif laplace == 1 and lm == 0:
    tagged = trainer.train_supervised(train, estimator=LaplaceProbDist)
elif laplace == 0 and lm == 1:
    tagged = trainer.train_supervised(train)
elif laplace == 1 and lm == 1:
    tagged = trainer.train_supervised(train, estimator=LaplaceProbDist)
  
#prints the accuracy
tagged.test(test)

#gives the tagged output   
                                            
for i in range(len(test_cipher)):
    print tagged.tag(test_cipher[i])

                                                   
                                                   
                                                   
                                                   
                                                   
                                                   
                                                   
                                                   
                                                   
                                                   
                                                   
                                                   
                                                   
                                                   
                                                   
                                                   
                                                   
                                                   
                                                   
                                                   
                                                   
                                                   
                                                   
                                                   
