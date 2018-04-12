#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 19:29:29 2018

@author: arnav
"""

import numpy as np
import itertools
import utils
import time
import nltk
import csv
import sys
import os
from RNNLM import RNNLanguageModel

nltk.download("book")

_Vocab_size = int(os.environ.get('Vocab_size', '8000'))
_Hidden = int(os.environ.get('Hidden', '80'))
_Alpha = float(os.environ.get('Alpha', '0.005'))
_Epochs = int(os.environ.get('Epochs', '100'))
_Model = os.environ.get('Model')

def train(model, train_data, train_label, alpha = 0.005, epochs = 100, loss_eval_after = 5):
    Losses = []
    till_now_examples = 0
    
    for i in range(epochs):
        if(i%loss_eval_after == 0):
            l = model.loss(train_data, train_label)
            Losses.append((till_now_examples, l))
            print "Value of loss after training examples = %d, epochs = %d is loss = %f" % (till_now_examples, epochs, l)
            
            if (len(Losses) > 1 and Losses[-1][1] > Losses[-2][1]):
                alpha = alpha * 0.5
                print "New value to learning rate set to %f" % alpha
            
            sys.stdout.flush()
            
            utils.save_param("./models/RNN_training_model.npz", model)
            
        for i in range(len(train_label)):
            model.grad_descent(train_data[i], train_label[i], alpha)
            till_now_examples += 1
            
            
vocabulary_size = _Vocab_size
not_known = "UNKNOWN"
start_of_sentence = "SENTENCE_START"
end_of_sentence = "SENTENCE_END"

print "Opening the input file..."
with open('models/google_reddit_chat.csv', 'rb') as file_descriptor:
    file_reader = csv.reader(file_descriptor, skipinitialspace=True)
    file_reader.next()
    
    #Complete chats are splitted into sentences
    Sen = itertools.chain(*[nltk.sent_tokenize(fr[0].decode('utf-8').lower()) for fr in file_reader])
    
    #Join the start and end of sentences
    Sen = ["%s %s %s" % (start_of_sentence, i, end_of_sentence) for i in Sen]

print "The number of parsed sentences = %d." % (len(Sen))

#Create tokens of sentences to make words
sen_token = [nltk.word_tokenize(i) for i in Sen]
freq_of_words = nltk.FreqDist(itertools.chain(*sen_token))
print "The number of new words tokens found are %d." % len(freq_of_words.items())

#Find out common words
Vocabulary = freq_of_words.most_common(vocabulary_size-1)

#Make Index - to - Word Vector
idx_to_wd = [v[0] for v in Vocabulary]
idx_to_wd.append(not_known)

#Make Word - to - Index Vector
wd_to_idx = dict([(word, idx) for idx, word in enumerate(idx_to_wd)])

print "The current length of vocab = %d." % vocabulary_size
print "The least frequent word = %s." % Vocabulary[-1][0]
print "The number of times of appearance of the least frequent word = %d." % Vocabulary[-1][1]

#The words not present in our vocab are changed to 'not_known' token
for i, s in enumerate(sen_token):
    sen_token[i] = [Word if Word in wd_to_idx else not_known for Word in s]
    
train_data = np.asarray([[wd_to_idx[Word] for Word in s[:-1]] for s in sen_token])
train_label = np.asarray([[wd_to_idx[Word] for Word in s[1:]] for s in sen_token])


model = RNNLanguageModel(vocabulary_size, hidden_size=_Hidden)
samay1 = time.time()
model.grad_descent(train_data[10], train_label[10], _Alpha)
samay2 = time.time()
print "Time of Gradient Descent step = %f ms" % ((samay2 - samay1) * 1000.)

if _Model != None:
    utils.load_param(_Model, model)

train(model, train_data, train_label, epochs = _Epochs, alpha = _Alpha)