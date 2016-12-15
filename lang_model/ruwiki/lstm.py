from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import to_categorical, pad_sequences

import csv
import numpy as np
from sklearn import metrics, cross_validation
# import pandas

import tensorflow as tf
# from tf.nn.rnn_* import rnn, rnn_cell
# import skflow

import random
import time

# import cPickle as pickle

# from nltk.corpus import stopwords

import string

import getopt
import sys
import os

data_dir = "text"   # directory contains text documents
model_size = 2000    # length of output vectors
nb_epochs      = 50    # number of training epochs
embedding_size = 300
label_file = "ruwikilabel2.txt" # convert Russian character in quality labels to Latin characters
MAX_FILE_ID = 50000

try:
      opts, args = getopt.getopt(sys.argv[1:],"hd:model_size:epoch:lb:es:",["model_size=","epoch=","es="])
except getopt.GetoptError as e:
      print ("Error of parameters")
      print (e)
      print (sys.argv[0] + " -h for help")
      sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print ('LSTM for Wikipedia classification')
        print (sys.argv[0] + " -h for help")
        sys.exit ()
    elif opt in ("-model_size","--model_size"):
        model_size = int (arg)
    elif opt in ("-epoch","--epoch"):
        nb_epochs = int (arg)
    elif opt in ["-es","--es"]:
        embedding_size = int (arg)

### Training data

qualities = ["IV","III","II","I","SA","GA","FA"]

print('Read labels')

def load_label (label_file):
    with open (label_file) as f:
        return f.read().splitlines()

Y = load_label(label_file)

for i in range(len(Y)):
    Y[i] = qualities.index(Y[i])

print('Read content')

def load_content (file_name):
    with open(file_name) as f:
        return f.read()

X = []
for i in range (MAX_FILE_ID):
    file_name = data_dir + '/' + str(i + 1)
    if os.path.isfile (file_name):
        X.append (load_content(file_name)) 

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y,
    test_size=0.2, random_state=2017)

Y_train = to_categorical (Y_train, nb_classes = len (qualities))
Y_test = to_categorical (Y_test, nb_classes = len (qualities))

### Process vocabulary

print('Process vocabulary')

vocab_processor = tflearn.data_utils.VocabularyProcessor(max_document_length = model_size, min_frequency = 0)
X_train = np.array(list(vocab_processor.fit_transform(X_train)))
X_test = np.array(list(vocab_processor.fit_transform(X_test)))

X_train = pad_sequences(X_train, maxlen=model_size, value=0.)
X_test = pad_sequences(X_test, maxlen=model_size, value=0.)

n_words = len(vocab_processor.vocabulary_)
print('Total words: %d' % n_words)

# pickle.dump (X_train, open ("xtrain.p", b))
# pickle.dump (X_test, open ("xtest.p", b))

# X_train = pickle.load (open ("xtrain.p", rb))
# X_test = pickle.load (open ("xtest.p", rb))

### Models

print('Build model')

net = tflearn.input_data([None, model_size])
net = tflearn.embedding(net, input_dim=n_words, output_dim=128)
net = tflearn.lstm(net, 128, dropout=0.5)
net = tflearn.fully_connected(net, len (qualities), activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')

print ('Train model')

model = tflearn.DNN(net, tensorboard_verbose=3, tensorboard_dir = "logdir/lstm")

print ('Predict')
model.fit(X_train, Y_train, validation_set=(X_test, Y_test), show_metric=True,
          batch_size=32)