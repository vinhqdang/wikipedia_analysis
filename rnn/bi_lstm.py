from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import to_categorical, pad_sequences

import csv
import numpy as np
from sklearn import metrics, cross_validation
from sklearn.model_selection import KFold
# import pandas

import tensorflow as tf
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell, GRUCell
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.embedding_ops import embedding
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
model_size = 20000    # length of output vectors
nb_epochs      = 200    # number of training epochs
embedding_size = 20000    # output size of the embedding layer
label_file = "enwikilabel"  # a text file contains correct quality classes
MAX_FILE_ID = 500000         # adhoc manner, should be set to a very large value
cell_size = [128]       # number of neurons per each layer
dropout_ratio = 0.5     # dropout ratio per each layer
dynamic = True          # use dynamic LSTM or not
activation_function = "relu"    # activation function, can be "sigmoid", "tanh" etc.
learning_rate = 0.001           # lower -> slower training
test_ratio = 0.2
cell_type = "lstm"
nfolds=5        # k-fold cross validation
batch_size = 32 # mini-batch training

try:
      opts, args = getopt.getopt(sys.argv[1:],"hd:model_size:epoch:lb:es:",["model_size=","epoch=","es=","cell_size=","dropout=","dynamic=","activation=","rate=","test_ratio=","cell_type="])
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
    elif opt in ["--cell_size"]:
        cell_size = arg.split (",")
        cell_size = [int (x) for x in cell_size]
    elif opt in ["--dropout"]:
        dropout_ratio = float (arg)
    elif opt in ["--dynamic"]:
        dynamic = bool (arg)
    elif opt in ["--activation"]:
        activation_function = arg
    elif opt in ["--rate"]:
        learning_rate = float (arg)
    elif opt in ["--test_ratio"]:
        test_ratio = float (arg)
    elif opt in ["--cell_type"]:
        if (arg != "lstm" and arg != "gru"):
            print ("Wrong cell type. Accept only lstm or gru.")
            sys.exit(1)
        cell_type = arg

### Training data

# only for English Wikipedia. need to be changed for each languages
qualities = ["stub","start","c","b","ga","fa"]

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

print ('Finish reading data')

print ('Create folds')
# create 5-fold
fold1, fold2, fold3, fold4, fold5 = np.split(X.sample(frac=1), [int(.2*len(df)),int(.4*len(df)),int(.6*len(df)), int(.8*len(df))])
kfolds=[fold1,fold2,fold3,fold4,fold5]
Y1, Y2, Y3, Y4, Y5 = np.split(Y.sample(frac=1), [int(.2*len(df)),int(.4*len(df)),int(.6*len(df)), int(.8*len(df))])
yfolds=[Y1,Y2,Y3,Y4,Y5]

for i in range (nfolds):

    # X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y,
    #     test_size=test_ratio, random_state=2017)
    X_train = kfolds[i]
    Y_train = yfolds[i]
    X_test = []
    Y_test = []
    for j in range (nfolds):
        if i != j:
            X_test += kfolds[j]
            Y_test += yfolds[j]

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

    ### Models

    print('Build model')

    net = input_data([None, model_size])
    net = embedding(net, input_dim=n_words, output_dim=embedding_size)
    if cell_type == "lstm":
        for i in range (len (cell_size)):
          if i < len(cell_size) - 1:
            net = bidirectional_rnn(net, BasicLSTMCell(cell_size[i]), BasicLSTMCell(cell_size[i]), return_seq = True)
            net = dropout(net, dropout_ratio)
          else:
            net = bidirectional_rnn(net, BasicLSTMCell(cell_size[i]), BasicLSTMCell(cell_size[i]))
            net = dropout(net, dropout_ratio)
    elif cell_type == "gru":
        for i in range (len (cell_size)):
          if i < len(cell_size) - 1:
            net = bidirectional_rnn(net, GRUCell(cell_size[i]), GRUCell(cell_size[i]), return_seq = True)
            net = dropout(net, dropout_ratio)
          else:
            net = bidirectional_rnn(net, GRUCell(cell_size[i]), GRUCell(cell_size[i]))
            net = dropout(net, dropout_ratio)
    net = fully_connected(net, len (qualities), activation='softmax')
    net = regression(net, optimizer='adam', learning_rate=learning_rate,
                             loss='categorical_crossentropy')

    print ('Train model')

    model = tflearn.DNN(net, tensorboard_verbose=1, tensorboard_dir = "logdir/bi_lstm")

    print ('Predict')
    model.fit(X_train, Y_train, validation_set=(X_test, Y_test), show_metric=True,
              batch_size=batch_size, n_epoch = nb_epochs)