from __future__ import division, print_function, absolute_import
import csv
import numpy as np
from sklearn import metrics, cross_validation
# import pandas

import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical, pad_sequences

import getopt
import sys
import os

data_dir = "text"   # directory contains text documents
model_size = 2000    # length of output vectors
nb_epochs      = 10    # number of training epochs
embedding_size = 300
label_file = "enwikilabel"
MAX_FILE_ID = 50000
cnn_size = 128
dropout_ratio = 0.5
dynamic = True
activation_function = "relu"

try:
      opts, args = getopt.getopt(sys.argv[1:],"hd:model_size:epoch:lb:es:",["model_size=","epoch=","es=","cnn_size=","dropout=","dynamic=","activation="])
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
    elif opt in ["--cnn_size"]:
        cnn_size = int (arg)
    elif opt in ["--dropout"]:
        dropout_ratio = float (arg)
    elif opt in ["--dynamic"]:
        dynamic = bool (arg)
    elif opt in ["--activation"]:
        activation_function = arg

### Training data

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

# Building convolutional network
print ('Build CNN')
network = input_data(shape=[None, model_size], name='input')
network = tflearn.embedding(network, input_dim=n_words, output_dim=cnn_size)
branch1 = conv_1d(network, cnn_size, 3, padding='valid', activation=activation_function, regularizer="L2")
branch2 = conv_1d(network, cnn_size, 4, padding='valid', activation=activation_function, regularizer="L2")
branch3 = conv_1d(network, cnn_size, 5, padding='valid', activation=activation_function, regularizer="L2")
network = merge([branch1, branch2, branch3], mode='concat', axis=1)
network = tf.expand_dims(network, 2)
network = global_max_pool(network)
network = dropout(network, dropout_ratio)
network = fully_connected(network, len(qualities), activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.001,
                     loss='categorical_crossentropy', name='target')
# Training
print ('Training')
model = tflearn.DNN(network, tensorboard_verbose=0)

print ('Testing')
model.fit(trainX, trainY, n_epoch = nb_epochs, shuffle=True, validation_set=(testX, testY), show_metric=True, batch_size=32)