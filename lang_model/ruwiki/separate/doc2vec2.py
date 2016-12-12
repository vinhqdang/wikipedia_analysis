# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
import subprocess

# numpy
import numpy

# shuffle
from random import shuffle

# logging
import logging
import os.path
import sys
import cPickle as pickle
import getopt

model_size = 200    # length of output vectors
nb_epochs      = 50    # number of training epochs
MAX_KEY = 500000

try:
      opts, args = getopt.getopt(sys.argv[1:],"hepoch:",["model_size=","epoch="])
except getopt.GetoptError as e:
      print ("Error of parameters")
      print (e)
      print (sys.argv[0] + " -h for help")
      sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print ('Convert text documents to vectors by Doc2Vec')
        print (sys.argv[0] + " -h for help")
        sys.exit ()
    elif opt in ("-model_size","--model_size"):
        model_size = int (arg)
    elif opt in ("-epoch","--epoch"):
        nb_epochs = int (arg)

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))

class LabeledLineSentence(object):

    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(LabeledSentence(
                        utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences

#sources = {'test-neg.txt':'TEST_NEG', 'test-pos.txt':'TEST_POS', 'train-neg.txt':'TRAIN_NEG', 'train-pos.txt':'TRAIN_POS', 'train-unsup.txt':'TRAIN_UNS'}

sources = {'FA.txt':'FA',
            'GA.txt':'GA',
            'SA.txt':'SA',
            'I.txt':'I',
            'II.txt':'II',
            'III.txt':'III',
            'IV.txt':'IV'}

sentences = LabeledLineSentence(sources)

model = Doc2Vec(min_count=0, window=10, size=model_size, sample=1e-4, negative=5, workers=8, dm = 1)

model.build_vocab(sentences.to_array())

for epoch in range(nb_epochs):
    logger.info('Epoch %d' % epoch)
    model.train(sentences.sentences_perm())

# model.save('./enwiki_quality.d2v')

# count number of line of a text file
# little bit awkward but okay
def count_lines (file_name):
    return sum(1 for line in open(file_name))

def convert_array_to_string (data):
    res = ""
    for i in range(len(data)):
        res = res + str (data[i])
        if (i < len(data) - 1):
            res = res + '\t'
    return res

def write_array_to_file (file_name, array_data):
    f = open (file_name, "w")
    for i in range (len(array_data)):
        f.write (str(array_data[i]) + "\n")
    f.close ()

qualities = ['FA','GA','B','I','II','III','IV']
train_labels = [0] * 23577
test_labels = [0] * 5891
train_content_file = "doc2vec_content.txt"
# test_content_file = "doc2vec_test_content.txt"
train_label_file = "doc2vec_label.txt"
# test_label_file = "doc2vec_test_label.txt"
train_cnt = 0
# test_cnt = 0
for i in range (len(qualities)):
    for j in range (MAX_KEY):
                key = qualities[i] + "_" + str(j)

                # check if key is greater than number of lines (i.e number of wikipedia entries in the file)
                file_name = qualities[i].lower() + ".txt"

                if key >= count_lines(file_name):
                    break

                data = model.docvecs[key]
                if (len(data) == model_size):
                    with open(train_content_file, "a") as myfile:
                        myfile.write(convert_array_to_string (data))
                        myfile.write("\n")
                    train_labels [train_cnt] = qualities[i]
                    train_cnt += 1
                # key = 'TEST_' + qualities[i] + "_" + str(j)
                # data = model.docvecs[key]
                # if (len(data) == DOCUMENT_LENGTH):
                #     with open(test_content_file, "a") as myfile:
                #         myfile.write(convert_array_to_string (data))
                #         myfile.write("\n")
                #     test_labels [test_cnt] = qualities[i]
                #     test_cnt += 1

write_array_to_file (file_name = train_label_file, array_data = train_labels)
# write_array_to_file (file_name = test_label_file, array_data = test_labels)

