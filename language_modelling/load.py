import gensim
import os
import re
# from nltk.tokenize import RegexpTokenizer
# from stop_words import get_stop_words
# from nltk.stem.porter import PorterStemmer
from gensim.models.doc2vec import TaggedDocument
 
def get_doc_list(folder_name):
    doc_list = []
    file_list = [folder_name+'/'+name for name in os.listdir(folder_name)]
    for file in file_list:
        st = open(file,'r').read()
        doc_list.append(st)
    print ('Found %s documents under the dir %s .....'%(len(file_list),folder_name))
    return doc_list
 
# folder_name: contains text documents
# label_file: a file contains label for each document in folder_name
def get_doc(folder_name, label_file):
 
    doc_list = get_doc_list(folder_name)
    # tokenizer = RegexpTokenizer(r'\w+')
    # en_stop = get_stop_words('en')
    # p_stemmer = PorterStemmer()
     
    labels = []
    with open (label_file, "r") as f:
        labels = f.read().splitlines()

    if (len(labels) != len(doc_list)):
        print ('Error! Number of labels is not as same as number of documents.')
        print ('There are ' + str (len(labels)) + ' labels')
        print ('There are ' + str (len(doc_list)) + ' documents')
        sys.exit (1)
 
    unique_labels = list (set (labels))
    index_count = [0] * len (unique_labels)
    taggeddoc = []
 
    texts = []
    for index,i in enumerate(doc_list):
        if index % 1000 == 0:
            print ("Processing file " + str (index) + "/" + str(len(doc_list)) + " = " + "{0:.2f}".format(float(index)*100/len(doc_list)) + "%")
        # print (i)
        # print ("----")
        # for tagged doc
        # wordslist = []
        # tagslist = []
 
        # # clean and tokenize document string
        # raw = i.lower()
        # tokens = tokenizer.tokenize(raw)
 
        # # remove stop words from tokens
        # stopped_tokens = [i for i in tokens if not i in en_stop]
 
        # # remove numbers
        # number_tokens = [re.sub(r'[\d]', ' ', i) for i in stopped_tokens]
        # number_tokens = ' '.join(number_tokens).split()
 
        # # stem tokens
        # stemmed_tokens = [p_stemmer.stem(i) for i in number_tokens]
        # # remove empty
        # length_tokens = [i for i in stemmed_tokens if len(i) &amp;amp;gt; 1]
        # # add tokens to list
        # texts.append(length_tokens)
        label = labels [index]
        td = TaggedDocument(gensim.utils.to_unicode(str.encode(i)).split(), label + str(index_count[unique_labels.index(label)]))
        index_count[unique_labels.index(label)] += 1
        taggeddoc.append(td)
 
    return taggeddoc