import sys
import gensim
import load
# import argparse

# parser = argparse.ArgumentParser(description='Convert documents to vectors')

# parser.add_argument('--data', dest='data_dir', action='store_const',
#                     const='text',
#                     help='provide the data directory which contains all documents')
# parser.add_argument('--size', dest='model_size', action='store_const',
#                     const=500,
#                     help='size of output vectors')
# parser.add_argument('--alpha', dest='learning_rate', action='store_const',
#                     const=0.005,
#                     help='learning rate')
# args = parser.parse_args()
 
documents = load.get_doc(sys.argv[1])

model_size = 500

if (len(sys.argv) >= 3):
    model_size = int (sys.argv[2])

nb_epochs = 200
if (len(sys.argv) >= 4):
    nb_epochs = int (sys.argv[3])

print ('Data Loading finished')
 
print (len(documents),type(documents))
 
# build the model
model = gensim.models.Doc2Vec(documents, dm = 1, alpha=0.025, size= model_size, min_alpha=0.025, min_count=0)
 
# start training
for epoch in range(nb_epochs):
    if epoch % 20 == 0:
        print ('Now training epoch %s'%epoch)
    model.train(documents)
    model.alpha -= 0.002  # decrease the learning rate
    model.min_alpha = model.alpha  # fix the learning rate, no decay
 
# shows the similar words
print (model.most_similar('suppli'))
 
# shows the learnt embedding
print (model['suppli'])
 
# shows the similar docs with id = 2
print (model.docvecs.most_similar(str(2)))