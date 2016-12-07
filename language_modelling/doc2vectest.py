import sys
import gensim
import load
 
documents = load.get_doc(sys.argv[1])
print ('Data Loading finished')
 
print (len(documents),type(documents))
 
# build the model
model = gensim.models.Doc2Vec(documents, dm = 0, alpha=0.025, size= 20, min_alpha=0.025, min_count=0)
 
# start training
for epoch in range(200):
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