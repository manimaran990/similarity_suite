import gensim  
from sentsim_gen import *

if __name__ == '__main__':

    csv_file = "/home/mani/Documents/challenges/05/data/importpython.csv"
    model = ModelGenerate(csv_file, "text", "id")
    model.save_model('IMPORTPYTHON_MODEL')
    new_sentence = model.tokenize_words("@bostonpython mind giving a shoutout for our newsletter and free job-board http://t.co/GRxlRwVzVz on Twitter ... Thanks".lower().split())	
    print(new_sentence)
    model = gensim.models.Doc2Vec.load('models/IMPORTPYTHON_MODEL')    
    result = model.docvecs.most_similar(positive=[model.infer_vector(new_sentence)],topn=5)
    print("{} - {}".format("id", "score"))
    for row in result:
    	print("{} - {:.2f}%".format(row[0], float(row[1])*100))


