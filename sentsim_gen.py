import re, os
from string import ascii_lowercase
from string import punctuation
from string import digits
from nltk.corpus import stopwords

import csv
from gensim.models.doc2vec import TaggedDocument, Doc2Vec

#constants
IS_LINK_OBJ = re.compile(r'^(?:@|https?://)')
STOPWORDS = set(stopwords.words('english'))

class ModelGenerate(object):

    def __init__(self, csv_file=None, label1=None, label2=None):
        self.csv_file = csv_file
        self.label1 = label1
        self.label2 = label2        

    #helper functions    
    def _is_ascii(self, w):
        return all(ord(c) < 128 for c in w)

    def _remove_punc_digit(self, w):    
        #punc_dig = punctuation+digits
        punc_dig = punctuation
        return ''.join([c for c in w if c not in punc_dig])

    def _strip_non_ascii(self, w):
        return ''.join([i for i in w if i in ascii_lowercase])

    def _get_filename(self, u):
        return os.path.splitext(os.path.basename(u))[0]

    def tokenize_words(self, words):
        words = [ word for word in words if word not in STOPWORDS ]
        words = [ word for word in words if self._is_ascii(word) ]
        words = [ word for word in words if not IS_LINK_OBJ.search(word) ]
        words = [ self._remove_punc_digit(word) for word in words ]        
        words = set([ word for word in words if len(word) >= 3])
        return words

    def save_model(self, model_name):
        texts = csv.DictReader(open(self.csv_file))
        sentences = [ TaggedDocument(self.tokenize_words(row[self.label1].lower().split()), [row[self.label2]]) for row in texts ]
        #model = Doc2Vec(vector_size=500, window=10, min_count=5, workers=11, alpha=0.005, min_alpha=0.005)

        model = Doc2Vec(size=1000, window=10, min_count=5, workers=11, alpha=0.025, min_alpha=0.025, iter=250)
        model.build_vocab(sentences)
        print("training model")
        model.train(sentences, epochs=model.epochs, total_examples=model.corpus_count)
        model.save(os.path.join("models", model_name))
        print("saved to {}".format(model_name))


