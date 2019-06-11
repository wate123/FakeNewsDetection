from utils import NewsContent
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import numpy as np
import multiprocessing
from utils import tsne_similar_word_plot

import re



# sentences iterable
words = NewsContent(
    '../FakeNewsNet/code/fakenewsnet_dataset', 'politifact', 'fake', 'title')
# model = KeyedVectors.load_word2vec_format(fname='./GoogleNews-vectors-negative300.bin', binary=True)
# words = model.index2word
model = Word2Vec(words, sg=1, size=100, workers=multiprocessing.cpu_count() *10, min_count=1)
word_vector = model.wv
print(len(word_vector.vocab))

tsne_similar_word_plot(model, "trump")