from utils import NewsContent
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
from gensim.models.word2vec import LineSentence, PathLineSentences
from gensim.utils import save_as_line_sentence
import numpy as np
# from nltk import bigrams, trigrams

import multiprocessing
from utils import tsne_similar_word_plot, get_ngram

import re



# sentences iterable
title = NewsContent('../FakeNewsNet/code/fakenewsnet_dataset', 'politifact', 'fake', 'title')
content = NewsContent('../FakeNewsNet/code/fakenewsnet_dataset', 'politifact', 'fake', 'text')
# model = KeyedVectors.load_word2vec_format(fname='./GoogleNews-vectors-negative300.bin', binary=True)
# words = model.index2word
save_as_line_sentence(title, "title_ls")
save_as_line_sentence(content, "content_ls")
sentence = LineSentence('content_ls')
# b = bigrams(sentence)
bigram = get_ngram(2, sentence)
trigram = get_ngram(3, sentence)
# print(get_ngram(1, sentence))
# print()
# title_model = Word2Vec(title, sg=1, size=100, workers=multiprocessing.cpu_count() *10, min_count=1)
# content_model = Word2Vec(sentence, sg=1, size=100, workers=multiprocessing.cpu_count() *10, min_count=1)

# tsne_similar_word_plot(model, "trump")