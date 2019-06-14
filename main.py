from utils import NewsContent
import json
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
from gensim.models.word2vec import LineSentence, PathLineSentences
from gensim.utils import save_as_line_sentence
import numpy as np
import multiprocessing
from utils import tsne_similar_word_plot, get_ngram
from word2vecfeature import Word2VecFeatureGenerator



# sentences iterable
# title = NewsContent('../FakeNewsNet/code/fakenewsnet_dataset', 'politifact', 'fake', 'title')
# content = NewsContent('../FakeNewsNet/code/fakenewsnet_dataset', 'politifact', 'fake', 'text')
data = NewsContent('../FakeNewsNet/code/fakenewsnet_dataset', 'politifact', 'fake')
save_as_line_sentence(data.get_features('title'), "title_ls")
save_as_line_sentence(data.get_features('text'), "body_ls")

save_as_line_sentence(data.get_features(), "news_ls")
data.save_reference_table()

w2v = Word2VecFeatureGenerator(LineSentence("news_ls"))
sim_vec = w2v.get_title_body_cos_sim(data.get_features("pair"))

# tsne_similar_word_plot(model, "trump")
