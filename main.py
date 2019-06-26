from utils import NewsContent
import json
from gensim.corpora.dictionary import Dictionary
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
from gensim.models.word2vec import LineSentence, PathLineSentences
from gensim.utils import save_as_line_sentence
import numpy as np
import multiprocessing
from utils import tsne_similar_word_plot, get_ngram
from Word2VecFeature import Word2VecFeatureGenerator
from CountFeature import CountFeatureGenerator
from SentimentFeature import SentimentFeatureGenerator
from TfidfFeature import TfidfFeature
from SvdFeature import SvdFeature
import json
from sklearn.feature_extraction.text import TfidfVectorizer


# call NewsContent class to preprocess/tokenize the news content
# data = NewsContent('../FakeNewsNet/code/fakenewsnet_dataset', ['politifact'], ['fake', 'real'])
# save_as_line_sentence(data.get_features(), "news_corpus.txt")
# data.save_in_sentence_form()

# w2v = Word2VecFeatureGenerator(LineSentence("news_corpus.txt"))
# w2v.process_and_save(data.get_features("pair"))

# w2v.read()
# sim_vec = w2v.get_title_body_cos_sim()

# print(sim_vec)
# tfidf = TfidfFeature()
# tfidf.process()
# tfidf.get_scores()

# tsne_similar_word_plot(model, "trump")

# svd = SvdFeature()
# svd.process()
# svd_results = svd.svd_results()
# Count feature
cfg = CountFeatureGenerator()
# cfg.process_and_save()
cfg.read(trained=False)


# std = SentimentFeatureGenerator()
# std.process_and_save()
# std.read()
