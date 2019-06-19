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
from FeatureGenerator import FeatureGenerator
import json

# sentences iterable
# title = NewsContent('../FakeNewsNet/code/fakenewsnet_dataset', 'politifact', 'fake', 'title')
# content = NewsContent('../FakeNewsNet/code/fakenewsnet_dataset', 'politifact', 'fake', 'text')

# call NewsContent class to preprocess/tokenize the news content
data = NewsContent('../FakeNewsNet/code/fakenewsnet_dataset', 'politifact', 'fake')
title_words_list = list(data.get_features('title'))

# save_as_line_sentence(data.get_features('title'), "title_ls.txt")
# save_as_line_sentence(data.get_features('text'), "body_ls.txt")
#
save_as_line_sentence(data.get_features(), "news_ls.txt")
data.save_in_sentence_form()

w2v = Word2VecFeatureGenerator(LineSentence("news_ls.txt"))

sim_vec = w2v.get_title_body_cos_sim(data.get_features("pair"))

# print(len(list(data.get_features('text'))))
# print(len(list(LineSentence('body_ls.txt'))))

# tsne_similar_word_plot(model, "trump")
# print(len(list(data.get_features("pair"))))
# for i in data.get_features("pair"):
# Count feature
cfg = CountFeatureGenerator(data.get_features("pair"))
cfg.process_and_save()

std = SentimentFeatureGenerator()
std.process_and_save()
# title_uni_count = cfg.get_article_part_count(title_words_list, 1)
# title_bi_count = cfg.get_article_part_count(title_words_list, 2)
# title_tri_count = cfg.get_article_part_count(title_words_list, 3)
# print(title_bi_count)
