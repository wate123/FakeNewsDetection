# from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import text_to_word_sequence
import os
import json
from gensim.models import KeyedVectors
from gensim.models import Word2Vec

with open('fake.json', 'r') as f:
    data = json.load(f)
news = data["text"]
# print(news)
vocab = set(text_to_word_sequence(news))
vocab_size = len(vocab)


# Memory saving loading words
class NewsContent(object):
    def __init__(self, dirname, site, news_type):
        self.dirname = dirname
        self.site = site
        self.news_type = news_type

    def __iter__(self):
        for file_path in self.get_gossipcop(self.site, self.news_type):
            for line in open(file_path):
                yield line.split()

    # for line in open(os.path.join(self.dirname, fname)):
    #     yield line.split()
    def get_news_content(self, site, news_type):
        site_folder = os.path.join(self.dirname, site)
        news_path = os.path.join(site_folder, news_type)
        return self.get_list_news_files(news_path, 'json')

    '''
    Return all the files you want in the provided directory
    @:param directory root direction you want to search
    '''
    @staticmethod
    def get_list_news_files(directory, file_ext):
        file_ext = file_ext.lower()
        list_news_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith("."+file_ext):
                    list_news_files.append(os.path.join(root, file))
        return list_news_files


# print(os.listdir(('./FakeNewsNet/code/fakenewsnet_dataset')))
# model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
# sentences = MySentences('/some/directory')
# model = Word2Vec(vocab, size=100, window=5, min_count=1, workers=4)
# word_vectors = model.wv
# print(model)
sentences = NewsContent(
    './FakeNewsNet/code/fakenewsnet_dataset', 'gossipcop', 'fake')
model = Word2Vec(sentences)
print((model))
