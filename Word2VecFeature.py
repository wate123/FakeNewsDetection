from gensim.models import Word2Vec

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from functools import reduce
from gensim.models.word2vec import LineSentence
from utils import NewsContent
from os.path import isfile
from joblib import Parallel, delayed

class Word2VecFeatureGenerator(object):
    """
    Class that builds a Word2Vec model (either pre-trained or not) from the news content
    """
    
    def __init__(self, pretrain=False):
        """
        Initializer function that takes a corpus and specification of whether to use a pretrained model or not
        """
        # if isfile('w2c_model'):
        #     self.model = Word2Vec.load("w2c_model")
        #     print("model loaded")
        # uses Google News Word2Vec model
        if pretrain:
            self.model = Word2Vec("./GoogleNews-vectors-negative300.bin", sg=1, size=300, workers=40, min_count=1)
            print("Google News pretrained model loaded")
        # to train own model
        else:
            self.model = Word2Vec(LineSentence('news_corpus.txt'), sg=1, size=300, workers=40, min_count=1)
            # self.model.wv.init_sims()
            print("model trained")
            self.model.save("w2c_model")
            print("model saved into w2c_model")

    def cosine_sim(self, x, y):
        """
        Function to compute cosine similarity
        :param x:
        :param y:
        :return:
        """
        try:
            if type(x) is np.ndarray: x = x.reshape(1, -1)  # get rid of the warning
            if type(y) is np.ndarray: y = y.reshape(1, -1)
            d = cosine_similarity(x, y)
            d = d[0][0]
        except:
            d = 0.
        return d

    def get_title_body_cos_sim(self, features):
        """
        Function to get cosine similarity between a title of article and its body content
        :param features: title and body pairs
        :return: cosine similarity of title and body
        """

        # unpack title and body tokens tuple into 2 separate list
        title_uni_list, body_uni_list = zip(*features)
        max_size_words = len(max(body_uni_list, key=len))
        len_news = len(body_uni_list)
        title_vec = np.zeros((max_size_words, len_news))

        # self.title_vec = np.array(list(map(lambda x:
        #                               reduce(np.add,
        #                                      [self.model.wv[word] for word in x if word in self.model],
        #                                      [0.] * 300),
        #                               title_uni_list)))
        # title = np.array(list(map(lambda x:
        #                                    [self.model.wv[word] for word in x if word in self.model],
        #                                    title_uni_list)))
        # body = np.array(list(map(lambda x:
        #                                    [self.model.wv[word] for word in x if word in self.model],
        #                                    body_uni_list)))
        # title_vec[:body.shape[0], :body.shape[1]] = body
        self.body_vec = np.array(list(map(lambda x:
                                         np.pad(np.array(
                                             [self.model.wv[word] for word in x if word in self.model]),
                                             (0, max_size_words - len(x)), mode="constant"),
                                         body_uni_list)))
        # self.title_vec = np.array(list(map(lambda x:
        #                               np.pad(np.array(
        #                                   [self.model.wv[word] for word in x if word in self.model]),
        #                                   (0, max_size_words - len(x)), mode="constant"),
        #                               title_uni_list)))

        self.body_vec = np.array(list(map(lambda x:
                                     reduce(np.add,
                                            [self.model.wv[word] for word in x if word in self.model],
                                            [0.] * 300),
                                     body_uni_list)))
        body_vec = np.array(list(map(lambda x:
                                     [self.model.wv[word] for word in x if word in self.model],
                                     body_uni_list)))
        # norm_title_vec = normalize(title_vec)
        # norm_body_vec = normalize(body_vec)

        self.sim_vec = np.array(list(map(lambda x, y: self.cosine_sim(x, y),
                                      self.title_vec,
                                      self.body_vec)))[:, np.newaxis]

        return np.hstack([self.title_vec, self.body_vec, self.sim_vec])

    # def get_nn_vec(self):
        # map(lambda x: [self.model.wv[word] for word in x if word in self.model],)

    def process_and_save(self, pair_data):
        """
        Function that uses Word2Vec feature to embed data and store in data frame then write to csv
        :param pair_data: title and body pairs
        """
        print("Generating Word2Vec Features")
        w2v_feature_df = pd.DataFrame(self.get_title_body_cos_sim(pair_data))
        w2v_feature_df["label"] = pd.read_csv("data.csv")["label"]
        w2v_feature_df.to_csv("w2v_feature.csv", index=False)
        print("Done! save into w2v_features.csv")

    def get_weights(self):
        return self.model.wv.vectors

    def read(self):
        """
        Function that reads directly from file
        :return: word2vec results without index to ensure model doesn't use index when training or testing
        """
        return pd.read_csv('w2v_feature.csv', index_col=False).drop("label", axis=1)


if __name__ == '__main__':
    data = NewsContent('../fakenewsnet_dataset', ['politifact'], ['fake', 'real'])
    Word2VecFeatureGenerator().get_title_body_cos_sim(data.get_features("pair"))
