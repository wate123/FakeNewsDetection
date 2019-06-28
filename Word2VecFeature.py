from gensim.models import Word2Vec

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from functools import reduce
from os.path import isfile


class Word2VecFeatureGenerator(object):
    """
    Class that builds a Word2Vec model (either pre-trained or not) from the news content
    """
    
    def __init__(self, corpus, pretrain=False):
        """
        Initializer function that takes a corpus and specification of whether to use a pretrained model or not
        """
        if isfile('w2c_model'):
            self.model = Word2Vec.load("w2c_model")
            print("model loaded")
        # uses Google News Word2Vec model
        elif pretrain:
            self.model = Word2Vec("./GoogleNews-vectors-negative300.bin", sg=1, size=100, workers=200, min_count=1)
            print("Google News pretrained model loaded")
        # to train own model
        else:
            self.model = Word2Vec(corpus, sg=1, size=300, workers=200, min_count=1)
            # self.model.wv.init_sims()
            print("model trained")
            self.model.save("w2c_model")
            print("model saved into w2c_model")
    
    def get_norm_vectors(self, words):
        """
        Function to retrieve normalized vectors
        :param words: all words in title or body
        :return:
        """
        vectors = []
        # iterating through words, normalizing their vectors
        for word in words:
            # ignore all empty
            try:
                # saving only the normalized vectors to list
                word_vec = self.model.wv.word_vec(word)
                vectors.append(word_vec)
            except KeyError or TypeError:
                vectors.append(np.zeros((100,)))
        if not vectors:
            return np.zeros((1,100))
        return np.array(vectors)

    def cosine_sim(self, x, y):
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

        title_vec = np.array(list(map(lambda x:
                                      reduce(np.add,
                                             [self.model.wv[word] for word in x if word in self.model],
                                             [0.] * 300),
                                      title_uni_list)))
        body_vec = np.array(list(map(lambda x:
                                     reduce(np.add,
                                            [self.model.wv[word] for word in x if word in self.model],
                                            [0.] * 300),
                                     body_uni_list)))
        # norm_title_vec = normalize(title_vec)
        # norm_body_vec = normalize(body_vec)

        sim_vec = np.asarray(list(map(lambda x, y: self.cosine_sim(x, y),
                                      title_vec,
                                      body_vec)))[:, np.newaxis]

        return np.hstack([title_vec, body_vec, sim_vec])

    def process_and_save(self, pair_data):
        """
        TODO Incomplete
        :param pair_data: title and body pairs
        :return:
        """
        print("Generating Word2Vec Features")
        w2v_feature_df = pd.DataFrame(self.get_title_body_cos_sim(pair_data))
        w2v_feature_df["label"] = pd.read_csv("data.csv")["label"]
        w2v_feature_df.to_csv("w2v_feature.csv", index=False)
        print("Done! save into w2v_features.csv")

    def read(self):
        return pd.read_csv('w2v_feature.csv', index_col=False).drop("label", axis=1)
