from gensim.models import Word2Vec

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from functools import reduce
from gensim.models.word2vec import LineSentence
from gensim.models import KeyedVectors
from utils import NewsContent, preprocess
from os.path import isfile
from joblib import Parallel, delayed
import h5py
import math

class Word2VecFeatureGenerator(object):
    """
    Class that builds a Word2Vec model (either pre-trained or not) from the news content
    """
    
    def __init__(self, features, dataset, pretrain=True):
        """
        Initializer function that takes a corpus and specification of whether to use a pretrained model or not
        """
        # if isfile('w2c_model'):
        #     self.model = Word2Vec.load("w2c_model")
        #     print("model loaded")
        # uses Google News Word2Vec model
        self.dataset = dataset
        self.pretrain = pretrain
        if pretrain:
            self.model = KeyedVectors.load_word2vec_format("./GoogleNews-vectors-negative300.bin", binary=True)
            # self.model = Word2Vec("./GoogleNews-vectors-negative300.bin", sg=1, size=300, workers=40, min_count=1)
            print("Google News pretrained model loaded")
        # to train own model
        else:
            self.model = Word2Vec(LineSentence('news_corpus.txt'), sg=1, size=300, workers=40, min_count=1)
            self.model.delete_temporary_training_data()
            # self.model.wv.init_sims()
            print("model trained")
            self.model.save("w2c_model")
            print("model saved into w2c_model")
        self.title_vec = []
        self.body_vec = []
        self.title_uni_list, self.body_uni_list = zip(*features)
        self.sim_vec = []

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

    def get_title_body_cos_sim(self):
        """
        Function to get cosine similarity between a title of article and its body content
        :param features: title and body pairs
        :return: cosine similarity of title and body
        """
        # open('w2v_feature_pad.txt', 'w').close()
        # unpack title and body tokens tuple into 2 separate list
        self.title_vec = np.array(list(map(lambda x:
                                      reduce(np.add,
                                             [self.model.wv[word] for word in x if word in self.model],
                                             [0.] * 300),
                                      self.title_uni_list)))
        self.body_vec = np.array(list(map(lambda x:
                                     reduce(np.add,
                                            [self.model.wv[word] for word in x if word in self.model],
                                            [0.] * 300),
                                     self.body_uni_list)))
        # norm_title_vec = normalize(title_vec)
        # norm_body_vec = normalize(body_vec)

        self.sim_vec = np.array(list(map(lambda x, y: self.cosine_sim(x, y),
                                      self.title_vec,
                                      self.body_vec)))[:, np.newaxis]

        return np.hstack([self.title_vec, self.body_vec, self.sim_vec])

    def get_nn_vecs(self):
        print("Start prepare word2vec vectors")
        max_title_length = len(max(self.title_uni_list, key=len))
        max_body_length = 1000
        w2vs = []

        for atitle, abody in zip(self.title_uni_list, self.body_uni_list):
            # temp_tokens = np.zeros((max_len, 300))
            temp_title_token = np.zeros((max_title_length, 300))
            temp_body_token = np.zeros((max_body_length, 300))
            for index, (title_word, body_word) in enumerate(zip(atitle, abody)):
                if title_word in self.model:
                    temp_title_token[index] = self.model.wv[title_word]
                else:
                    temp_title_token[index] = np.zeros(300, )
                if body_word in self.model:
                    temp_body_token[index] = self.model.wv[body_word]
                    # body_vec.append(np.pad(np.array(self.model.wv[body_word]), (0, max_size_words - len(abody)), mode="constant"))
                else:
                    temp_body_token[index] = np.zeros(300, )
            w2vs.append(np.concatenate((temp_title_token, temp_body_token), axis=0))
        print("Done")
        # np.save('w2v_feature_pad.npy', np.array(w2vs))
        if self.pretrain:
            with h5py.File("./Features/"+"-".join(self.dataset)+"/w2v_feature_" + str(max_body_length) + "-".join(self.dataset) + "_pretrain_pad.hdf5", "w") as f:
                f.create_dataset("w2v", data=w2vs)
            print("Save into w2v_feature_1000"+ "-".join(self.dataset) +"_pretrain_pad.hdf5")
        else:
            with h5py.File("./Features/"+"-".join(self.dataset)+"/w2v_feature_" + str(max_body_length) + "-".join(self.dataset) + "_pad.hdf5", "w") as f:
                f.create_dataset("w2v", data=w2vs)
            print("Save into w2v_feature_1000"+ "-".join(self.dataset) +"_pad.hdf5")
        # return np.array(w2vs)

    def process_and_save(self):
        """
        Function that uses Word2Vec feature to embed data and store in data frame then write to csv
        :param pair_data: title and body pairs
        """
        print("Generating Word2Vec Features")
        w2v_feature_df = pd.DataFrame(self.get_title_body_cos_sim())
        w2v_feature_df["label"] = pd.read_csv("data.csv")["label"]
        w2v_feature_df.to_csv("./Features/"+"-".join(self.dataset)+"/w2v_feature.csv", index=False)
        print("Done! save into w2v_features.csv")

    def get_weights(self):
        return self.model.wv.vectors

    def read(self):
        """
        Function that reads directly from file
        :return: word2vec results without index to ensure model doesn't use index when training or testing
        """
        return pd.read_csv("./Features/"+"-".join(self.dataset)+"/w2v_feature.csv", index_col=False).drop("label", axis=1)


if __name__ == '__main__':
    dataset = ['politifact']
    data = NewsContent('../fakenewsnet_dataset', dataset, ['fake', 'real'])
    Word2VecFeatureGenerator(data.get_features("pair"), dataset).get_nn_vecs()
