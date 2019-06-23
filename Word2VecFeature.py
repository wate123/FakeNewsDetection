from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence, PathLineSentences
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# model = KeyedVectors.load_word2vec_format(fname="./GoogleNews-vectors-negative300.bin", binary=True)
# print("model loaded")

class Word2VecFeatureGenerator(object):
    """
    Class that builds a Word2Vec model (either pre-trained or not) pertaining to the news content
    """
    
    def __init__(self, corpus, pretrain=False):
        """
        Initializer function that takes a corpus and specification of whether to use a pretrained model or not
        """
        # uses Google News Word2Vec model
        if pretrain:
            self.model = Word2Vec("./GoogleNews-vectors-negative300.bin", sg=1, size=100, workers=200, min_count=1)
            print("Google News pretrained model loaded")
        # to train own model
        else:
            self.model = Word2Vec(corpus, sg=1, size=100, workers=200, min_count=1)
            print("model loaded")
    
    def get_norm_vectors(self, words):
        """
        Function to retrieve normalized vectors
        :param words: all words in title or body
        :return:
        """
        vectors = []

        # Pre-compute L2-normalized vectors.
        self.model.wv.init_sims()
        # iterating through words, normalizing their vectors
        for word in words:
            # ignore all empty
            try:
                # saving only the normalized vectors to list
                word_vec = self.model.wv.word_vec(word)
                vectors.append(word_vec)
            except KeyError or TypeError:
                return
        return np.asarray(vectors)

    def get_title_body_cos_sim(self, features):
        """
        Function to get cosine similarity between a title of article and its body content
        :param features: title and body pairs
        :return: cosine similarity of title and body
        """
        # title_vec = []
        # body_vec = []
        sim_vec = []
        # body_sim_vec = []

        # test = [{i: j} for i, j in features]
        # for count, (title, body) in enumerate(features):
        #     if count == 32:
        #         print(title)
        #         print(body)
        # iterating through news content's title and body
        for title, body in features:
            # print(title)
            # print(body)
            # obtain normalized vector of title and body
            title_vec = self.get_norm_vectors(title)
            body_vec = self.get_norm_vectors(body)
            if title_vec is None or body_vec is None:
                pass
            else:
                # sim_vec.append(title_vec)
                # body_sim_vec.append(body_vec)
                # compare normalized vectors using cosine similarity
                sim_vec.append(cosine_similarity(title_vec, body_vec))
        # print(sim_vec)

        return np.asarray(sim_vec)

    def process_and_save(self, pair_data):
        """
        TODO Incomplete
        :param pair_data: title and body pairs
        :return:
        """
        w2v_feature_df = pd.DataFrame(self.get_title_body_cos_sim(pair_data))
        w2v_feature_df["label"] = pd.read_json("data.json")["label"]

        X = w2v_feature_df.drop("label", axis=1)
        # X = self.get_title_body_cos_sim(pair_data)
        y = w2v_feature_df["label"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        clf = LogisticRegression()

        model = clf.fit(X_train, y_train)

        result = clf.predict(X_test)
        score = metrics.accuracy_score(y_test, result)
        precision = metrics.precision_score(y_test, result, ["fake", "real"], pos_label="real")
        recall = metrics.recall_score(y_test, result, ["fake", "real"], pos_label="real")
        f1 = metrics.f1_score(y_test, result, ["fake", "real"], pos_label="real")
        print("accuracy: ", score)
        print("precision: ", precision)
        print("recall: ", recall)
        print("f1: ", f1)

        w2v_feature_df.to_csv("w2v_feature.csv")
        self.model.save("w2v_model")

    def read(self):
        df = pd.read_csv('w2v_feature.csv')
        X = df.drop("label", axis=1)
        y = df["label"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        clf = LogisticRegression(solver="saga")

        model = clf.fit(X_train, y_train)

        result = clf.predict(X_test)
        score = metrics.accuracy_score(y_test, result)
        precision = metrics.precision_score(y_test, result, ["fake", "real"], pos_label="real")
        recall = metrics.recall_score(y_test, result, ["fake", "real"], pos_label="real")
        f1 = metrics.f1_score(y_test, result, ["fake", "real"], pos_label="real")
        print("accuracy: ", score)
        print("precision: ", precision)
        print("recall: ", recall)
        print("f1: ", f1)

