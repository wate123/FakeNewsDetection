from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence, PathLineSentences
from sklearn.preprocessing import normalize
import numpy as np, json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV as GCV
from sklearn.preprocessing import normalize
from functools import reduce
from os.path import isfile
# model = KeyedVectors.load_word2vec_format(fname="./GoogleNews-vectors-negative300.bin", binary=True)
# print("model loaded")

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

        # Pre-compute L2-normalized vectors.

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
        # with open("data.json", mode="r") as f:
        #     pair_data = json.load(f)
        w2v_feature_df = pd.DataFrame(self.get_title_body_cos_sim(pair_data))
        # w2v_feature_df = pd.DataFrame()
        # title_vec, body_vec, title_body_cos_sim = self.get_title_body_cos_sim(pair_data)
        # print("w2v shape: ", title_vec.shape)
        # w2v_feature_df["title_vec"] = title_vec
        # w2v_feature_df["body_vec"] = body_vec
        # w2v_feature_df["title_body_cos_sim"] = title_body_cos_sim
        w2v_feature_df["label"] = pd.read_csv("data.csv")["label"]
        w2v_feature_df.to_csv("w2v_feature.csv", index=False)
        # pd.DataFrame.to_csv(w2v_feature_df, "w2v_feature.csv")
        print("Done! save into w2v_features.csv")
        # w2v_feature_df.to_csv("w2v_feature.csv")

    def read(self):
        return pd.read_csv('w2v_feature.csv', index_col=False).drop("label", axis=1)
        # X = df
        # # X = self.get_title_body_cos_sim(pair_data)
        # y = df["label"].values
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        #
        # grid_C = [0.5 * i for i in range(1, 21)]
        # parameters = {"tol": [5e-4], "C": grid_C, "random_state": [1],
        #               "solver": ["newton-cg", "sag", "saga", "lbfgs"],
        #               "max_iter": [4000], "multi_class": ["multinomial", "ovr", "auto"]}
        #
        # clf = GCV(LogisticRegression(), parameters, cv=10, n_jobs=-1)
        # # clf = LogisticRegression(tol=0.0005, C=4.5, max_iter=4000, multi_class='ovr', random_state=1,
        # #                          solver='newton-cg', )
        # # if trained:
        # #     clf = load('logreg_count_feature.joblib')
        # # else:
        # #     clf = GCV(LogisticRegression(), parameters, cv=10, n_jobs=-1)
        # #     dump(clf, 'logreg_count_feature.joblib')
        # clf.fit(X_train, y_train)
        #
        # y_predict = clf.predict(X_test)
        # tpfptnfn = metrics.confusion_matrix(y_test, y_predict)
        # preRecF1 = metrics.classification_report(y_test, y_predict)
        # # print(clf.best_params_)
        # print(preRecF1)
        # plot_learning_curve(clf, "Learning curve", X_train, y_train, n_jobs=-1, cv=10)

