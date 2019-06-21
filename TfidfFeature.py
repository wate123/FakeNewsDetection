from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from scipy.sparse.csr import csr_matrix
import json
from utils import *


class TfidfFeature(object):

    def __init__(self):

        self.doc_list = []
        self.tf = TfidfVectorizer()
        self.tfidf_feature = {}

    def process(self):
        doc_list = []
        with open('data.json', mode="r") as f:
            data = json.load(f)
            for news in data:
               doc_list.append(news['title'] + ". " + news['body'])
               label.append(news['label'])

        X = doc_list
        y = label
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)

        tfidf_matrix = self.tf.fit_transform(X_train)

        testtf_matrix = self.tf.transform(X_test)
        
        # The cost is proportional to the square of the value of the weight coefficients
        # penalty of l2 as regularization has no feature selection, computationally efficient, nonsparse output
        # L1 regularization inefficient on non-sparse cases, built in feature selection, sparse output
        # instantiate the model (default parameters)
        # default max iter is 100 examples showing 10000
        # saga handles l2 and is fast for large dataset also could use sag
        # multiclass is ovr (one vs rest) as we only need binary classification

        logreg = LogisticRegression(penalty='l2', solver='saga', max_iter=1000)

        logreg.fit(tfidf_matrix, y_train)

        y_pred = logreg.predict(testtf_matrix)

        score = metrics.accuracy_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred, ["fake", "real"], pos_label="real")
        recall = metrics.recall_score(y_test, y_pred, ["fake", "real"], pos_label="real")
        f1 = metrics.f1_score(y_test, y_pred, ["fake", "real"], pos_label="real")
        print("accuracy: ", score)
        print("precision: ", precision)
        print("recall: ", recall)
        print("f1: ", f1)
        
        
        
        
        # tfidf_matrix = self.tf.fit_transform(doc_list)
        # will gives a list off all tokens or ngrams or words
        
        
        # assuming defaults of alpha = 1 (additive smoothing parameter,
        # fit_prior = True to learn class prior probabilities,
        # class_prior = None  the prior probabilites of the classes
        # clf = MultinomialNB()
        # clf.fit(tfidf_matrix, y_train)

        # X should be the tfidf matrix of news articles, y is the train target


        # y_pred = clf.predict(testtf_matrix)
        #
        # score = metrics.accuracy_score(y_test, y_pred)
        # precision = metrics.precision_score(y_test, y_pred, ["fake", "real"], pos_label="real")
        # recall = metrics.recall_score(y_test, y_pred, ["fake", "real"], pos_label="real")
        # f1 = metrics.f1_score(y_test, y_pred, ["fake", "real"], pos_label="real")
        # print("accuracy: ", score)
        # print("precision: ", precision)
        # print("recall: ", recall)
        # print("f1: ", f1)
        print(tfidf_matrix.shape)
        
        return tfidf_matrix



    def get_scores(self):


        score_matrix = self.process()

        tfidf_names = self.tf.get_feature_names()

        doc = 0
        feature_index = score_matrix[doc, :].nonzero()[1]

        tfidf_scores = zip(feature_index, [score_matrix[doc, x] for x in feature_index])

        for w, s in [(tfidf_names[i], s) for (i, s) in tfidf_scores]:
            print(w, s)
