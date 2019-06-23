from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
# from scipy.sparse.csr import csr_matrix
import json
from utils import *
from sklearn.preprocessing import MaxAbsScaler
from sklearn import svm
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


class TfidfFeature(object):

    def __init__(self):

        self.doc_list = []
        self.tf = TfidfVectorizer(strip_accents='unicode' ,stop_words='english')
        self.tfidf_feature = {}

    def process(self):
        doc_list = []
        label = []
        with open('data.json', mode="r") as f:
            data = json.load(f)
            for news in data:
               doc_list.append(news['title'] + ". " + news['body'])
               label.append(news['label'])



        X = doc_list
        y = label

        ss = MaxAbsScaler()

        
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)[:50]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)

        pd.DataFrame(X_train, y_train).to_csv("Training.csv")
        pd.DataFrame(X_test, y_test).to_csv("Testing.csv")
        # small_xtrain = X_train[:50]
        # small_xtest = X_test[:50]
        # small_ytrain = y_train[:50]
        #
        tfidf_matrix = self.tf.fit_transform(X_train)
        #tfidf_matrix = ss.fit(tfidf_matrix)


        testtf_matrix = self.tf.transform(X_test)
        #testtf_matrix = ss.fit(testtf_matrix)
        
        
        # decision tree should control size of trees due to large dataset
        # max_depth min_samples_leaf should play with 

        dtree = DecisionTreeClassifier(random_state=42)
        dtree.fit(tfidf_matrix, y_train)

        y_pred = dtree.predict(testtf_matrix)

        score = metrics.accuracy_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred, ["fake", "real"], pos_label="real")
        recall = metrics.recall_score(y_test, y_pred, ["fake", "real"], pos_label="real")
        f1 = metrics.f1_score(y_test, y_pred, ["fake", "real"], pos_label="real")
        print("accuracy: ", score)
        print("precision: ", precision)
        print("recall: ", recall)
        print("f1: ", f1)

        print(metrics.confusion_matrix(y_test, y_pred))

        # K nearest neighbors find a predefined number of training samples closest in distance to
        # the new point and predict the label from these, number of samples is user defined constant
        # simple majority vote of the nearest neighbors of each point
        # lower the dimensionality the better should try SVD
        # default k neighbors is 5

        knn = KNeighborsClassifier()
        knn.fit(tfidf_matrix, y_train)

        y_pred = knn.predict(testtf_matrix)

        score = metrics.accuracy_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred, ["fake", "real"], pos_label="real")
        recall = metrics.recall_score(y_test, y_pred, ["fake", "real"], pos_label="real")
        f1 = metrics.f1_score(y_test, y_pred, ["fake", "real"], pos_label="real")
        print("accuracy: ", score)
        print("precision: ", precision)
        print("recall: ", recall)
        print("f1: ", f1)

        print(metrics.confusion_matrix(y_test, y_pred))








        #SVM algs not scale invariant recommended to scale data
        #LinearSVC recommnded for out large number of samples
        #uses c as regularization parameter, kernel is rbf denoted by keyword gamma
        #set kernel cache size to 500MB or 1000MB if possible

        SVM = svm.LinearSVC(random_state=42)
        SVM.fit(tfidf_matrix, y_train)

        y_pred = SVM.predict(testtf_matrix)

        score = metrics.accuracy_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred, ["fake", "real"], pos_label="real")
        recall = metrics.recall_score(y_test, y_pred, ["fake", "real"], pos_label="real")
        f1 = metrics.f1_score(y_test, y_pred, ["fake", "real"], pos_label="real")
        print("accuracy: ", score)
        print("precision: ", precision)
        print("recall: ", recall)
        print("f1: ", f1)

        print(metrics.confusion_matrix(y_test, y_pred))

        #np.mean(y_pred)



        
        # The cost is proportional to the square of the value of the weight coefficients
        # penalty of l2 as regularization has no feature selection, computationally efficient, nonsparse output
        # L1 regularization inefficient on non-sparse cases, built in feature selection, sparse output
        # instantiate the model (default parameters)
        # default max iter is 100 examples showing 10000
        # saga handles l2 and is fast for large dataset also could use sag
        # multiclass is ovr (one vs rest) as we only need binary classification
        #
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
        #
        #
        print(metrics.confusion_matrix(y_test, y_pred))

        # tfidf_matrix = self.tf.fit_transform(doc_list)
        # will gives a list off all tokens or ngrams or words
        
        
        # assuming defaults of alpha = 1 (additive smoothing parameter,
        # fit_prior = True to learn class prior probabilities,
        # class_prior = None  the prior probabilites of the classes
        clf = MultinomialNB()
        clf.fit(tfidf_matrix, y_train)
        #
        # # X should be the tfidf matrix of news articles, y is the train target
        #
        #
        y_pred = clf.predict(testtf_matrix)
        #
        score = metrics.accuracy_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred, ["fake", "real"], pos_label="real")
        recall = metrics.recall_score(y_test, y_pred, ["fake", "real"], pos_label="real")
        f1 = metrics.f1_score(y_test, y_pred, ["fake", "real"], pos_label="real")
        print("accuracy: ", score)
        print("precision: ", precision)
        print("recall: ", recall)
        print("f1: ", f1)
        # print(tfidf_matrix.shape)
        print(metrics.confusion_matrix(y_test, y_pred))
        
        return tfidf_matrix



    def get_scores(self):


        score_matrix = self.process()

        tfidf_names = self.tf.get_feature_names()

        doc = 0
        feature_index = score_matrix[doc, :].nonzero()[1]

        tfidf_scores = zip(feature_index, [score_matrix[doc, x] for x in feature_index])

        for w, s in [(tfidf_names[i], s) for (i, s) in tfidf_scores]:
            print(w, s)
