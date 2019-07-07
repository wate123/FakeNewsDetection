from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.preprocessing import Normalizer
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV as GCV
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import *
from sklearn.decomposition import NMF

class SvdFeature(object):
    """
    Class that builds a Tf-Idf model and uses SVD to compress the sparse matrix
    """

    def __init__(self):
        """
        Initializer function that creates SVD model with 100 dimensions and a Tf-Idf Vectorizer
        """

        self.svd_model = TruncatedSVD(n_components=100, random_state=1)
        self.normalizer = Normalizer(copy=False)
        self.tf = TfidfVectorizer(strip_accents='unicode', stop_words='english', min_df=2, max_df=.5)
        # to reduce dimension min_df=.10, max_df=.75

    def process_tfidf(self):
        """
        Function to first transform raw text from dataset into TF IDF matrix
        """

        doc_list = []
        label = []
        data = pd.read_csv("data.csv")
        doc_list = data['title'].map(str) + data['body']
        # with open('data.json', mode="r") as f:
        #     data = json.load(f)
        #     for news in data:
        #        doc_list.append(news['title'] + ". " + news['body'])
        #        label.append(news['label'])

        tfidf_matrix = self.tf.fit_transform(doc_list.values)

        X = tfidf_matrix.toarray()

        return X

    def process_and_save(self):
        """
        Function to use SVD (for Latent Semantic Analysis) to decompose the term-document matrix from Tf Idf
        """

        print("Generating SVD feature")
        # tfidf = TfidfFeature()
        tfidf_matrix = self.process_tfidf()


        nmf_matrix = NMF(n_components = 100, random_state=1).fit_transform(tfidf_matrix)
        svd_matrix = self.svd_model.fit_transform(tfidf_matrix)

        svd_matrix_df = pd.DataFrame(nmf_matrix)
        # svd_matrix_df = pd.DataFrame(svd_matrix)
        svd_matrix_df["label"] = pd.read_csv("data.csv")["label"]
        svd_matrix_df.to_csv("svd_feature.csv", index=False)
        # print(svd_matrix.shape)

        print("Done! Save into svd_featur.csv")

    def read(self):
        """
        Function to read the results from SVD (and Tf Idf) and classify using logistic regression
        """

        return pd.read_csv("svd_feature.csv").drop("label", axis=1)
        # X = df.drop("label", axis=1)
        # y = df["label"]

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)[:50]
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)

        # The cost is proportional to the square of the value of the weight coefficients
        # penalty of l2 as regularization has no feature selection, computationally efficient, nonsparse output
        # L1 regularization inefficient on non-sparse cases, built in feature selection, sparse output
        # instantiate the model (default parameters)
        # default max iter is 100 examples showing 10000
        # saga handles l2 and is fast for large dataset also could use sag
        # multiclass is ovr (one vs rest) as we only need binary classification
        #

        # wparameters = {'C': [4], 'penalty': ['l2'], 'tol': [5e-4], 'random_state': [1], 'multi_class': ['multinomial'],
        #                'max_iter': [1000], 'solver': ['newton-cg']}
        #
        # logreg = LogisticRegression()
        #
        # clf = GCV(logreg, wparameters, cv=10, n_jobs=-1)
        #
        # #logreg.fit(X_train, y_train)
        # clf.fit(X_train, y_train)
        #
        # # y_pred = logreg.predict(X_test)
        # y_pred = clf.predict(X_test)
        #
        # print(metrics.confusion_matrix(y_test, y_pred))
        # print(metrics.classification_report(y_test, y_pred))

    def get_tfidf_scores(self):
        """
        Function to view vocab and their corresponding values
        """

        score_matrix = self.process_tfidf()

        tfidf_names = self.tf.get_feature_names()

        doc = 0
        feature_index = score_matrix[doc, :].nonzero()[1]

        tfidf_scores = zip(feature_index, [score_matrix[doc, x] for x in feature_index])

        for w, s in [(tfidf_names[i], s) for (i, s) in tfidf_scores]:
            print(w, s)
