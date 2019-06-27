from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV as GCV
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import *



class SvdFeature(object):

    def __init__ (self):
        self.svd_model = TruncatedSVD(n_components=100, random_state=42)
        self.normalizer = Normalizer(copy=False)
        #self.doc_list = []
        self.tf = TfidfVectorizer(strip_accents='unicode', stop_words='english', min_df=2, max_df=.5)
        # to reduce dimension min_df=.10, max_df=.75

    '''
        Function to first transform raw text from dataset into TF IDF matrix 
    '''

    def process_tfidf(self):
        doc_list = []
        label = []
        with open('data.json', mode="r") as f:
            data = json.load(f)
            for news in data:
               doc_list.append(news['title'] + ". " + news['body'])
               label.append(news['label'])

        tfidf_matrix = self.tf.fit_transform(doc_list)

        X = tfidf_matrix.toarray()
        # preprocessing.scale(X, with_mean=False)

        return X


    '''
        Function to use SVD (for Latent Semantic Analysis) to decompose the term-document matrix from Tf Idf
    '''
    def process_svd(self):

        # tfidf = TfidfFeature()
        tfidf_matrix = self.process_tfidf()

        svd_matrix = self.svd_model.fit_transform(tfidf_matrix)
        svd_matrix_df = pd.DataFrame(svd_matrix)
        svd_matrix_df["label"] = pd.read_json("data.json")["label"]
        svd_matrix_df.to_csv("svd_feature.csv", index=False)

        print(svd_matrix.shape)

    '''
        Function to read the results from SVD (and Tf Idf) and classify using logistic regression
    '''

    def read(self):

        df = pd.read_csv("svd_feature.csv")
        X = df.drop("label", axis=1)
        y = df["label"]

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


    '''
        Function to view vocab and their corresponding values 
    '''
    def get_tfidf_scores(self):


        score_matrix = self.process_tfidf()

        tfidf_names = self.tf.get_feature_names()

        doc = 0
        feature_index = score_matrix[doc, :].nonzero()[1]

        tfidf_scores = zip(feature_index, [score_matrix[doc, x] for x in feature_index])

        for w, s in [(tfidf_names[i], s) for (i, s) in tfidf_scores]:
            print(w, s)