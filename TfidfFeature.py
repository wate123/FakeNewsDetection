from sklearn.feature_extraction.text import TfidfVectorizer

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
            for title,body in data.items():
               doc_list.append(title + ". " + body)

        tfidf_matrix = self.tf.fit_transform(doc_list)
        # will gives a list off all tokens or ngrams or words

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
