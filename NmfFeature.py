from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import *
import os

class NmfFeature(object):
    """
    Class that builds a Tf-Idf model and uses SVD to compress the sparse matrix
    """

    def __init__(self, out_file_path, seed):
        """
        Initializer function
        """
        self.seed = seed
        self.normalizer = Normalizer(copy=False)
        self.tf = TfidfVectorizer(strip_accents='unicode', stop_words='english', min_df=2, max_df=.5)
        self.out_file_path = out_file_path
        self.datasetName = out_file_path.split("_")[0]
        # to reduce dimension min_df=.10, max_df=.75

    def process_tfidf(self):
        """
        Function to first transform raw text from dataset into TF IDF matrix
        """

        doc_list = []
        label = []
        data = pd.read_csv(self.out_file_path)
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
        Function to use NMF to decompose the term-document matrix from Tf Idf
        """

        print("Generating NMF feature")
        # tfidf = TfidfFeature()
        tfidf_matrix = self.process_tfidf()

        # 100 dimensions
        nmf_matrix = NMF(n_components = 100, random_state=self.seed).fit_transform(tfidf_matrix)

        nmf_matrix_df = pd.DataFrame(nmf_matrix)
        # svd_matrix_df = pd.DataFrame(svd_matrix)
        nmf_matrix_df["label"] = pd.read_csv(self.out_file_path)["label"]
        try:
            os.makedirs("./Features/"+self.datasetName)
        except OSError:
            pass
        nmf_matrix_df.to_csv("./Features/"+self.datasetName+"/nmf_feature.csv", index=False)
        # print(svd_matrix.shape)

        print("Done! Save into nmf_feature.csv")
        return {"NMF Feature Path": "./Features/"+self.datasetName+"/nmf_feature.csv"}

    def read(self):
        """
        Function to read the results from NMF (and Tf Idf) and classify using logistic regression
        """

        return pd.read_csv("./Features/"+self.datasetName+"/nmf_feature.csv").drop("label", axis=1)