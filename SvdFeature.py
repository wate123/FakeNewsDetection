from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from TfidfFeature import TfidfFeature

class SvdFeature(object):

    def __init__ (self):
        self.svd_model = TruncatedSVD(n_components=100, random_state=42)
        self.normalizer = Normalizer(copy=False)

    '''
    Function to use SVD (for Latent Semantic Analysis) to decompose the term-document matrix 
    '''

    def process(self):

        tfidf = TfidfFeature()
        tfidf_matrix = tfidf.process()

        #self.svd_model.fit_transform(tfidf_matrix)

        # print(svd_matrix.shape)

        lsa = make_pipeline(self.svd_model, self.normalizer)
        lsa_res = lsa.fit_transform(tfidf_matrix)
        #print(lsa_res)

        #need to use lsa_res for models like kmeans


    def svd_results(self):

        # topics
        # print(self.svd_model.components_)
        # print(self.svd_model.singular_values_)
        print("Sum of explained variance ratio")
        print(self.svd_model.explained_variance_ratio_.sum() )
        print("------------------------------------------------------------------------")

        # the percentage of variance explained by each of the selected topics (components)
        print("Explained Variance Ratio")
        print(self.svd_model.explained_variance_ratio_)

        print("-----------------------------------------------------------------\n", "Explained Variance")
        #the variance of the samples transformed by a projection to each topic (component)
        print(self.svd_model.explained_variance_)
