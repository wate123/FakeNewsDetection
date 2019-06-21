from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence, PathLineSentences
from sklearn.preprocessing import normalize
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# model = KeyedVectors.load_word2vec_format(fname="./GoogleNews-vectors-negative300.bin", binary=True)
# print("model loaded")

'''
Class that builds a Word2Vec model (either pre-trained or not) pertaining to the news content 
'''

class Word2VecFeatureGenerator(object):

    """
    Initializer function that takes a corpus and specification of whether to use a pretrained model or not
    """
    
    def __init__(self, corpus, pretrain=False):
        
        # uses Google News Word2Vec model
        if pretrain:
            self.model = Word2Vec("./GoogleNews-vectors-negative300.bin", sg=1, size=100, workers=200, min_count=1)
            print("Google News pretrained model loaded")
        # to train own model
        else:
            self.model = Word2Vec(corpus, sg=1, size=100, workers=200, min_count=1)
            print("model loaded")
    
    def get_norm_vectors(self, words):
        """Function to retrieve normalized vectors"""
        vectors = []

        # Pre-compute L2-normalized vectors.
        self.model.wv.init_sims()
        # iterating through words, normalizing their vectors
        for word in words:
            # ignore all empty
            try:
                # saving only the normalized vectors to list
                vectors.append(self.model.wv.word_vec(word))
            except KeyError:
                pass
        return vectors

    def get_title_body_cos_sim(self, features):
        """
        Function to get cosine similarity between a title of article and its body content
        """

        # title_vec = []
        # body_vec = []
        sim_vec = []
        
        # iterating through news content's title and body
        for title, body in features:
            # obtain normalized vector of title and body
            title_vec = self.get_norm_vectors(title)
            body_vec = self.get_norm_vectors(body)
            # compare normalized vectors using cosine similarity
            sim_vec.append(cosine_similarity(title_vec, body_vec))

        return np.asarray(sim_vec)[:, np.newaxis]
