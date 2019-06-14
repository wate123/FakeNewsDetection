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

class Word2VecFeatureGenerator():
    
    '''
    Initializer function that takes a corpus and specification of whether to use a pretrained model or not
    '''
    
    def __init__(self, corpus, pretrain=False):
        
        # uses Google News Word2Vec model
        if pretrain:
            self.model = Word2Vec("./GoogleNews-vectors-negative300.bin", sg=1, size=100, workers=200, min_count=1)
            
        # to train own model    
        else:
            self.model = Word2Vec(corpus, sg=1, size=100, workers=200, min_count=1)

    '''
    Function to retrieve normalized vectors 
    '''
    
    def get_norm_vectors(self, words):
        # assert type(feature_ls) == LineSentence, "Only LineSentence"
        vectors = []
        
        # if empty vector no normalization needed naturally
        if len(words) == 0:
            return None
        else:
            # iteraing through words, normalizing their vectors
            for word in words:
                try:
                    # print(self.model.wv.word_vec(word, use_norm=True))
                    self.model.init_sims()
                    
                    # saving only the normalized vectors to list
                    vectors.append(self.model.wv.word_vec(word))
                except KeyError:
                    pass
        return vectors

    '''
    Function to get results of comparison between a title of article and its body content 
    '''
    
    def get_title_body_cos_sim(self, features):
        title_vec = None
        body_vec = None
        
        # iterating through news content's title and body
        for title, body in features:
            
            # obtain normalized vector of title and body
            title_vec = self.get_norm_vectors(title)
            body_vec = self.get_norm_vectors(body)

        # compare normalized vectors using cosine similarity
        sim_vec = cosine_similarity(title_vec, body_vec)
        return sim_vec
