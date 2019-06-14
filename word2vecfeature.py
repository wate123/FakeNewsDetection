from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence, PathLineSentences
from sklearn.preprocessing import normalize
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# model = KeyedVectors.load_word2vec_format(fname="./GoogleNews-vectors-negative300.bin", binary=True)
# print("model loaded")


class Word2VecFeatureGenerator():
    def __init__(self, corpus, pretrain=False):
        if pretrain:
            self.model = Word2Vec("./GoogleNews-vectors-negative300.bin", sg=1, size=100, workers=200, min_count=1)
        else:
            self.model = Word2Vec(corpus, sg=1, size=100, workers=200, min_count=1)

    def get_norm_vectors(self, words):
        # assert type(feature_ls) == LineSentence, "Only LineSentence"
        vectors = []
        if len(words) == 0:
            return None
        else:
            for word in words:
                try:
                    # print(self.model.wv.word_vec(word, use_norm=True))
                    self.model.init_sims()
                    vectors.append(self.model.wv.word_vec(word))
                except KeyError:
                    pass
        return vectors

    def get_title_body_cos_sim(self, features):
        title_vec = None
        body_vec = None
        for title, body in features:
            title_vec = self.get_norm_vectors(title)
            body_vec = self.get_norm_vectors(body)

        sim_vec = cosine_similarity(title_vec, body_vec)
        return sim_vec
