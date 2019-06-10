from utils import NewsContent
from gensim.models import KeyedVectors
from gensim.models import Word2Vec

# sentences iterable
sentences = NewsContent(
    './fakenewsnet_dataset', 'politifact', 'fake')
model = Word2Vec(sentences, sg=1, size=100, workers=4, min_count=1)
word_vector = model.wv
print(len(model.wv.vocab))